"""event_export's pure pieces: typed times, request ranges, names, the
frequency-ceiling bounds, event clipping, the CSV and image writers -- plus a
guard that fails loudly if a private pipeline helper it calls changes shape."""
import inspect
import os
import sys
from datetime import date, datetime, time, timedelta

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import event_export as ee
import fft_excel_export
import ffttest
from short_interval import ShortIntervalPlotter
try:
	from spectro.data_intervals import Burst, DataInterval, merge_bursts
except ImportError:
	from src.spectro.data_intervals import Burst, DataInterval, merge_bursts

D23, D24 = date(2021, 2, 23), date(2021, 2, 24)
SETTINGS = f'pad{ee.PAD:g}s_bins{ee.NUM_BINS}'


@pytest.mark.parametrize('text, expected', [
	('', None), ('   ', None),
	('9:05', time(9, 5)), ('09:05', time(9, 5)), ('09:05:03', time(9, 5, 3)),
	('09:05:03.5', time(9, 5, 3, 500_000)), ('09:05:03.25', time(9, 5, 3, 250_000)),
	('23:59:59.99', time(23, 59, 59, 990_000)),
])
def test_parse_time_text_accepts(text, expected):
	assert ee.parse_time_text(text) == expected


@pytest.mark.parametrize('text', ['24:00', '09:60', '09:05:60', '09:05:03.123', '9', '09:5', 'noon', '09:05:'])
def test_parse_time_text_rejects(text):
	with pytest.raises(ValueError):
		ee.parse_time_text(text)


def test_blank_end_time_stores_next_midnight_so_the_end_date_is_included():
	rng = ee.resolve_range(D23, None, D24, None)
	assert rng.start == datetime(2021, 2, 23)
	assert rng.end == datetime(2021, 2, 25)
	assert rng.whole_days and rng.last_day == D24


def test_typed_end_midnight_is_taken_literally():
	rng = ee.resolve_range(D23, None, D24, time(0))
	assert rng.end == datetime(2021, 2, 24)
	assert rng.whole_days and rng.last_day == D23


def test_single_whole_day():
	rng = ee.resolve_range(D23, None, D23, None)
	assert (rng.start, rng.end, rng.last_day) == (datetime(2021, 2, 23), datetime(2021, 2, 24), D23)


def test_typed_start_with_blank_end():
	rng = ee.resolve_range(D23, time(10, 30), D24, None)
	assert rng.end == datetime(2021, 2, 25) and not rng.whole_days


@pytest.mark.parametrize('args', [
	(D24, None, D23, None),
	(D23, None, D23, time(0)),
	(D23, time(12), D23, time(11, 59)),
])
def test_empty_or_backwards_range_is_rejected(args):
	with pytest.raises(ValueError):
		ee.resolve_range(*args)


def test_name_ts_pads_every_place_and_truncates_to_hundredths():
	assert ee.name_ts(datetime(2021, 2, 3, 4, 5, 6, 999_999)) == '2021-02-03-04H-05M-06.99S'
	assert ee.name_ts(datetime(2021, 2, 23, 21, 24, 55, 50_869)) == '2021-02-23-21H-24M-55.05S'
	assert ee.name_ts(datetime(2021, 2, 23)) == '2021-02-23-00H-00M-00.00S'


def test_event_folder_name():
	name = ee.event_folder_name(datetime(2021, 2, 23, 21, 24, 55, 50_869),
	                            datetime(2021, 2, 23, 21, 53, 0, 995_586))
	assert name == '2021-02-23-21H-24M-55.05S_2021-02-23-21H-53M-00.99S'


def test_top_folder_whole_days_names_first_and_last_included_day():
	rng = ee.resolve_range(D23, None, D24, None)
	assert ee.top_folder_name(rng, ('B', 'E'), 3000, False) == f'2021-02-23_2021-02-24_BE_{SETTINGS}_lpf3000Hz'
	assert ee.top_folder_name(rng, ('E', 'B'), 3000, True) == f'2021-02-23_2021-02-24_BE_{SETTINGS}_lpf3000Hz_fce'


def test_top_folder_with_a_time_names_both_ends_with_the_stored_end():
	rng = ee.resolve_range(D23, time(10), D24, None)
	assert ee.top_folder_name(rng, ('B',), 4096, False) == \
		f'2021-02-23-10H-00M-00.00S_2021-02-25-00H-00M-00.00S_B_{SETTINGS}_lpf4096Hz'


def test_top_folder_typed_end_midnight_names_last_included_day():
	rng = ee.resolve_range(D23, None, D24, time(0))
	assert ee.top_folder_name(rng, ('E',), 3000, False) == f'2021-02-23_2021-02-23_E_{SETTINGS}_lpf3000Hz'


def test_fields_suffix_needs_a_field():
	with pytest.raises(ValueError):
		ee.fields_suffix(())


def test_lpf_bounds_reach_the_nyquist_row():
	# PAD 0.25 and SAMPLE_RATE 8192: Df = 2 Hz, Nyquist row at 4096 Hz -- and the
	# ceiling now reaches it. It stopped at 4094 while _compute_file_ffts still
	# doubled an even window's Nyquist bin, because that bin was overstated 2x;
	# the doubling is gone (DC and Nyquist are both left undoubled), so the row
	# is ordinary correctly-scaled data.
	assert ee.lpf_bounds() == (100, 4096)


def test_lpf_bounds_follow_the_sample_rate():
	# The bound is derived from fs, not hardcoded. SCM high burst samples at
	# twice the rate, so its Nyquist row -- and therefore the highest ceiling
	# worth offering -- sits at twice the frequency.
	from ffttest import BurstRate
	assert ee.lpf_bounds(fs=BurstRate.SCB) == (100, 4096)
	assert ee.lpf_bounds(fs=BurstRate.SCHB) == (100, 8192)


def test_clip_bursts_keeps_inside_clips_edges_and_drops_outside():
	rng = ee.resolve_range(D23, time(10), D23, time(12))
	at = lambda h, m: datetime(2021, 2, 23, h, m)
	bursts = [Burst(at(10, 30), at(10, 40), 1), Burst(at(9, 58), at(10, 5), 1),
	          Burst(at(11, 55), at(12, 10), 1), Burst(at(12, 30), at(12, 40), 1)]
	events = ee.clip_bursts(bursts, rng, 3, 'B')
	assert [(e.start, e.end, e.clipped_start, e.clipped_end) for e in events] == [
		(at(10, 30), at(10, 40), False, False),
		(at(10, 0), at(10, 5), True, False),
		(at(11, 55), at(12, 0), False, True),
	]
	assert all(e.probe == 3 and e.field == 'B' for e in events)


def test_range_edge_inside_a_dropout_still_clips_the_event_to_the_edge():
	rng = ee.resolve_range(D23, time(10), D23, time(12))
	ivs = [DataInterval(datetime(2021, 2, 23, 9, 59, 50), datetime(2021, 2, 23, 9, 59, 59, 500_000), 'a'),
	       DataInterval(datetime(2021, 2, 23, 10, 0, 0, 800_000), datetime(2021, 2, 23, 10, 1), 'a')]
	(ev,) = ee.clip_bursts(merge_bursts(ivs, tol=timedelta(seconds=ee.EVENT_GAP_S)), rng, 1, 'E')
	assert ev.start == rng.start and ev.clipped_start and not ev.clipped_end


@pytest.mark.parametrize('layout', ['tall', 'wide'])
def test_event_csv_time_headers_are_unique_hundredths(tmp_path, layout):
	start = datetime(2021, 2, 23, 21, 24, 55, 50_869)
	# The empty-cell marker is taken from the one place that owns it, not written
	# out as a literal here. It used to be the same number as the colour-scale
	# floor, and hardcoding that floor is exactly what stops being correct once
	# the scale is something the user picks.
	n_time, n_chan = 7, 3
	mean_g = np.full((n_time, n_chan), ee.empty_fill_for('B'))
	mean_g[[0, 1, 2, 4, 5, 6]] = 1e-3                # row 3 has no data and is dropped
	path = ee.write_event_csv(str(tmp_path / 'ffts.csv'), start, np.arange(n_time + 1) * 0.05,
	                          mean_g, 'B', n_chan, layout)
	df = pd.read_csv(path)
	stamps = list(df.columns[1:-2]) if layout == 'wide' else list(df['time_utc'][:-2])
	assert stamps == ['2021-02-23 21:24:55.05', '2021-02-23 21:24:55.10', '2021-02-23 21:24:55.15',
	                  '2021-02-23 21:24:55.25', '2021-02-23 21:24:55.30', '2021-02-23 21:24:55.35']


def test_event_png_is_written_with_and_without_fce(tmp_path):
	bins, n_chan = ShortIntervalPlotter._analytic_log_bins(ee.PAD, 3000, ee.NUM_BINS)
	n_time = 40
	mean_g = 10 ** np.random.default_rng(0).uniform(-6, -2, size=(n_time, n_chan))
	binned = (mean_g, np.arange(n_time + 1) * 0.05, np.arange(n_chan + 1), bins, n_chan, 1e-6, 1e-2, 0.05)
	start = datetime(2021, 2, 23, 21, 24, 55, 50_869)
	ev = ee.Event(1, 'B', start, start + timedelta(seconds=2), False, False)
	times = np.datetime64(start, 'ns') + np.arange(20) * np.timedelta64(100, 'ms')
	plotter = ShortIntervalPlotter(data_dir='unused', pickle_root='unused')
	for fce in (None, (times, np.full(20, 500.0))):
		path = ee.draw_event_png(str(tmp_path / 'graph.png'), plotter, ev, binned, fce)
		assert os.path.getsize(path) > 10_000


@pytest.mark.parametrize('fn, params', [
	# The ceiling/bin-count/cell-width trio now travels as one BinSpec, so what
	# event_export passes is `binning` -- see spectro.specs.BinSpec.
	(ShortIntervalPlotter._bin_fft_files,
	 ['fft_paths', 'key', 'pad', 'start', 'end', 'binning', 'step']),
	(ShortIntervalPlotter._draw_onto,
	 ['ax', 'mean_g', 'time_bins', 'ye', 'bins', 'n_chan', 'vmi', 'vma', 'key', 'pad', 'start',
	  'display_dt', 'is_top_row', 'is_bottom_row', 'is_left_col', 'probe', 'fce']),
	(ShortIntervalPlotter._write_bins_readme_for, ['bins_readme_dir', 'key', 'pad', 'lpf_lim', 'num_bins', 'step']),
	(fft_excel_export._probe_dataframe, ['t_bins', 'mean', 'p_cnt', 'st', 'n_chan', 'with_date', 'layout']),
	(ffttest._plan_fft_jobs, ['data_dir', 'pickle_root', 'specs', 'download']),
	(ffttest._fft_pickle_path, ['pickle_root', 'inst', 'mode', 'pad', 'file_time', 'probe']),
	(ffttest._segments_via_sidecar, ['cdf_path', 'key', 'fs', 'probe', 'pickle_root']),
	(ffttest.find_local_cdfs, ['data_dir', 'inst', 'mode', 'start', 'end', 'probe', 'pickle_root']),
	(ffttest._scan_cdf_files, ['data_dir', 'inst', 'mode', 'probe']),
	(ffttest._bounds_key, ['inst']),
	(ffttest._parse_cdf_time, ['path']),
])
def test_private_helpers_still_take_what_event_export_passes(fn, params):
	have = inspect.signature(fn).parameters
	assert not [p for p in params if p not in have], f'{fn.__qualname__} changed its parameters'


def _touch_cdf(root, probe, inst, mode, stamp):
	parts = [str(root), f'mms{probe}', inst, 'brst', 'l2'] + ([mode] if mode else [])
	folder = os.path.join(*parts, stamp[:4], stamp[4:6], stamp[6:8])
	os.makedirs(folder, exist_ok=True)
	path = os.path.join(folder, f'mms{probe}_{inst}_brst_l2_{mode + "_" if mode else ""}{stamp}_v1.0.0.cdf')
	open(path, 'w').close()
	return path


def test_count_local_files_ignores_a_stale_file_before_the_range(tmp_path, monkeypatch):
	ends = {}
	monkeypatch.setattr(ee, 'DATA_DIR', str(tmp_path))
	# The files here are empty stand-ins whose bounds are faked, so validation is
	# faked with them. PICKLE_ROOT too: the real one is the data drive.
	monkeypatch.setattr(ee, 'PICKLE_ROOT', str(tmp_path / 'cache'))
	monkeypatch.setattr(ee, 'cdf_check', lambda *a, **k: (lambda paths: list(paths)))
	monkeypatch.setattr(ee, 'cdf_time_bounds', lambda path, key, probe: (None, ends[os.path.basename(path)]))
	rng = ee.resolve_range(D23, time(10, 30), D23, time(11, 10))

	stale = _touch_cdf(tmp_path, 1, 'scm', 'scb', '20210220120000')
	ends[os.path.basename(stale)] = datetime(2021, 2, 20, 12, 5)
	assert ee.count_local_files('B', 1, rng.start, rng.end) == 0    # find_cdf_files(download=False) would return it

	anchor = _touch_cdf(tmp_path, 1, 'scm', 'scb', '20210223102835')
	ends[os.path.basename(anchor)] = datetime(2021, 2, 23, 10, 31, 50)
	_touch_cdf(tmp_path, 1, 'scm', 'scb', '20210223103635')
	_touch_cdf(tmp_path, 1, 'scm', 'scb', '20210223111000')          # starts at the excluded end
	assert ee.count_local_files('B', 1, rng.start, rng.end) == 2
	assert ee.count_local_files('E', 1, rng.start, rng.end) == 0


def test_download_missing_acquires_each_instrument_over_its_probes(monkeypatch):
	"""One acquisition.acquire per instrument, over the probes missing it; acquire
	then makes one paced SDC call per probe (Graham, 2026-09-28)."""
	calls = []
	monkeypatch.setattr(ee, 'acquire', lambda letter, probes, start, end, **kw: calls.append(
		(letter, probes, kw.get('download', True), kw['cancelled'])))
	rng = ee.resolve_range(D23, None, D23, None)
	stop = lambda: False
	ee.download_missing(rng, [(3, 'B'), (1, 'B'), (1, 'F')], log=lambda *_: None, cancelled=stop)
	assert calls == [('B', [1, 3], True, stop), ('F', [1], True, stop)]
