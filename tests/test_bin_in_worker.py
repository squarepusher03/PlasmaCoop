"""
Binning in the FFT workers (2026-11-23, Graham: "Get rid of the binning scratch"; and on quitting: "the program
should wait for the workers to finish, delete their output, then cancel mid run").

prepare_export no longer writes FFT pickles for export_event and the event images to read back: each pool worker
computes its file's FFTs in memory and returns _FftGrid.partial of them for every window the file feeds, and the
parent folds those, in each window's file order, with the same += _FftGrid.add does. These tests hold that to the
old path bit for bit, and check the stoppable pool (ffttest._pool_imap) behind it. Synthetic data in tmp_path.

    python -m pytest -q tests/test_bin_in_worker.py
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLBACKEND', 'Agg')

import event_export as ee
import ffttest
from short_interval import ShortIntervalPlotter, _FftGrid

try:
	from spectro.specs import BinSpec
except ImportError:
	from src.spectro.specs import BinSpec


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
	monkeypatch.setattr(ffttest, '_fft_scratch_roots', set())
	monkeypatch.setattr(ffttest, '_fft_scratch_users', {})


def _grid(start, seconds, lpf=3000):
	binning = BinSpec(lpf_lim=lpf, num_bins=ee.NUM_BINS, display_dt=None)
	bins, n_chan = ShortIntervalPlotter._analytic_log_bins(ee.PAD, lpf, ee.NUM_BINS)
	return _FftGrid('B', ee.PAD, start, start + pd.Timedelta(seconds=seconds).to_pytimedelta(), binning, ee.STEP,
	                bins, n_chan)


def _frame(rng, start, t0, t1):
	"""FFT rows (time, frequency, power) every 0.05 s over [t0, t1) s after start, every 2 Hz to 4096 Hz."""
	times = np.datetime64(start, 'ns') + (np.arange(t0, t1, 0.05) * 1e9).astype('timedelta64[ns]')
	freqs = np.arange(0, 4097, 2)
	t = np.repeat(times, len(freqs))
	f = np.tile(freqs, len(times))
	return pd.DataFrame({'time': t, 'frequency': f, 'power': 10 ** rng.uniform(-8, -1, size=len(t))})


def _same(a, b):
	assert (a is None) == (b is None)
	if a is None:
		return
	for x, y in zip(a, b):
		if isinstance(x, np.ndarray):
			assert x.dtype == y.dtype and np.array_equal(x, y)
		else:
			assert x == y


def test_partials_folded_in_order_are_add_bit_for_bit():
	from datetime import datetime
	start = datetime(2021, 2, 23, 10, 0, 0)
	rng = np.random.default_rng(7)
	# overlapping frames, as two files' seam windows overlap, plus one outside the window and one empty
	frames = [_frame(rng, start, 0, 4), _frame(rng, start, 3, 7), _frame(rng, start, 6.5, 12),
	          _frame(rng, start, 20, 22), pd.DataFrame(columns=['time', 'frequency', 'power'])]
	old, new = _grid(start, 10), _grid(start, 10)
	for fdf in frames:
		old.add(fdf)
	worker = _grid(start, 10)
	parts = [worker.partial(fdf) for fdf in frames]
	assert worker.p_sum is None                         # a worker's grid never allocates the window
	assert parts[3] is None and parts[4] is None        # nothing lands
	for part in parts:
		new.add_partial(part)
	_same(old.result(), new.result())


def test_a_grid_nothing_landed_in_is_none():
	from datetime import datetime
	g = _grid(datetime(2021, 2, 23), 5)
	g.add_partial(None)
	assert g.result() is None and g.p_sum is None


def _square_slowly(job):
	x, folder = job
	open(os.path.join(folder, f'started_{x}'), 'w').close()
	time.sleep(0.15)
	return x * x


def test_pool_imap_yields_in_job_order(tmp_path):
	jobs = [(x, str(tmp_path)) for x in range(9)]
	assert list(ffttest._pool_imap(_square_slowly, jobs, 3, 'test')) == [x * x for x in range(9)]
	assert multiprocessing.active_children() == []


def test_pool_imap_stops_when_cancelled_and_leaves_no_worker(tmp_path):
	jobs = [(x, str(tmp_path)) for x in range(40)]
	got = []
	for r in ffttest._pool_imap(_square_slowly, jobs, 2, 'test', cancelled=lambda: len(got) >= 3, ahead=4):
		got.append(r)
	assert got == [0, 1, 4]
	assert multiprocessing.active_children() == []     # the pool was shut down and its workers joined
	started = len([n for n in os.listdir(tmp_path) if n.startswith('started_')])
	# 3 consumed + at most `ahead` submitted beyond them + the one the pool's call queue may already hold
	assert 3 <= started <= 3 + 4 + 1 < len(jobs)


def _old_path_grids(settings, windows):
	"""What export_event / draw_image binned before 2026-11-23: prewarm's pickles, read by _bin_fft_files."""
	ffttest.prewarm(ee.DATA_DIR, ee.PICKLE_ROOT,
	                [(ee.field_spec(f)['sig_key'], p, ee.PAD, settings.lpf_hz, w) for (f, p), w in windows.items()],
	                write_manifest=False, step=ee.STEP, max_workers=2)
	plotter = ShortIntervalPlotter(data_dir=ee.DATA_DIR, pickle_root=ee.PICKLE_ROOT)
	out = {}
	for (f, p), wins in windows.items():
		for st, ts, _s in wins:
			out[(f, p, st, ts)] = plotter._bin_fft_files(
				fft_paths=ee._event_fft_paths(ee.Event(p, f, st, ts, False, False)), key=f, pad=ee.PAD, start=st,
				end=ts, binning=BinSpec(lpf_lim=settings.lpf_hz, num_bins=ee.NUM_BINS, display_dt=None),
				step=ee.STEP, scale=settings.scale_for(f))
	return out


def _pickles(root):
	return [os.path.join(r, n) for r, _d, names in os.walk(root) for n in names if n.endswith('.pkl')]


def test_prepare_export_bins_what_the_pickles_gave_and_writes_none(tmp_path, monkeypatch):
	pytest.importorskip('spacepy.pycdf')
	from test_direct_cdf import write_cdf, T
	data = str(tmp_path / 'pydata')
	write_cdf(data, 'B', '20210223100000', T(0), 32800)
	write_cdf(data, 'B', '20210223100004', T(0), 32736, first=32800)     # a seam inside the events
	write_cdf(data, 'E', '20210223100000', T(0), 40000)
	monkeypatch.setattr(ee, 'DATA_DIR', data)
	monkeypatch.setattr(ffttest, 'MAX_POOL_WORKERS', 2)
	rng = ee.RequestRange(T(0), T(10))
	settings = ee.ExportSettings(root=str(tmp_path / 'out'), rng=rng, fields=('B', 'E'), lpf_hz=3000, fce=False,
	                             layout='tall', scales=None, csv=True, graph=False)
	events = [ee.Event(1, 'B', T(0.5), T(6.5), False, False), ee.Event(1, 'B', T(7), T(7.2), False, False),
	          ee.Event(1, 'E', T(1), T(4.75), False, False)]
	extra = {('B', 1): [(T(0), T(8), False)], ('E', 1): [(T(0.5), T(6.5), False)]}
	windows = {}
	for ev in events:
		windows.setdefault((ev.field, ev.probe), []).append((ev.start, ev.end, False))
	for pair, more in extra.items():
		windows.setdefault(pair, []).extend(more)

	new_root = str(tmp_path / 'cache_new')
	monkeypatch.setattr(ee, 'PICKLE_ROOT', new_root)
	seen = []
	store = ee.prepare_export(settings, events, log=lambda *_: None, extra_windows=extra,
	                          progress=lambda d, n: seen.append((d, n)))
	assert _pickles(new_root) == []                     # no binning scratch
	assert store.complete and ee.prepared_ffts() is store
	assert seen[0] == (0, 3) and seen[-1] == (3, 3)     # three files, B's two and E's one

	monkeypatch.setattr(ee, 'PICKLE_ROOT', str(tmp_path / 'cache_old'))
	old = _old_path_grids(settings, windows)
	assert len(old) == len(store) == 5
	assert any(g is not None for g in old.values())
	for key, grid in old.items():
		_same(grid, store.get(*key))

	# and export_event writes from it
	ee.export_event(settings, events[0], log=lambda *_: None)
	assert os.path.exists(os.path.join(ee.event_dir(settings.top_dir, events[0]), ee.CSV_NAME))


def test_prepare_export_cancelled_is_incomplete(tmp_path, monkeypatch):
	pytest.importorskip('spacepy.pycdf')
	from test_direct_cdf import write_cdf, T
	data = str(tmp_path / 'pydata')
	write_cdf(data, 'B', '20210223100000', T(0), 32800)
	write_cdf(data, 'B', '20210223100004', T(0), 32736, first=32800)
	monkeypatch.setattr(ee, 'DATA_DIR', data)
	monkeypatch.setattr(ee, 'PICKLE_ROOT', str(tmp_path / 'cache'))
	monkeypatch.setattr(ffttest, 'MAX_POOL_WORKERS', 2)
	rng = ee.RequestRange(T(0), T(10))
	settings = ee.ExportSettings(root=str(tmp_path / 'out'), rng=rng, fields=('B',), lpf_hz=3000, fce=False,
	                             layout='tall', scales=None, csv=True, graph=False)
	ev = ee.Event(1, 'B', T(0.5), T(6.5), False, False)
	store = ee.prepare_export(settings, [ev], log=lambda *_: None, cancelled=lambda: True)
	assert not store.complete and not store.has('B', 1, ev.start, ev.end)
	assert multiprocessing.active_children() == []
	with pytest.raises(RuntimeError, match='no FFTs were prepared'):
		ee.export_event(settings, ev, log=lambda *_: None)
	ee.forget_prepared()
	with pytest.raises(RuntimeError, match='call prepare_export first'):
		ee.export_event(settings, ev, log=lambda *_: None)


def _fail_on_three(job):
	x, _folder = job
	if x == 3:
		raise ValueError('job 3 failed')
	time.sleep(0.05)
	return x


def test_pool_imap_a_failing_job_raises_and_leaves_no_worker(tmp_path):
	got = []
	with pytest.raises(ValueError, match='job 3 failed'):
		for r in ffttest._pool_imap(_fail_on_three, [(x, str(tmp_path)) for x in range(20)], 2, 'test'):
			got.append(r)
	assert got == [0, 1, 2]
	assert multiprocessing.active_children() == []
