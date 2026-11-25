"""The pipeline reads burst CDFs directly; the data pickles are gone from it.
Everything they used to give has to come out the same:

  * read_burst_data returns the frame the data pickle held, time in ns;
  * its seam read (until_ns) holds every sample before until_ns, in the same rows,
    and falls back to the whole file when the head stops short;
  * _segments_from_cdf gives _segments_from_pickle's runs, and _good_time_bounds
    the pickle's bounds;
  * find_local_cdfs picks the files find_data_pickles picked, anchor included;
  * FFTs computed from CDFs equal FFTs computed from data pickles, across the
    seam into the next file;
  * prewarm computes from CDFs and writes no data pickle;
  * check_cdfs deletes and logs an unreadable CDF, as ensure_data_pickle did;
  * the good-run sidecar is reused, and goes stale with its CDF or its version;
  * f_ce comes from the FGM CDFs, unchanged.

The old side of every comparison is the real old code (ensure_data_pickle,
find_data_pickles, _segments_from_pickle), run on the same synthetic CDFs --
files at the real paths and names, holding just the variables the pipeline
reads, written with spacepy (the `plasma` env).
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLBACKEND', 'Agg')

pycdf = pytest.importorskip('spacepy.pycdf')

import ffttest
from ffttest import SAMPLE_RATE

STEP_NS = 122_072                # the real scb/dce sample step (8191.9 Hz)
FGM_STEP_NS = 7_812_500          # 128 Hz
INSTRUMENTS = {'B': ('scm', 'scb'), 'E': ('edp', 'dce'), 'F': ('fgm', '')}


def T(seconds):
	return datetime(2021, 2, 23, 10, 0, 0) + timedelta(seconds=seconds)


def write_cdf(data_dir, key, stamp, start, n, probe=1, step_ns=STEP_NS, first=0, bad=()):
	"""A burst CDF at the real path and name. Sample i (from `first`) is at
	start + i * step_ns plus a sub-microsecond offset, as real times carry; the
	sample slices in `bad` are flagged quality-bad. Returns the path."""
	inst, mode = INSTRUMENTS[key]
	folder = os.path.join(data_dir, f'mms{probe}', inst, 'brst', 'l2', *([mode] if mode else []),
	                      stamp[:4], stamp[4:6], stamp[6:8])
	os.makedirs(folder, exist_ok=True)
	path = os.path.join(folder, f'mms{probe}_{inst}_brst_l2_{mode + "_" if mode else ""}{stamp}_v1.0.0.cdf')
	raw = (int(pycdf.lib.datetime_to_tt2000(start)) + 45_175
	       + (first + np.arange(n, dtype=np.int64)) * step_ns)
	good = np.ones(n, dtype=bool)
	for sl in bad:
		good[sl] = False
	rng = np.random.default_rng(int(stamp) % 2**32 + first)
	pycdf.lib.set_backward(False)
	with pycdf.CDF(path, '') as cdf:
		epoch = f'mms{probe}_edp_epoch_brst_l2' if key == 'E' else 'Epoch'
		cdf.new(epoch, type=pycdf.const.CDF_TIME_TT2000)
		cdf.raw_var(epoch)[:] = raw
		if key == 'B':
			cdf.new(f'mms{probe}_scm_acb_gse_scb_brst_l2', data=rng.normal(size=(n, 3)).astype(np.float32),
			        type=pycdf.const.CDF_FLOAT)
			cdf.new(f'mms{probe}_scm_qf_scm123_scb_brst_l2', data=np.where(good, 'GGG', 'ZZZ'),
			        type=pycdf.const.CDF_CHAR)
		elif key == 'E':
			cdf.new(f'mms{probe}_edp_dce_gse_brst_l2', data=rng.normal(size=(n, 3)).astype(np.float32),
			        type=pycdf.const.CDF_FLOAT)
			tier = np.arange(n) % 2              # good samples alternate tiers 2 and 3, bad ones 0 and 1
			cdf.new(f'mms{probe}_edp_quality_brst_l2', data=np.where(good, 2 + tier, tier).astype(np.uint8),
			        type=pycdf.const.CDF_UINT1)
		else:
			cdf.new(f'mms{probe}_fgm_b_gse_brst_l2', data=rng.normal(size=(n, 4)).astype(np.float32),
			        type=pycdf.const.CDF_FLOAT)
	return path


def pickles_under(root):
	return [os.path.join(r, f) for r, _d, files in os.walk(root) for f in files if f.endswith('.pkl')]


@pytest.mark.parametrize('name, parsed', [
	('mms1_scm_brst_l2_scb_20210223032923_v2.2.0.cdf', (1, 'scm', 'scb', datetime(2021, 2, 23, 3, 29, 23))),
	('mms4_edp_brst_l2_dce_20210223072503_v3.0.0.cdf', (4, 'edp', 'dce', datetime(2021, 2, 23, 7, 25, 3))),
	('mms2_fgm_brst_l2_20210223032923_v5.283.0.cdf', (2, 'fgm', '', datetime(2021, 2, 23, 3, 29, 23))),
	('mms3_scm_brst_l2_schb_20190816093103_v2.2.0.cdf', (3, 'scm', 'schb', datetime(2019, 8, 16, 9, 31, 3))),
])
def test_burst_cdf_names_parse(name, parsed):
	assert ffttest._burst_cdf_name(os.path.join('E:/PlasmaCoop/pydata/x', name)) == parsed


@pytest.mark.parametrize('key, bad', [
	('B', [slice(0, 3000)]),                                           # the leading convolution edge
	('B', [slice(9000, 9001), slice(15000, 15400), slice(19000, 20000)]),
	('E', []),
	('E', [slice(0, 10), slice(5000, 5001), slice(12000, 20000)]),
	('B', [slice(0, 20000)]),                                          # nothing good at all
])
def test_cdf_readers_give_what_the_data_pickle_gave(tmp_path, key, bad):
	cdf = write_cdf(str(tmp_path / 'pydata'), key, '20210223100000', T(0), 20000, bad=bad)
	pkl = ffttest.ensure_data_pickle(cdf, str(tmp_path / 'ref'), key + 'z', probe=1)
	old = ffttest.read_data_pickle(pkl)
	new = ffttest.read_burst_data(cdf, key + 'z')
	assert new['time'].dtype == 'datetime64[ns]'
	pd.testing.assert_frame_equal(new, old, check_dtype=not old.empty)
	assert ffttest._segments_from_cdf(cdf, key, SAMPLE_RATE, 1) == ffttest._segments_from_pickle(pkl, SAMPLE_RATE)
	assert ffttest._good_time_bounds(cdf, key, 1) == ffttest._pickle_time_bounds(pkl)
	assert ffttest._good_time_bounds(cdf, key, 1, pickle_root=str(tmp_path / 'cache')) == \
		ffttest._pickle_time_bounds(pkl)


def test_fgm_readers_give_what_the_data_pickle_gave(tmp_path):
	cdf = write_cdf(str(tmp_path / 'pydata'), 'F', '20210223100000', T(0), 3000, step_ns=FGM_STEP_NS)
	pkl = ffttest.ensure_data_pickle(cdf, str(tmp_path / 'ref'), 'F', probe=1)
	pd.testing.assert_frame_equal(ffttest.read_burst_data(cdf, 'F'), ffttest.read_data_pickle(pkl))
	assert ffttest._good_time_bounds(cdf, 'F', 1) == ffttest._pickle_time_bounds(pkl)


def test_read_burst_data_still_reads_a_data_pickle(tmp_path):
	cdf = write_cdf(str(tmp_path / 'pydata'), 'E', '20210223100000', T(0), 4000)
	pkl = ffttest.ensure_data_pickle(cdf, str(tmp_path / 'ref'), 'Ez', probe=1)
	pd.testing.assert_frame_equal(ffttest.read_burst_data(pkl, 'Ez'), ffttest.read_data_pickle(pkl))


@pytest.mark.parametrize('key', ['B', 'E'])
@pytest.mark.parametrize('until_s', [-5.0, 0.0, 0.3, 1.0, 2.43, 2.6, 30.0])
def test_seam_read_holds_every_sample_before_until(tmp_path, key, until_s):
	# 20000 samples span 2.44 s; the first 0.49 s are quality-bad, as an SCM
	# file's convolution edge is, so a short head can hold no good sample at all.
	cdf = write_cdf(str(tmp_path / 'pydata'), key, '20210223100000', T(0), 20000,
	                bad=[slice(0, 4000), slice(7000, 7001)])
	until_ns = pd.Timestamp(T(until_s)).value
	full = ffttest.read_burst_data(cdf, key + 'z')
	head = ffttest.read_burst_data(cdf, key + 'z', until_ns=until_ns)
	pd.testing.assert_frame_equal(head, full.iloc[:len(head)])
	assert (full['time'].values.astype(np.int64) < until_ns).sum() <= len(head)
	if until_s in (0.3, 1.0):
		assert len(head) < len(full)           # really a partial read


def test_seam_read_falls_back_to_the_whole_file_when_the_head_stops_short(tmp_path):
	data = str(tmp_path / 'pydata')
	until_ns = pd.Timestamp(T(1.5)).value
	normal = write_cdf(data, 'B', '20210223100000', T(0), 40000)
	assert ffttest._records_before(normal, 'Bz', 1, until_ns, SAMPLE_RATE) == int(np.ceil(1.5 * SAMPLE_RATE)) + SAMPLE_RATE // 4
	# Samples twice as dense as the rate the head is sized with: that many records
	# cover only half the time, so the head cannot reach until_ns.
	dense = write_cdf(data, 'B', '20210223100100', T(0), 40000, step_ns=STEP_NS // 2)
	assert ffttest._records_before(dense, 'Bz', 1, until_ns, SAMPLE_RATE) is None
	pd.testing.assert_frame_equal(ffttest.read_burst_data(dense, 'Bz', until_ns=until_ns),
	                              ffttest.read_burst_data(dense, 'Bz'))


@pytest.mark.parametrize('key, next_bad', [('B', []), ('B', [slice(0, 4000)]), ('E', [slice(100, 101)])])
def test_ffts_from_cdfs_equal_ffts_from_data_pickles(tmp_path, monkeypatch, key, next_bad):
	data, ref = str(tmp_path / 'pydata'), str(tmp_path / 'ref')
	# One continuous recording split into two files at 4 s, so the first file's
	# last windows borrow from the second.
	curr = write_cdf(data, key, '20210223100000', T(0), 32800)
	nxt = write_cdf(data, key, '20210223100004', T(0), 32736, first=32800, bad=next_bad)
	curr_pkl = ffttest.ensure_data_pickle(curr, ref, key + 'z', probe=1)
	next_pkl = ffttest.ensure_data_pickle(nxt, ref, key + 'z', probe=1)
	inst, mode = INSTRUMENTS[key]
	args = (key + 'z', inst, mode, SAMPLE_RATE, 0.25, int(SAMPLE_RATE * 0.5), 2.0, T(0), T(4))

	reads = []
	real = ffttest._cdf_to_df

	def spy(path, k, probe=1, n_records=None):
		reads.append((path, n_records))
		return real(path, k, probe=probe, n_records=n_records)

	monkeypatch.setattr(ffttest, '_cdf_to_df', spy)
	from_cdfs = ffttest._compute_file_ffts(curr, nxt, *args)
	from_pickles = ffttest._compute_file_ffts(curr_pkl, next_pkl, *args)
	assert len(from_cdfs) > 0
	pd.testing.assert_frame_equal(from_cdfs, from_pickles)
	assert [p for p, _n in reads] == [curr, nxt]
	assert reads[1][1] is not None and reads[1][1] < 32736      # the next file was read only to its seam
	if not next_bad:
		# The seam windows really borrowed: rows exist within half a window of the next file.
		assert (from_cdfs['time'] > np.datetime64(T(3.8), 'ns')).any()


def _anchor_tree(data):
	slow = 10_000_000                       # 100 Hz keeps minute-long files small
	d23 = lambda h, m, s=0: datetime(2021, 2, 23, h, m, s)
	return [
		write_cdf(data, 'B', '20210220120000', datetime(2021, 2, 20, 12), 600, step_ns=slow),   # days earlier
		# raw samples run to 10:30:05, good ones stop at 10:29:59
		write_cdf(data, 'B', '20210223102835', d23(10, 28, 35), 9000, step_ns=slow, bad=[slice(8400, 9000)]),
		write_cdf(data, 'B', '20210223103635', d23(10, 36, 35), 600, step_ns=slow),
		write_cdf(data, 'B', '20210223111000', d23(11, 10), 600, step_ns=slow),
		write_cdf(data, 'B', '20210223111500', d23(11, 15), 600, step_ns=slow),
	]


@pytest.mark.parametrize('start, expected', [
	# The naive anchor's raw data reaches 10:30 but its good data does not: fall forward.
	(datetime(2021, 2, 23, 10, 30), ['103635', '111000']),
	# Its good data reaches 10:29:30: it anchors.
	(datetime(2021, 2, 23, 10, 29, 30), ['102835', '103635', '111000']),
	# Before every file: the first file after start.
	(datetime(2021, 2, 19), ['120000', '102835', '103635', '111000']),
])
@pytest.mark.parametrize('sidecars', [False, True])
def test_find_local_cdfs_picks_what_find_data_pickles_picked(tmp_path, start, expected, sidecars):
	data, ref = str(tmp_path / 'pydata'), str(tmp_path / 'ref')
	files = _anchor_tree(data)
	end = datetime(2021, 2, 23, 11, 10)
	for f in files:
		ffttest.ensure_data_pickle(f, ref, 'Bz', probe=1)
	old = ffttest.find_data_pickles(ref, 'scm', 'scb', start, end, probe=1)
	new = ffttest.find_local_cdfs(data, 'scm', 'scb', start, end, probe=1,
	                              pickle_root=str(tmp_path / 'cache') if sidecars else None)
	assert [t for t, _p in new] == [t for t, _p in old]
	assert [t.strftime('%H%M%S') for t, _p in new] == expected
	assert all(p.endswith('.cdf') for _t, p in new)
	# Raw bounds would have kept that anchor at 10:30 -- the reason for good bounds.
	assert ffttest.cdf_time_bounds(files[1], key='B', probe=1)[1] >= datetime(2021, 2, 23, 10, 30)


def test_check_cdfs_deletes_and_logs_unreadable_files(tmp_path, capsys):
	good = write_cdf(str(tmp_path / 'pydata'), 'B', '20210223100000', T(0), 5000)
	folder = os.path.dirname(good)
	html = os.path.join(folder, 'mms1_scm_brst_l2_scb_20210223100100_v1.0.0.cdf')
	with open(html, 'w') as fh:
		fh.write('<html><body>429 Too Many Requests</body></html>')
	truncated = os.path.join(folder, 'mms1_scm_brst_l2_scb_20210223100200_v1.0.0.cdf')
	with open(good, 'rb') as src, open(truncated, 'wb') as dst:
		dst.write(src.read()[:os.path.getsize(good) // 2])
	log = str(tmp_path / 'corrupted.txt')

	jobs = [(good, 'Bz', 1), (html, 'Bz', 1), (truncated, 'Bz', 1)]
	assert ffttest.check_cdfs(jobs, corrupted_log=log,
	                          pickle_root=str(tmp_path / 'cache')) == [good, None, None]
	assert os.path.exists(good) and not os.path.exists(html) and not os.path.exists(truncated)
	with open(log) as fh:
		assert fh.read().splitlines() == [html, truncated]
	assert capsys.readouterr().out.count('Bad CDF, deleting:') == 2


def test_check_cdfs_checks_each_file_version_once_per_process(tmp_path, monkeypatch):
	cdf = write_cdf(str(tmp_path / 'pydata'), 'B', '20210223100000', T(0), 5000)
	opened = []
	real = ffttest._check_cdf
	monkeypatch.setattr(ffttest, '_check_cdf', lambda *a: opened.append(a[0]) or real(*a))
	monkeypatch.setattr(ffttest, '_CHECKED_CDFS', set())
	jobs = [(cdf, 'Bz', 1)]
	cache = str(tmp_path / 'cache')

	assert ffttest.check_cdfs(jobs, pickle_root=cache) == [cdf]
	assert ffttest.check_cdfs(jobs, pickle_root=cache) == [cdf]
	assert opened == [cdf]                             # the second call did not open it

	later = os.path.getmtime(cdf) + 10
	os.utime(cdf, (later, later))                      # replaced on disk since: checked again
	assert ffttest.check_cdfs(jobs, pickle_root=cache) == [cdf]
	assert opened == [cdf, cdf]


def test_segments_sidecar_is_reused_and_goes_stale_with_its_cdf(tmp_path, monkeypatch):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	# The sidecar splits at any missing good sample (GOOD_RUN_GAP_FACTOR), so any
	# interior hole splits the run; this one is wider than the event threshold as
	# well, so the fixture stays two runs AND two events whatever either is set to.
	hole = int(ffttest.segment_gap_factor(SAMPLE_RATE)) + 10
	cdf = write_cdf(data, 'B', '20210223100000', T(0), 2 * hole + 6000,
	                bad=[slice(0, 3000), slice(9000, 9000 + hole)])
	expected = ffttest._segments_from_cdf(cdf, 'B', SAMPLE_RATE, 1)
	assert len(expected) == 2

	reads = []
	real = ffttest._good_sample_times
	monkeypatch.setattr(ffttest, '_good_sample_times', lambda *a: reads.append(a) or real(*a))

	assert ffttest._segments_via_sidecar(cdf, 'B', SAMPLE_RATE, 1, cache) == expected
	sidecar = ffttest._segments_sidecar_path(cache, cdf, SAMPLE_RATE)
	# The threshold is in the directory name on purpose, so a changed threshold
	# lands in a new folder instead of reusing runs cut at the old one. Derived
	# here rather than hardcoded, so choosing a different one is not a test edit.
	gap_dir = f'gap{int(ffttest.GOOD_RUN_GAP_FACTOR * 1e9 / SAMPLE_RATE)}ns'
	assert sidecar.replace('\\', '/').endswith(
		f'mms/1/scm/scb/segments/{gap_dir}/2021/02/23/100000.npy')
	assert os.path.getsize(sidecar) < 1000 and ffttest._read_version(sidecar) == ffttest.SEGMENTS_VERSION
	assert len(reads) == 1

	assert ffttest._segments_via_sidecar(cdf, 'B', SAMPLE_RATE, 1, cache) == expected       # from the sidecar
	assert len(reads) == 1

	older = os.path.getmtime(cdf) - 10
	os.utime(sidecar, (older, older))                  # the CDF is newer: it was replaced since
	assert ffttest._segments_via_sidecar(cdf, 'B', SAMPLE_RATE, 1, cache) == expected
	assert len(reads) == 2

	ffttest._write_version(sidecar, ffttest.SEGMENTS_VERSION + 1)
	assert ffttest._segments_via_sidecar(cdf, 'B', SAMPLE_RATE, 1, cache) == expected
	assert len(reads) == 3

	# The directory tracks the THRESHOLD in nanoseconds, so a changed one cannot
	# reuse runs cut at the old value. "One missing sample" is a shorter span at a
	# higher rate, so the same factor at another rate is another folder; scb and
	# schb never share an entry anyway -- mode is in the path.
	assert ffttest._segments_sidecar_path(cache, cdf, 2 * SAMPLE_RATE) != sidecar
	assert ffttest._segments_sidecar_path(cache, cdf, SAMPLE_RATE, gap_factor=2 * SAMPLE_RATE) != sidecar

	with pytest.raises(ValueError):
		ffttest._segments_via_sidecar(cdf, 'F', SAMPLE_RATE, 1, cache)


def test_a_run_over_another_interval_reuses_the_segment_sidecars(tmp_path, monkeypatch):
	"""One segment sidecar per CDF, in the cache: whichever interval a run covers, it
	reuses the sidecar an earlier run -- over any other interval, however many runs
	ago -- wrote for a CDF they share (Graham, 2026-09-15)."""
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	a = write_cdf(data, 'B', '20210223100000', T(0), 32800)
	b = write_cdf(data, 'B', '20210223100004', T(0), 32768, first=32800)
	c = write_cdf(data, 'B', '20210223100008', T(0), 32768, first=65568)
	reads = []
	real = ffttest._good_sample_times
	monkeypatch.setattr(ffttest, '_good_sample_times', lambda path, *rest: reads.append(path) or real(path, *rest))

	def segments_for_run(start, end):
		fft_jobs, _owner, _specs = ffttest._plan_fft_jobs(data, cache, [('Bz', 1, 0.25, 3000, [(start, end, False)])])
		cdfs = sorted(job[0][0] for job in fft_jobs.values())
		return {cdf: ffttest._segments_via_sidecar(cdf, 'B', SAMPLE_RATE, 1, cache) for cdf in cdfs}

	first = segments_for_run(T(0), T(6))          # touches a and b
	assert sorted(first) == [a, b] and sorted(reads) == [a, b]
	second = segments_for_run(T(5), T(11))        # touches b and c
	assert sorted(second) == [b, c]
	assert reads[2:] == [c]                        # b came from the first run's sidecar
	assert second[b] == first[b]
	sidecars = sorted(os.path.relpath(os.path.join(r, f), cache).replace(os.sep, '/')
	                  for r, _d, files in os.walk(cache) for f in files if f.endswith('.npy'))
	gap_dir = f'gap{int(ffttest.GOOD_RUN_GAP_FACTOR * 1e9 / SAMPLE_RATE)}ns'
	assert sidecars == [f'mms/1/scm/scb/segments/{gap_dir}/2021/02/23/{s}.npy'
	                    for s in ('100000', '100004', '100008')]


def test_a_short_quality_hole_is_two_good_runs_but_one_event(tmp_path):
	"""Graham, 2026-09-29: the sidecar covers all good data and the manifest groups
	it into physical events. So a one-sample hole splits the good runs, and the
	event threshold joins them back into one event: real MMS data has nothing
	between its widest EDP dropout (647.8 ms) and 4.000 s, and 1.5 s sits in that
	empty band. A hole wider than the threshold stays two events."""
	try:
		from spectro.data_intervals import cdf_data_intervals, merge_bursts
	except ImportError:
		from src.spectro.data_intervals import cdf_data_intervals, merge_bursts
	data = str(tmp_path / 'pydata')

	def runs_and_events(cdf):
		runs = ffttest._segments_from_cdf(cdf, 'B', SAMPLE_RATE, 1)
		events = merge_bursts(cdf_data_intervals([cdf], segments_resolver=lambda _p: runs),
		                      tol=ffttest.segment_gap_tol(SAMPLE_RATE))
		return len(runs), len(events)

	short = write_cdf(data, 'B', '20210223110000', T(0), 20000,
	                  bad=[slice(9000, 9001)])                      # one sample: 244 us
	assert runs_and_events(short) == (2, 1)

	wide = int(ffttest.segment_gap_factor(SAMPLE_RATE)) + 10             # just over 1.5 s
	split = write_cdf(data, 'B', '20210223120000', T(0), 2 * wide + 6000,
	                  bad=[slice(9000, 9000 + wide)])
	assert runs_and_events(split) == (2, 2)


def test_the_manifest_groups_the_sidecar_runs_into_events(tmp_path):
	"""collect_fft_data's manifest reads each CDF's sidecar (all its good data) and
	groups the runs at the event threshold: a 1 s quality hole leaves ONE row in the
	used table, whose `good` is the hole's length short of its `event` (Graham,
	2026-09-29). The recorded table's one row spans the same event with nothing
	missing, because the hole's samples were recorded, only flagged (Graham,
	2026-09-30: recorded and good are two separate things)."""
	try:
		from spectro.data_intervals import read_burst_manifest
	except ImportError:
		from src.spectro.data_intervals import read_burst_manifest
	data, cache, mdir = str(tmp_path / 'pydata'), str(tmp_path / 'cache'), str(tmp_path / 'manifests')
	cdf = write_cdf(data, 'B', '20210223100000', T(0), 5 * SAMPLE_RATE,
	                bad=[slice(2 * SAMPLE_RATE, 3 * SAMPLE_RATE)])
	ffttest.collect_fft_data(data, cache, 'Bz', 0.25, T(0), T(5), probe=1, download=False,
	                         manifest_dir=mdir)
	used = read_burst_manifest(_only_manifest(mdir))
	recorded = read_burst_manifest(_only_manifest(mdir), table='recorded')
	assert len(used) == 1 and len(recorded) == 1
	event = (used[0].end - used[0].start).total_seconds()
	assert abs((event - used[0].recorded.total_seconds()) - (SAMPLE_RATE + 1) / SAMPLE_RATE) < 0.001
	assert (recorded[0].start, recorded[0].end) == (used[0].start, used[0].end)
	assert abs((recorded[0].end - recorded[0].start - recorded[0].recorded).total_seconds()) < 0.001   # file keeps ms
	assert len(np.load(ffttest._segments_sidecar_path(cache, cdf, SAMPLE_RATE))) == 2
	assert len(np.load(ffttest._segments_sidecar_path(cache, cdf, SAMPLE_RATE, kind='recorded'))) == 1


def test_the_threshold_is_the_same_inside_a_file_and_across_a_seam(tmp_path):
	"""Graham, 2026-09-25: "the segment threshold should absolutely apply over
	file boundaries." It was 1.5 periods inside a file and 2.0 across a seam, so
	the same hole was a break or not depending on where a file happened to end."""
	assert ffttest.segment_gap_tol(SAMPLE_RATE).total_seconds() == \
		ffttest.segment_gap_factor(SAMPLE_RATE) / SAMPLE_RATE
	# and it does not change meaning when the rate does
	assert ffttest.segment_gap_tol(2 * SAMPLE_RATE) == ffttest.segment_gap_tol(SAMPLE_RATE)


def test_the_cadence_jitter_margin_is_not_the_segment_threshold(tmp_path):
	"""Wiring them together would quietly loosen the window-fill test: at 1 s
	every EDP dropout up to 125 ms would count as normal spacing, dragging the
	measured sample period up and letting windows missing ~177 real samples pass
	as full."""
	assert ffttest.CADENCE_JITTER_FACTOR == 1.5
	assert ffttest.CADENCE_JITTER_FACTOR < ffttest.segment_gap_factor(SAMPLE_RATE)


def test_an_unreadable_cdf_leaves_no_sidecar(tmp_path, monkeypatch):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	cdf = write_cdf(data, 'E', '20210223100000', T(0), 1000)

	def held_open(*_a):
		raise OSError('the file is held open by another process')

	monkeypatch.setattr(ffttest, '_good_sample_times', held_open)
	assert ffttest._segments_via_sidecar(cdf, 'E', SAMPLE_RATE, 1, cache) == []
	assert not os.path.exists(ffttest._segments_sidecar_path(cache, cdf, SAMPLE_RATE))


def test_prewarm_computes_from_cdfs_and_writes_no_data_pickle(tmp_path):
	data, cache, ref = str(tmp_path / 'pydata'), str(tmp_path / 'cache'), str(tmp_path / 'ref')
	curr = write_cdf(data, 'B', '20210223100000', T(0), 32800)
	nxt = write_cdf(data, 'B', '20210223100004', T(0), 32736, first=32800)
	spec = [('Bz', 1, 0.25, 3000, [(T(0), T(8), False)])]

	fft_jobs, _owner, _specs = ffttest._plan_fft_jobs(data, cache, spec)
	assert sorted(job[0][:2] for job in fft_jobs.values()) == [(curr, nxt), (nxt, None)]

	ffttest.prewarm(data, cache, spec, write_manifest=False, max_workers=2)
	written = pickles_under(cache)
	assert len(written) == 2 and all(os.sep + 'fft' + os.sep in p for p in written)

	curr_pkl = ffttest.ensure_data_pickle(curr, ref, 'Bz', probe=1)
	next_pkl = ffttest.ensure_data_pickle(nxt, ref, 'Bz', probe=1)
	# prev is part of the inputs now: a window overhanging the START of a file
	# borrows from the file before it, exactly as a trailing one borrows from the
	# file after (see _compute_file_ffts). curr has nothing before it on disk;
	# nxt has curr. Leaving prev out here made the reference the forward-only
	# answer and cost nxt its first pad/step windows.
	for c, n, p, ft, nft in ((curr_pkl, next_pkl, None, T(0), T(4)),
	                         (next_pkl, None, curr_pkl, T(4), None)):
		want = ffttest._compute_file_ffts(c, n, 'Bz', 'scm', 'scb', SAMPLE_RATE, 0.25, 4096, 2.0, ft, nft,
		                                  prev_path=p, prev_file_time=T(0) if p else None)
		got = pd.read_pickle(ffttest._fft_pickle_path(cache, 'scm', 'scb', 0.25, ft, probe=1))
		pd.testing.assert_frame_equal(got, want)


def test_collect_fce_reads_fgm_cdfs_directly(tmp_path):
	import fce
	data, cache, ref = str(tmp_path / 'pydata'), str(tmp_path / 'cache'), str(tmp_path / 'ref')
	write_cdf(data, 'F', '20210223100000', T(0), 6000, step_ns=FGM_STEP_NS)
	write_cdf(data, 'F', '20210223100200', T(120), 6000, step_ns=FGM_STEP_NS)
	st, ts = T(30), T(200)

	times, f_ce = fce.collect_fce(data, cache, st, ts, probe=1, download=False)
	assert times.size > 0 and not pickles_under(cache)

	# What the data-pickle path yielded, step for step.
	for cdf in ffttest.find_cdf_files(data, 'fgm', '', st, ts, probe=1, download=False):
		ffttest.ensure_data_pickle(cdf, ref, 'F', probe=1)
	want_t, want_b = [], []
	for _ft, pkl in ffttest.find_data_pickles(ref, 'fgm', 'brst', st, ts, probe=1):
		df = pd.read_pickle(pkl)
		t = df['time'].values.astype('datetime64[ns]')
		m = (t >= np.datetime64(st, 'ns')) & (t <= np.datetime64(ts, 'ns'))
		want_t.append(t[m])
		want_b.append(df['Bmag'].values[m])
	assert np.array_equal(times, np.concatenate(want_t))
	assert np.array_equal(f_ce, fce.fce_from_bmag(np.concatenate(want_b)))


# ---------------------------------------------------------------------------
# the validation database, wired in
# ---------------------------------------------------------------------------

def test_check_cdfs_skips_the_open_for_a_file_that_passed_in_an_earlier_run(tmp_path, monkeypatch):
	"""The point of the database: the ~60 ms is the library validating the file at
	OPEN time, so the answer has to be looked up before opening. A fresh process
	(empty _CHECKED_CDFS) must not re-open a file an earlier run already passed."""
	import cdf_validation
	cache = str(tmp_path / 'cache')
	cdf = write_cdf(str(tmp_path / 'pydata'), 'B', '20210223100000', T(0), 5000)
	opened = []
	real = ffttest._check_cdf
	monkeypatch.setattr(ffttest, '_check_cdf', lambda *a: opened.append(a[0]) or real(*a))
	monkeypatch.setattr(ffttest, '_CHECKED_CDFS', set())
	jobs = [(cdf, 'Bz', 1)]

	assert ffttest.check_cdfs(jobs, pickle_root=cache) == [cdf]
	assert opened == [cdf]
	assert cdf_validation.passed(cdf, 'Bz', pickle_root=cache) is True

	monkeypatch.setattr(ffttest, '_CHECKED_CDFS', set())      # as a new process starts
	assert ffttest.check_cdfs(jobs, pickle_root=cache) == [cdf]
	assert opened == [cdf]                                    # still one open, ever

	later = os.path.getmtime(cdf) + 10
	os.utime(cdf, (later, later))                             # replaced since: checked again
	monkeypatch.setattr(ffttest, '_CHECKED_CDFS', set())
	assert ffttest.check_cdfs(jobs, pickle_root=cache) == [cdf]
	assert opened == [cdf, cdf]


def test_check_cdfs_counts_failures_and_gives_up_at_the_limit(tmp_path, monkeypatch):
	"""Graham, 2026-09-25: retry a corrupted file, and only after 5 attempts give
	up on it. The count lives in the database keyed on the FILENAME, which is why
	it survives check_cdfs deleting the file in order to refetch it."""
	import cdf_validation
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	log = str(tmp_path / 'corrupted.txt')
	folder = os.path.join(data, 'mms1/scm/brst/l2/scb/2021/02/23')
	os.makedirs(folder, exist_ok=True)
	bad = os.path.join(folder, 'mms1_scm_brst_l2_scb_20210223100000_v1.0.0.cdf')

	for attempt in range(1, cdf_validation.GIVE_UP_AFTER + 1):
		with open(bad, 'wb') as fh:                          # a fresh bad refetch
			fh.write(b'<html>503</html>' + bytes([attempt]))
		monkeypatch.setattr(ffttest, '_CHECKED_CDFS', set())
		assert ffttest.check_cdfs([(bad, 'Bz', 1)], corrupted_log=log, pickle_root=cache) == [None]
		assert not os.path.exists(bad)                       # deleted so it can be refetched
		assert cdf_validation.attempts(bad, 'Bz', pickle_root=cache) == attempt
		assert cdf_validation.give_up(bad, 'Bz', pickle_root=cache) is (
			attempt >= cdf_validation.GIVE_UP_AFTER)

	assert cdf_validation.corrupted_names('Bz', pickle_root=cache) == [os.path.basename(bad)]

	# and a copy that finally arrives intact clears the record
	good = write_cdf(data, 'B', '20210223100000', T(0), 5000)
	assert good == bad
	monkeypatch.setattr(ffttest, '_CHECKED_CDFS', set())
	assert ffttest.check_cdfs([(bad, 'Bz', 1)], corrupted_log=log, pickle_root=cache) == [bad]
	assert cdf_validation.attempts(bad, 'Bz', pickle_root=cache) == 0
	assert cdf_validation.corrupted_names('Bz', pickle_root=cache) == []


def test_check_cdfs_does_not_reopen_a_known_bad_file(tmp_path, monkeypatch):
	"""Once these exact bytes are recorded bad, re-opening them buys the same
	answer at 60 ms. Still on disk means the last delete failed (a locked file);
	no corrupt file may outlive its detection (Graham, 2026-09-27), so it is
	deleted now -- and not counted again, since no download failed."""
	import cdf_validation
	cache = str(tmp_path / 'cache')
	folder = os.path.join(str(tmp_path / 'pydata'), 'mms1/scm/brst/l2/scb/2021/02/23')
	os.makedirs(folder, exist_ok=True)
	bad = os.path.join(folder, 'mms1_scm_brst_l2_scb_20210223100000_v1.0.0.cdf')
	with open(bad, 'wb') as fh:
		fh.write(b'<html>503</html>')
	cdf_validation.record_failure(bad, 'Bz', pickle_root=cache)

	opened = []
	monkeypatch.setattr(ffttest, '_check_cdf', lambda *a: opened.append(a[0]))
	monkeypatch.setattr(ffttest, '_CHECKED_CDFS', set())
	assert ffttest.check_cdfs([(bad, 'Bz', 1)], pickle_root=cache) == [None]
	assert opened == []
	assert not os.path.exists(bad)
	assert cdf_validation.attempts(bad, 'Bz', pickle_root=cache) == 1


def test_manifest_carries_a_corrupted_section(tmp_path):
	"""A manifest row says data IS here. Without this section a file that never
	arrived is indistinguishable from data that was never taken -- and the record
	dies with the deleted file. Written as comments so existing readers skip it."""
	try:
		from spectro.data_intervals import (write_burst_manifest, read_burst_manifest,
		                                    read_corrupted_names, Burst)
	except ImportError:
		from src.spectro.data_intervals import (write_burst_manifest, read_burst_manifest,
		                                        read_corrupted_names, Burst)
	out = str(tmp_path / 'manifest.txt')
	bursts = [Burst(start=T(0), end=T(10), n_files=2)]
	names = ['mms1_scm_brst_l2_scb_20210223100000_v1.0.0.cdf',
	         'mms1_scm_brst_l2_scb_20210223100004_v1.0.0.cdf']

	write_burst_manifest(out, bursts, corrupted=names)
	assert read_corrupted_names(out) == names
	assert len(read_burst_manifest(out)) == 1        # the coverage reader is untouched

	write_burst_manifest(out, bursts)               # nothing given up on
	assert read_corrupted_names(out) == []
	assert len(read_burst_manifest(out)) == 1


# ---------------------------------------------------------------------------
# a corrupt CDF meets check_cdfs before anything else opens it
# ---------------------------------------------------------------------------

N_4S = 32_800                    # 4.004 s of samples: A and C tile T(0)..T(12) around B


def _tree_around_a_bad_file(root, bad):
	"""A and C good. Between them B is an HTML error page saved under a real CDF
	name when `bad`, and absent otherwise -- the tree B's deletion should leave."""
	data = str(root / 'pydata')
	a = write_cdf(data, 'B', '20210223100000', T(0), N_4S)
	write_cdf(data, 'B', '20210223100008', T(0), N_4S, first=2 * N_4S)
	b = os.path.join(os.path.dirname(a), 'mms1_scm_brst_l2_scb_20210223100004_v1.0.0.cdf')
	if bad:
		with open(b, 'wb') as fh:
			fh.write(b'<html>503 Service Unavailable</html>')
	return data, b


def _only_manifest(manifest_dir):
	[path] = [os.path.join(r, f) for r, _d, files in os.walk(manifest_dir) for f in files]
	return path


@pytest.mark.parametrize('entry', ['collect_fft_data', 'gen_fft'])
def test_a_corrupt_cdf_is_deleted_logged_and_kept_out_of_the_manifest(tmp_path, entry):
	"""The manifest step opens every file it is handed, and it used to run BEFORE
	check_cdfs. collect_fft_data -- the live short-interval path -- died on the bad
	file, leaving it on disk, unlogged and uncounted. gen_fft's resolver fails open
	to filename bounds, so its manifest reported the bad file's whole span as
	recorded data that no FFT then used. Checked first, the file is deleted, logged
	and counted, and the manifest is exactly the one a tree without it gets."""
	import cdf_validation
	manifests = {}
	for tree, bad in (('with_bad', True), ('without', False)):
		data, b = _tree_around_a_bad_file(tmp_path / tree, bad)
		cache, log = str(tmp_path / tree / 'cache'), str(tmp_path / tree / 'corrupted.txt')
		mdir = str(tmp_path / tree / 'manifests')
		kw = dict(corrupted_log=log, probe=1, download=False, manifest_dir=mdir, write_manifest=True)
		if entry == 'collect_fft_data':
			fdf = ffttest.collect_fft_data(data, cache, 'Bz', 0.25, T(1), T(11), **kw)
		else:
			fdf = pd.concat(list(ffttest.gen_fft(data, cache, 'Bz', 0.25, T(1), T(11), **kw)))
		assert not fdf.empty
		with open(_only_manifest(mdir)) as fh:
			manifests[tree] = fh.read()
		if bad:
			assert not os.path.exists(b)
			with open(log) as fh:
				assert fh.read().splitlines() == [b]
			assert cdf_validation.attempts(b, 'Bz', pickle_root=cache) == 1
	assert manifests['with_bad'] == manifests['without']


def test_a_request_whose_only_file_is_corrupt_comes_back_empty(tmp_path):
	"""Nothing survives validation, so the request is one that found no files: an
	empty frame and no manifest, as for any request with no data -- not the
	exception the manifest step used to raise with the bad file still on disk."""
	folder = tmp_path / 'pydata' / 'mms1' / 'scm' / 'brst' / 'l2' / 'scb' / '2021' / '02' / '23'
	folder.mkdir(parents=True)
	bad = str(folder / 'mms1_scm_brst_l2_scb_20210223100000_v1.0.0.cdf')
	with open(bad, 'wb') as fh:
		fh.write(b'<html>503 Service Unavailable</html>')
	mdir = str(tmp_path / 'manifests')
	fdf = ffttest.collect_fft_data(str(tmp_path / 'pydata'), str(tmp_path / 'cache'), 'Bz', 0.25,
	                               T(1), T(3), probe=1, download=False, manifest_dir=mdir)
	assert fdf.empty and not os.path.exists(bad)
	assert not os.path.exists(mdir)


@pytest.mark.parametrize('which', ['prev', 'next'])
def test_a_corrupt_seam_neighbour_is_validated_before_the_fft_reads_it(tmp_path, which):
	"""pair_with_next_cdf takes the first file's predecessor and the last file's
	successor off disk, outside anything the request validated, and the FFT
	worker reads them for seam samples. A bad one used to crash the request; now
	it is deleted before it is handed on, and the FFT is the one computed as if it
	had never been there."""
	import cdf_validation
	out = {}
	for tree, bad in (('with_bad', True), ('without', False)):
		root = tmp_path / tree
		data = str(root / 'pydata')
		a = write_cdf(data, 'B', '20210223100000', T(0), N_4S)
		write_cdf(data, 'B', '20210223100004', T(0), N_4S, first=N_4S)
		stamp = '20210223095955' if which == 'prev' else '20210223100008'
		nb = os.path.join(os.path.dirname(a), f'mms1_scm_brst_l2_scb_{stamp}_v1.0.0.cdf')
		if bad:
			with open(nb, 'wb') as fh:
				fh.write(b'<html>503 Service Unavailable</html>')
		cache = str(root / 'cache')
		out[tree] = ffttest.collect_fft_data(data, cache, 'Bz', 0.25, T(0.1), T(7.9), probe=1,
		                                     download=False, write_manifest=False)
		if bad:
			assert not os.path.exists(nb)
			assert cdf_validation.attempts(nb, 'Bz', pickle_root=cache) == 1
	pd.testing.assert_frame_equal(out['with_bad'], out['without'])


def test_find_cdf_files_validates_the_anchor_before_reading_its_bounds(tmp_path, monkeypatch):
	"""With download=True discovery reads the anchor's own recorded bounds to
	check it reaches `start`. That read used to open an unvalidated file and die
	on a bad one; now the anchor is validated first, and a bad one is deleted,
	counted, and fallen forward past."""
	import cdf_validation
	monkeypatch.setattr(ffttest, 'attempt_download', lambda *a, **k: None)
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	a = write_cdf(data, 'B', '20210223100000', T(0), N_4S)
	b = write_cdf(data, 'B', '20210223100004', T(0), N_4S, first=N_4S)
	with open(a, 'wb') as fh:
		fh.write(b'<html>503 Service Unavailable</html>')
	check = ffttest.cdf_check('Bz', 1, pickle_root=cache)
	assert ffttest.find_cdf_files(data, 'scm', 'scb', T(1), T(7), probe=1, download=True, check=check) == [b]
	assert not os.path.exists(a)
	assert cdf_validation.attempts(a, 'Bz', pickle_root=cache) == 1


def test_fgm_runs_validates_before_it_opens(tmp_path, monkeypatch):
	"""fgm_runs opened every FGM file in the range with no check at all, and a bad
	one was logged and left on disk. Now it is validated first: deleted, counted,
	never opened as data."""
	import cdf_validation
	import event_export as ee
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	monkeypatch.setattr(ee, 'DATA_DIR', data)
	monkeypatch.setattr(ee, 'PICKLE_ROOT', cache)
	good = write_cdf(data, 'F', '20210223100000', T(0), 513, step_ns=FGM_STEP_NS)
	bad = os.path.join(os.path.dirname(good), 'mms1_fgm_brst_l2_20210223100004_v1.0.0.cdf')
	with open(bad, 'wb') as fh:
		fh.write(b'<html>503 Service Unavailable</html>')

	class Rng:
		start, end = T(1), T(7)

	runs = ee.fgm_runs(1, Rng, log=lambda *a: None)
	assert len(runs) == 1 and not os.path.exists(bad)
	assert cdf_validation.attempts(bad, 'F', pickle_root=cache) == 1
