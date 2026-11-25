"""Graham, 2026-10-09: "A lone good sample should never count since we're doing math that relies on
continuous data. Single digit numbers of samples do not count as events."

A good run of fewer than ffttest.MIN_GOOD_RUN_SAMPLES samples is no part of an event, wherever good runs
are grouped into events:
  * the event UI -- event_export.find_events
  * the manifests' used table -- write_request_manifest (its recorded table keeps every run)
  * the driver's chunks -- ffttest.stream_good_chunk_ffts
The sidecar still holds every good run, since it covers all good data. A run that a file boundary cuts in
two is counted whole, and two files holding the same samples count them once.
"""
import os
import sys
from datetime import timedelta

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
sys.path.insert(0, HERE)
os.environ.setdefault('MPLBACKEND', 'Agg')

import event_export as ee
import ffttest
from spectro.data_intervals import cdf_data_intervals, drop_short_runs, read_burst_manifest, write_request_manifest
from test_direct_cdf import STEP_NS, T, write_cdf

FS = 8192
RULE = ffttest.good_run_rule(FS)
MAIN = (2 * FS, 8 * FS)                 # the burst's good body, samples [2 s, 8 s) of a 10 s file


def row(start, n, step_ns=STEP_NS, path='a.cdf'):
	"""(start, end, path) of a run of n samples from `start`, one per step."""
	return (start, start + timedelta(microseconds=round((n - 1) * step_ns / 1000)), path)


@pytest.mark.parametrize('step_ns', [122_000, STEP_NS, 123_000])      # times arrive rounded to the microsecond
@pytest.mark.parametrize('n, counts', [(1, False), (2, False), (9, False), (10, True), (11, True)])
def test_a_run_counts_from_ten_samples(n, counts, step_ns):
	rows = [row(T(0), n, step_ns)]
	assert drop_short_runs(rows, **RULE) == (rows if counts else [])


def test_a_run_a_file_boundary_cuts_in_two_counts_whole():
	a = row(T(0), 6, path='a.cdf')
	b = row(a[1] + timedelta(microseconds=122), 6, path='b.cdf')     # the next sample, in the next file
	assert drop_short_runs([a, b], **RULE) == [a, b]                 # one run of 12
	b_late = row(a[1] + timedelta(microseconds=244), 6, path='b.cdf')  # one sample missing between them
	assert drop_short_runs([a, b_late], **RULE) == []                # two runs of 6


def test_two_files_holding_the_same_samples_count_them_once():
	same = [row(T(0), 6, path='a.cdf'), row(T(0), 6, path='b.cdf')]
	assert drop_short_runs(same, **RULE) == []


def test_without_the_rule_cdf_data_intervals_keeps_every_run():
	short, body = row(T(0), 5), row(T(1), 10_000)
	segments = {'a.cdf': [short[:2], body[:2]]}
	default = cdf_data_intervals(['a.cdf'], segments_resolver=segments.get)
	assert [(iv.start, iv.end) for iv in default] == [short[:2], body[:2]]
	ruled = cdf_data_intervals(['a.cdf'], segments_resolver=segments.get, **RULE)
	assert [(iv.start, iv.end) for iv in ruled] == [body[:2]]


def burst_with_short_run(data_dir, key, n_short):
	"""A 10 s file: n_short good samples ending 0.5 s before the good body [2 s, 8 s), bad everywhere else."""
	n = 10 * FS
	short_end = MAIN[0] - FS // 2
	return write_cdf(data_dir, key, '20210223100000', T(0), n,
	                 bad=[slice(0, short_end - n_short), slice(short_end, MAIN[0]), slice(MAIN[1], n)])


@pytest.mark.parametrize('key', ['B', 'E'])
@pytest.mark.parametrize('n_short, counts', [(1, False), (9, False), (10, True)])
def test_find_events_starts_on_the_first_run_that_counts(tmp_path, monkeypatch, key, n_short, counts):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	monkeypatch.setattr(ee, 'DATA_DIR', data)
	monkeypatch.setattr(ee, 'PICKLE_ROOT', cache)          # never the data drive
	cdf = burst_with_short_run(data, key, n_short)
	[event] = ee.find_events(ee.RequestRange(T(0), T(10)), (key,), (1,), log=lambda *a, **k: None)
	if counts:
		assert event.start < T(1.6)                  # on the short run, which ends 0.5 s before the body
	else:
		assert event.start > T(1.9)                  # on the body, which starts at 2.0 s
	# The sidecar still holds the short run: it covers all good data.
	runs = ffttest._segments_via_sidecar(cdf, key, FS, 1, cache)
	assert len(runs) == 2 and round((runs[0][1] - runs[0][0]).total_seconds() * 1e9 / STEP_NS) + 1 == n_short


def test_the_used_table_drops_a_short_run_and_the_recorded_table_keeps_it(tmp_path):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	cdf = burst_with_short_run(data, 'E', 9)
	resolvers = dict(segments_resolver=lambda p: ffttest._segments_via_sidecar(p, 'E', FS, 1, cache),
	                 recorded_resolver=lambda p: ffttest._segments_via_sidecar(p, 'E', FS, 1, cache, recorded=True))
	ruled = write_request_manifest(str(tmp_path / 'ruled'), [cdf], 'Ex', 'edp', 'dce', 1, 0.25, T(0), T(10),
	                               tol=ffttest.segment_gap_tol(FS), **resolvers, **RULE)
	[used] = read_burst_manifest(ruled)
	[rec] = read_burst_manifest(ruled, table='recorded')
	assert used.start > T(1.9)                                            # the body, not the run before it
	assert (rec.end - rec.start).total_seconds() == pytest.approx(10.0, abs=1e-3)
	plain = write_request_manifest(str(tmp_path / 'plain'), [cdf], 'Ex', 'edp', 'dce', 1, 0.25, T(0), T(10),
	                               tol=ffttest.segment_gap_tol(FS), **resolvers)
	[old] = read_burst_manifest(plain)
	assert old.start < used.start                                         # without the rule it started on the run


def test_the_chunk_stream_leaves_a_short_run_out(tmp_path):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	burst_with_short_run(data, 'B', 9)
	used = {}
	for _chunk in ffttest.stream_good_chunk_ffts(data, cache, 'Bz', [1], 0.25, T(0), T(10), download=False,
	                                             max_workers=1, trust_good_pickles=True, segments_sink=used):
		pass
	[(start, _end, _ft)] = used[1]
	assert start > T(1.9)
