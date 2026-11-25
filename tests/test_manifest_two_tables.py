"""Every manifest holds two tables: the used (good) segments and the recorded ones.

Graham, 2026-09-30: "the recorded should show where it was recorded and not
necessarily when good quality data was recorded, those are 2 separate things that
should be visible in the manifest", then "There should be two tables, one for used aka
good segments and one for recorded segments".

What real data does, measured the same day (_wsl_scratch/recorded_vs_good_probe.out):
SCM flags the first and last 2 s of every burst, and EDP's dropouts are samples that
were recorded but flagged. The synthetic files here reproduce both shapes.
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLBACKEND', 'Agg')

pytest.importorskip('spacepy.pycdf')

import ffttest
from ffttest import SAMPLE_RATE
from test_direct_cdf import T, write_cdf, _only_manifest

try:
	from spectro.data_intervals import (Burst, read_burst_manifest, read_corrupted_names,
	                                    write_burst_manifest, write_request_manifest)
except ImportError:
	from src.spectro.data_intervals import (Burst, read_burst_manifest, read_corrupted_names,
	                                        write_burst_manifest, write_request_manifest)

FS = SAMPLE_RATE
EDGE = 2 * FS                       # SCM's flagged lead and trail, in samples


def scm_burst(data, stamp='20210223100000', seconds=10):
	"""A burst file flagged like real SCM: first and last 2 s bad."""
	n = seconds * FS
	return write_cdf(data, 'B', stamp, T(0), n, bad=[slice(0, EDGE), slice(n - EDGE, n)])


def test_the_two_tables_are_written_in_order_and_read_back_by_name(tmp_path):
	t0 = datetime(2021, 2, 23, 10)
	used = [Burst(t0 + timedelta(seconds=2), t0 + timedelta(seconds=8), 1, timedelta(seconds=5.5))]
	recorded = [Burst(t0, t0 + timedelta(seconds=10), 1, timedelta(seconds=10))]
	out = write_burst_manifest(str(tmp_path / 'm.txt'), used, event_gap=timedelta(seconds=1.5),
	                           corrupted=['mms1_scm_brst_l2_scb_20210223100010_v1.0.0.cdf'], recorded=recorded)
	lines = open(out).read().splitlines()
	assert lines[0] == '# merged at event_gap = 1.5 s; durations in seconds'
	assert lines[1].split('\t')[:2] == ['# table', 'used']
	assert lines[2].split('\t') == ['# start', 'end', 'event', 'good', 'n_files']
	assert lines[3].split('\t')[2:] == ['6.000', '5.500', '1']
	assert lines[4].split('\t')[:2] == ['# table', 'recorded']
	assert lines[5].split('\t') == ['# start', 'end', 'event', 'recorded', 'n_files']
	assert lines[6].split('\t')[2:] == ['10.000', '10.000', '1']
	assert read_burst_manifest(out) == used
	assert read_burst_manifest(out, table='recorded') == recorded
	assert read_corrupted_names(out) == ['mms1_scm_brst_l2_scb_20210223100010_v1.0.0.cdf']


def test_a_v4_manifest_is_the_used_table_and_has_no_recorded_one(tmp_path):
	v4 = tmp_path / 'v4.txt'
	v4.write_text('# merged at event_gap = 1.5 s; event and recorded in seconds\n'
	              '# start\tend\tevent\trecorded\tn_files\n'
	              '2021-02-23 10:00:02\t2021-02-23 10:00:08\t6.000\t5.500\t1\n')
	[row] = read_burst_manifest(str(v4))
	assert row.recorded == timedelta(seconds=5.5) and row.n_files == 1
	assert read_burst_manifest(str(v4), table='recorded') == []


def test_recorded_runs_cover_flagged_samples_and_get_their_own_sidecar(tmp_path, monkeypatch):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	cdf = scm_burst(data)
	good = ffttest._segments_via_sidecar(cdf, 'B', FS, 1, cache)
	rec = ffttest._segments_via_sidecar(cdf, 'B', FS, 1, cache, recorded=True)
	assert len(good) == 1 and len(rec) == 1
	assert (good[0][0] - rec[0][0]).total_seconds() == pytest.approx(2.0, abs=1e-3)   # the flagged lead
	assert (rec[0][1] - good[0][1]).total_seconds() == pytest.approx(2.0, abs=1e-3)   # the flagged trail
	seg_path = ffttest._segments_sidecar_path(cache, cdf, FS)
	rec_path = ffttest._segments_sidecar_path(cache, cdf, FS, kind='recorded')
	assert '/segments/' in seg_path.replace(os.sep, '/') and '/recorded/' in rec_path.replace(os.sep, '/')
	assert os.path.exists(seg_path) and os.path.exists(rec_path)
	assert np.load(rec_path).shape == (1, 2)

	def reopened(*_args):
		raise AssertionError('the CDF was read again instead of the sidecar')

	# The second call is answered from the sidecar; the CDF is not read again.
	monkeypatch.setattr(ffttest, '_recorded_sample_times', reopened)
	assert ffttest._segments_via_sidecar(cdf, 'B', FS, 1, cache, recorded=True) == rec


def test_collect_fft_data_writes_both_tables(tmp_path):
	data, cache, mdir = str(tmp_path / 'pydata'), str(tmp_path / 'cache'), str(tmp_path / 'manifests')
	scm_burst(data)
	ffttest.collect_fft_data(data, cache, 'Bz', 0.25, T(0), T(10), probe=1, download=False, manifest_dir=mdir)
	path = _only_manifest(mdir)
	assert path.endswith('_v7.txt')
	[used] = read_burst_manifest(path)
	[rec] = read_burst_manifest(path, table='recorded')
	assert (used.end - used.start).total_seconds() == pytest.approx(6.0, abs=1e-3)    # 10 s less both edges
	assert (rec.end - rec.start).total_seconds() == pytest.approx(10.0, abs=1e-3)
	assert rec.start < used.start and rec.end > used.end


def test_whole_file_bounds_rows_are_both_tables(tmp_path):
	"""No quality filter applied (FGM, or bounds mode): what was used IS what was
	recorded, so the recorded table repeats the used one."""
	data = str(tmp_path / 'pydata')
	cdf = write_cdf(data, 'B', '20210223100000', T(0), 4 * FS)
	out = write_request_manifest(str(tmp_path / 'm'), [cdf], 'Bz', 'scm', 'scb', 1, 0.25, T(0), T(4),
	                             bounds_resolver=lambda p: ffttest.cdf_time_bounds(p, key='B', probe=1))
	assert read_burst_manifest(out, table='recorded') == read_burst_manifest(out)


def test_quality_rows_without_a_recorded_resolver_write_the_used_table_alone(tmp_path):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	cdf = scm_burst(data)
	out = write_request_manifest(str(tmp_path / 'm'), [cdf], 'Bz', 'scm', 'scb', 1, 0.25, T(0), T(10),
	                             segments_resolver=lambda p: ffttest._segments_via_sidecar(p, 'B', FS, 1, cache))
	assert len(read_burst_manifest(out)) == 1
	assert read_burst_manifest(out, table='recorded') == []


def test_the_chunk_stream_fills_the_recorded_sink(tmp_path):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	scm_burst(data)
	used, recorded = {}, {}
	for _chunk in ffttest.stream_good_chunk_ffts(data, cache, 'Bz', [1], 0.25, T(0), T(10), download=False,
	                                             max_workers=1, trust_good_pickles=True,
	                                             segments_sink=used, recorded_sink=recorded):
		pass
	[(gs, ge, _ft)] = used[1]
	[(rs, re_, _ft2)] = recorded[1]
	assert (gs - rs).total_seconds() == pytest.approx(2.0, abs=1e-3)
	assert (re_ - ge).total_seconds() == pytest.approx(2.0, abs=1e-3)
