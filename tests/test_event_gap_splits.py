"""Graham, 2026-10-01: "The 2s gap should just split events to be careful."

SCM zeroes its convolution edges -- 2 s at each end of a processing block, flag 'ZZZ', B
exactly 0 -- and a stretch of zeros is never good data ("No stretch of 0's should ever be
used"). Where two files meet inside a continuous recording, the good data stops 2 s before
the boundary and resumes 2 s after it. Every path that groups good data into events must
split there, and at any gap of 2 s or more:
  * the manifests -- ffttest.segment_gap_tol
  * the event UI and FGM coverage -- event_export.EVENT_GAP_S
  * the driver's chunk clustering -- max(2 * PAD, segment_gap_tol), see
    ffttest.stream_good_chunk_ffts
"""
import os
import sys
from datetime import datetime, timedelta

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
sys.path.insert(0, HERE)
os.environ.setdefault('MPLBACKEND', 'Agg')

import event_export as ee
import ffttest
from spectro.data_intervals import DataInterval, merge_bursts

TWO_S = timedelta(seconds=2)
TOLERANCES = {
	'event UI and FGM coverage': timedelta(seconds=ee.EVENT_GAP_S),
	'manifests': ffttest.segment_gap_tol(ffttest.SAMPLE_RATE),
	'chunk clustering': max(timedelta(seconds=2 * ee.PAD), ffttest.segment_gap_tol(ffttest.SAMPLE_RATE)),
}


def test_every_event_tolerance_is_under_two_seconds():
	for name, tol in TOLERANCES.items():
		assert tol < TWO_S, name


@pytest.mark.parametrize('gap_s', [2.0, 4.0])      # one zeroed edge, or both sides of a boundary
def test_a_gap_of_two_seconds_or_more_splits(gap_s):
	t0 = datetime(2019, 8, 1, 20, 58, 23)
	runs = [DataInterval(t0, t0 + timedelta(seconds=100), 'a.cdf'),
	        DataInterval(t0 + timedelta(seconds=100 + gap_s), t0 + timedelta(seconds=200), 'b.cdf')]
	for name, tol in TOLERANCES.items():
		assert len(merge_bursts(runs, tol=tol)) == 2, name


def test_find_events_splits_where_scm_zeroed_both_sides_of_a_boundary(tmp_path, monkeypatch):
	from test_direct_cdf import T, write_cdf
	monkeypatch.setattr(ee, 'DATA_DIR', str(tmp_path / 'pydata'))
	monkeypatch.setattr(ee, 'PICKLE_ROOT', str(tmp_path / 'cache'))      # never the data drive
	n, edge = 10 * 8192, 2 * 8192
	# One continuous recording across two files (the second starts one sample after the
	# first ends); SCM flagged the last 2 s of the first and the first 2 s of the second.
	write_cdf(str(tmp_path / 'pydata'), 'B', '20210223100000', T(0), n, bad=[slice(n - edge, n)])
	write_cdf(str(tmp_path / 'pydata'), 'B', '20210223100010', T(0), n, first=n, bad=[slice(0, edge)])
	events = ee.find_events(ee.RequestRange(T(0), T(20)), ('B',), (1,), log=lambda *a, **k: None)
	assert len(events) == 2
	first, second = events
	assert first.end <= T(8.001) and second.start >= T(11.999)       # the 4 s zeroed stretch is outside both
