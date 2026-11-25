"""event_export's FGM coverage: full / partial / none per event, with FGM runs
merged over gaps shorter than EVENT_GAP_S (1.5 s since 2026-10-01; 2 s before)."""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import event_export as ee

T0 = datetime(2021, 2, 23, 7, 25, 0)


def s(sec):
	return T0 + timedelta(seconds=sec)


def test_run_spanning_the_event_is_full():
	cov = ee.coverage_of(s(10), s(70), [(s(0), s(100))])
	assert (cov.state, cov.fraction, cov.percent, cov.gaps) == (ee.FGM_FULL, 1.0, 100, ())


def test_gaps_shorter_than_the_event_gap_still_count_as_full():
	# edges short by 1.0 s, and a 1.4 s hole inside -- all under EVENT_GAP_S (1.5 s)
	cov = ee.coverage_of(s(10), s(70), [(s(11.0), s(40)), (s(41.4), s(69.0))])
	assert cov.state == ee.FGM_FULL


def test_a_hole_between_the_event_gap_and_two_seconds_now_counts():
	# 1.9 s was bridged while the gap was 2 s; at 1.5 s it is a real hole
	cov = ee.coverage_of(s(0), s(100), [(s(0), s(50)), (s(51.9), s(100))])
	assert cov.state == ee.FGM_PARTIAL and cov.gaps == ((s(50), s(51.9)),)


def test_partial_reports_the_share_and_the_gaps():
	cov = ee.coverage_of(s(0), s(100), [(s(0), s(62))])
	assert cov.state == ee.FGM_PARTIAL
	assert abs(cov.fraction - 0.62) < 1e-9
	assert cov.percent == 62
	assert cov.gaps == ((s(62), s(100)),)


def test_a_two_second_hole_counts():
	cov = ee.coverage_of(s(0), s(100), [(s(0), s(50)), (s(52), s(100))])
	assert cov.state == ee.FGM_PARTIAL and cov.gaps == ((s(50), s(52)),)
	assert cov.percent == 98


def test_no_overlap_is_none():
	assert ee.coverage_of(s(10), s(20), [(s(0), s(10)), (s(20), s(30))]).state == ee.FGM_NONE
	assert ee.coverage_of(s(10), s(20), []).percent == 0


def test_partial_never_displays_as_0_or_100():
	tiny = ee.coverage_of(s(0), s(1000), [(s(0), s(0.5))])
	assert tiny.state == ee.FGM_NONE or tiny.percent >= 1
	almost = ee.coverage_of(s(0), s(10_000), [(s(0), s(9_997))])
	assert almost.state == ee.FGM_PARTIAL and almost.percent == 99


def test_fgm_coverage_reads_runs_once_per_probe():
	evs = [ee.Event(1, 'B', s(0), s(60), False, False), ee.Event(1, 'E', s(0), s(60), False, False),
	       ee.Event(2, 'B', s(0), s(60), False, False)]
	calls = []

	def runs_for(probe):
		calls.append(probe)
		return [(s(0), s(60))] if probe == 1 else []
	rng = ee.RequestRange(s(-60), s(120))
	cov = ee.fgm_coverage(evs, rng, runs_for=runs_for)
	assert calls == [1, 2]
	assert [cov[e].state for e in evs] == [ee.FGM_FULL, ee.FGM_FULL, ee.FGM_NONE]
