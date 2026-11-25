"""Grouping probes across events (starts within 2 s of the group's first start)
and leaving probes out of the found events."""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import event_export as ee

T0 = datetime(2021, 2, 23, 7, 25, 0)


def at(sec):
	return T0 + timedelta(seconds=sec)


def ev(probe, start, end, field='B', cs=False, ce=False):
	return ee.Event(probe, field, at(start), at(end), cs, ce)


# Graham's example: MMS1/2/4 start within 2 s, MMS3 starts minutes later.
EXAMPLE = [ev(1, 3.0, 400.0), ev(2, 4.2, 398.0), ev(4, 4.9, 422.0), ev(3, 180.0, 300.0)]


def test_groups_starts_within_two_seconds():
	groups = ee.group_events(EXAMPLE)
	assert [g.probes for g in groups] == [(1, 2, 4), (3,)]
	first = groups[0]
	assert (first.start, first.end) == (at(3.0), at(422.0))


def test_leaving_a_probe_out_recalculates_start_and_end():
	groups = ee.group_events(ee.without_probes(EXAMPLE, {1}))
	assert groups[0].probes == (2, 4)
	assert (groups[0].start, groups[0].end) == (at(4.2), at(422.0))
	assert ee.without_probes(EXAMPLE, {3, 4}) == EXAMPLE[:2]


def test_boundary_is_inclusive_and_not_chained():
	groups = ee.group_events([ev(1, 0, 10), ev(2, 2.0, 10), ev(3, 3.5, 10)])
	assert [g.probes for g in groups] == [(1, 2), (3,)]              # 3.5 s from the FIRST start
	assert [g.probes for g in ee.group_events([ev(1, 0, 10), ev(2, 2.01, 10)])] == [(1,), (2,)]


def test_fields_group_separately_and_sort_by_start():
	groups = ee.group_events([ev(1, 5, 50, 'E'), ev(2, 5.5, 50, 'B'), ev(1, 5, 60, 'B')])
	assert [(g.field, g.probes) for g in groups] == [('B', (1, 2)), ('E', (1,))]


def test_as_events_spans_the_group_for_every_probe():
	g = ee.group_events([ev(1, 0, 100, cs=True), ev(2, 1, 150), ev(3, 1.5, 90, ce=True)])[0]
	out = g.as_events()
	assert [e.probe for e in out] == [1, 2, 3]
	assert all((e.start, e.end, e.clipped_start, e.clipped_end) == (at(0), at(150), True, True) for e in out)
	assert g.duration_s == 150
