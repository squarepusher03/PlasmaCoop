"""Which event images get drawn: groups by start within 5 s, plus a whole-
duration image for an event at least 2x the next longest in its group."""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import event_export as ee
import event_images as ei

T0 = datetime(2021, 2, 23, 7, 0, 0)


def hms(h, m, s):
	return T0.replace(hour=h, minute=m) + timedelta(seconds=s)


def ev(probe, start, end, field='B'):
	return ee.Event(probe, field, start, end, False, False)


def summary(plans):
	return [(p.field, p.name, p.kind, p.start, p.end) for p in plans]


def test_first_example_from_graham():
	events = [ev(1, hms(7, 25, 3.04), hms(7, 31, 40.10)), ev(2, hms(7, 25, 4.20), hms(7, 31, 38.00)),
	          ev(3, hms(7, 25, 3.10), hms(7, 27, 0.00)), ev(2, hms(7, 29, 30.00), hms(7, 31, 0.00))]
	assert summary(ei.plan_images(events)) == [
		('B', 'MMS1_2021-02-23-07H-25M-03.04S.png', 'group', hms(7, 25, 3.04), hms(7, 31, 40.10)),
		('B', 'MMS2_2021-02-23-07H-29M-30.00S.png', 'group', hms(7, 29, 30.00), hms(7, 31, 0.00)),
	]


def test_long_event_example_from_graham():
	events = [ev(1, hms(7, 25, 3), hms(7, 40, 0)), ev(2, hms(7, 25, 4), hms(7, 28, 0)),
	          ev(3, hms(7, 25, 5), hms(7, 28, 10)), ev(4, hms(7, 33, 0), hms(7, 41, 30))]
	plans = ei.plan_images(events)
	assert summary(plans) == [
		('B', 'MMS1_2021-02-23-07H-25M-03.00S.png', 'long', hms(7, 25, 3), hms(7, 40, 0)),
		('B', 'MMS2_2021-02-23-07H-25M-04.00S.png', 'group', hms(7, 25, 4), hms(7, 28, 10)),
		('B', 'MMS4_2021-02-23-07H-33M-00.00S.png', 'group', hms(7, 33, 0), hms(7, 41, 30)),
	]
	long = plans[0]
	assert {e.probe for e in long.members} == {1, 2, 3, 4}          # MMS4 overlaps its end


def test_five_seconds_is_inclusive_and_measured_from_the_first_start():
	base = hms(8, 0, 0)
	plans = ei.plan_images([ev(1, base, base + timedelta(seconds=60)),
	                        ev(2, base + timedelta(seconds=5.0), base + timedelta(seconds=60)),
	                        ev(3, base + timedelta(seconds=9.0), base + timedelta(seconds=60))])
	assert [len(p.members) for p in plans] == [2, 1]


def test_just_under_twice_as_long_is_not_long():
	base = hms(9, 0, 0)
	plans = ei.plan_images([ev(1, base, base + timedelta(seconds=199)),
	                        ev(2, base + timedelta(seconds=1), base + timedelta(seconds=101))])
	assert [p.kind for p in plans] == ['group']


def test_fields_get_their_own_images_and_names_never_collide():
	base = hms(10, 0, 0)
	events = [ev(p, base + timedelta(seconds=p), base + timedelta(seconds=300 if p == 1 else 60), f)
	          for f in ('B', 'E') for p in (1, 2, 3)]
	plans = ei.plan_images(events)
	assert [p.field for p in plans] == ['B', 'B', 'E', 'E']
	for f in ('B', 'E'):
		names = [p.name for p in plans if p.field == f]
		assert len(names) == len(set(names))


def test_fft_windows_cover_every_probe_for_every_image():
	base = hms(11, 0, 0)
	plans = ei.plan_images([ev(1, base, base + timedelta(seconds=30)), ev(2, base, base + timedelta(seconds=40), 'E')])
	windows = ei.fft_windows(plans, (1, 2, 3))
	assert set(windows) == {('B', 1), ('B', 2), ('B', 3), ('E', 1), ('E', 2), ('E', 3)}
	assert windows[('B', 3)] == [(base, base + timedelta(seconds=30), False)]


def test_a_grouped_row_names_its_image_after_the_probe_that_started_it():
	"""Expanded from a grouped row, every probe shares the group's start. The
	image must still be named after the probe whose event started the group --
	the name an ungrouped run gives it -- not after the lowest probe number.
	(Found on real data: a grouped re-run wrote MMS1_... beside the MMS2_...
	image of the same event.)"""
	mms2 = ev(2, hms(12, 0, 0.00), hms(12, 3, 0))
	mms1 = ev(1, hms(12, 0, 1.50), hms(12, 3, 5))
	group = ee.group_events([mms1, mms2])[0]
	assert [e.probe for e in group.as_events()] == [2, 1]
	grouped = ei.plan_images(group.as_events())
	ungrouped = ei.plan_images([mms2, mms1])
	assert [p.name for p in grouped] == [p.name for p in ungrouped] == ['MMS2_2021-02-23-12H-00M-00.00S.png']


def test_private_helpers_still_take_what_event_images_passes():
	"""event_images composes private helpers; a signature change must fail here,
	not at the end of a long export."""
	import inspect
	from short_interval import ShortIntervalPlotter
	inspect.signature(ShortIntervalPlotter._bin_fft_files).bind(
		None, fft_paths=[], key='B', pad=0.25, start=T0, end=T0, binning=None, step=0.05, scale=None)
	inspect.signature(ShortIntervalPlotter._draw_onto).bind(
		None, ax=None, mean_g=None, time_bins=None, ye=None, bins=None, n_chan=0, vmi=0, vma=0, key='B',
		pad=0.25, start=T0, display_dt=None, is_top_row=False, is_bottom_row=True, is_left_col=True,
		probe=1, fce=None)
	inspect.signature(ee._event_fft_paths).bind(None)
	inspect.signature(ee._figure_title).bind('B')
	inspect.signature(ee.collect_fce).bind('data', 'pickles', T0, T0, probe=1, download=False)
	inspect.signature(ee.prepare_export).bind(None, [], extra_windows={})
