"""The colour scale is visual-only, and the ceiling drops rather than folds.

Two properties that are cheap to state and expensive to lose:

  1. PowerScale moves the display limits and NOTHING else. It used to be the
     case that an empty cell was filled with the colour floor, which quietly
     made a display choice decide which CSV rows counted as "no data" -- so the
     moment the scale became a user option, re-scaling a plot would have changed
     the exported file. These tests pin the two apart.

  2. A row above the frequency ceiling is dropped. This matters more than it
     looks: _channelize returns 0 for any frequency above the top bin edge, so
     an unmasked above-ceiling row lands in the LOWEST channel and inflates it,
     rather than being discarded. Nothing about that failure is visible -- the
     image and the CSV both just quietly read too high at the bottom of the
     band. test_channelize_would_fold_above_ceiling_rows_into_channel_zero
     records why the mask has to exist at all.
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import event_export as ee
from short_interval import BinSpec, ShortIntervalPlotter, _FftGrid

LPF = 3000
START = datetime(2021, 2, 23, 10, 30)
END = START + timedelta(seconds=1)
IN_BAND = (10, 100, 1000)
ABOVE = 4000                      # above LPF, and below Nyquist, so it is really in the pickle


@pytest.fixture(scope='module')
def grid_shape():
	return ShortIntervalPlotter._analytic_log_bins(ee.PAD, LPF, ee.NUM_BINS)


def _frame(freqs):
	"""Rows in three time cells only, so most of the grid stays empty."""
	rows = []
	for offset in (0.0, 0.05, 0.50):
		stamp = np.datetime64(START + timedelta(seconds=offset), 'ns')
		rows.extend((stamp, f, 1e-3) for f in freqs)
	return pd.DataFrame(rows, columns=['time', 'frequency', 'power'])


def _binned(grid_shape, scale, freqs):
	bins, n_chan = grid_shape
	grid = _FftGrid('B', ee.PAD, START, END,
	                BinSpec(lpf_lim=LPF, num_bins=ee.NUM_BINS, display_dt=0.05), 0.05,
	                bins, n_chan, scale=scale)
	grid.add(_frame(freqs))
	return grid.result()


def test_default_scale_is_the_field_default(grid_shape):
	result = _binned(grid_shape, None, IN_BAND)
	assert (result[5], result[6]) == ee.PowerScale.for_field('B').as_tuple()


def test_a_supplied_scale_reaches_the_display_limits(grid_shape):
	result = _binned(grid_shape, ee.PowerScale(1e-9, 1e-1), IN_BAND)
	assert (result[5], result[6]) == (1e-9, 1e-1)


def test_the_scale_does_not_change_a_single_binned_value(grid_shape):
	default = _binned(grid_shape, None, IN_BAND)
	rescaled = _binned(grid_shape, ee.PowerScale(1e-9, 1e-1), IN_BAND)
	assert np.array_equal(default[0], rescaled[0])


def test_empty_cells_keep_the_fixed_marker_not_the_new_floor(grid_shape):
	fill = ee.empty_fill_for('B')
	rescaled = _binned(grid_shape, ee.PowerScale(1e-9, 1e-1), IN_BAND)
	assert (rescaled[0] == fill).sum() > 0, 'the fixture should leave empty cells'
	assert not (rescaled[0] == 1e-9).any(), 'an empty cell took the display floor'


def test_the_csv_is_byte_identical_across_scales(tmp_path, grid_shape):
	written = []
	for name, scale in (('a.csv', None), ('b.csv', ee.PowerScale(1e-9, 1e-1))):
		mean_g, time_bins, _ye, _bins, n_chan, _vmi, _vma, _dd = _binned(grid_shape, scale, IN_BAND)
		path = ee.write_event_csv(str(tmp_path / name), START, time_bins, mean_g, 'B', n_chan, 'tall')
		written.append(open(path, 'rb').read())
	assert written[0] == written[1]


def test_above_ceiling_rows_are_dropped_not_folded(grid_shape):
	without = _binned(grid_shape, None, IN_BAND)
	with_above = _binned(grid_shape, None, IN_BAND + (ABOVE,))
	assert np.array_equal(without[0], with_above[0])


def test_channelize_would_fold_above_ceiling_rows_into_channel_zero(grid_shape):
	"""Why the mask exists. If this ever stops being 0, re-read the mask's reason."""
	bins, _n_chan = grid_shape
	assert ABOVE > bins[-1]
	assert ShortIntervalPlotter._channelize(ABOVE, bins) == 0


@pytest.mark.parametrize('vmin, vmax', [(0, 1e-2), (1e-2, 1e-6), (-1, 1), (1e-6, 1e-6)])
def test_power_scale_rejects_a_range_that_cannot_be_drawn(vmin, vmax):
	with pytest.raises(ValueError):
		ee.PowerScale(vmin, vmax)
