"""
A display column is never finer than the FFT step.

main.py's E plot asked for display_dt=0.02 over FFTs computed every 0.05 s, so
only 2 of every 5 columns held an FFT centre and the rest drew as no-data
stripes (62% of columns empty, 2026-09-29). Both binners now widen display_dt
to the step; display_dt >= step is untouched. Synthetic frames only.

Run (the `plasma` conda env):

    python -m pytest -q tests/test_display_dt_guard.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

from short_interval import BinSpec, ShortIntervalPlotter, _FftGrid, empty_fill_for

STEP = 0.05
PAD = 0.1
START = datetime(2019, 8, 16, 9, 31, 58)
SPAN_S = 12.0
END = START + timedelta(seconds=SPAN_S)


def _every_step_frame():
	"""One FFT row per frequency at every STEP from START to END, like a
	gap-free pickle: 5 Hz bins (pad 0.1) up to 500 Hz."""
	times = np.datetime64(START, 'ns') + (np.arange(0.0, SPAN_S + 1e-9, STEP) * 1e9).astype('timedelta64[ns]')
	freqs = np.arange(0.0, 500.0 + 1e-9, 1.0 / (2 * PAD))
	t, f = np.meshgrid(times, freqs, indexing='ij')
	rng = np.random.default_rng(0)
	return pd.DataFrame({
		'time': t.ravel(),
		'frequency': f.ravel(),
		'power': rng.uniform(1e-4, 1e-1, size=t.size),
	})


def _empty_columns(mean_g, key='E'):
	return int(np.all(mean_g == empty_fill_for(key), axis=1).sum())


def _plotter():
	return ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')


def _bin(display_dt):
	return _plotter()._bin_fft_dataframe(
		_every_step_frame(), key='E', pad=PAD, start=START, end=END,
		binning=BinSpec(lpf_lim=500, num_bins=40, display_dt=display_dt), step=STEP)


def test_bin_fft_dataframe_widens_display_dt_to_step():
	mean_g, *_, dd_used = _bin(0.02)
	assert dd_used == STEP
	assert _empty_columns(mean_g) == 0


def test_bin_fft_dataframe_leaves_display_dt_at_or_above_step():
	for dd in (STEP, 0.1, 0.5):
		assert _bin(dd)[-1] == dd


def test_fft_grid_widens_display_dt_to_step():
	plotter = _plotter()
	bins, n_chan = plotter._analytic_log_bins(PAD, 500, 40)
	grid = _FftGrid('E', PAD, START, END, BinSpec(lpf_lim=500, num_bins=40, display_dt=0.02), STEP,
	                bins, n_chan)
	assert grid.display_dt == STEP
	grid.add(_every_step_frame())
	mean_g = grid.result()[0]
	assert _empty_columns(mean_g) == 0


def test_fft_grid_leaves_display_dt_at_or_above_step():
	plotter = _plotter()
	bins, n_chan = plotter._analytic_log_bins(PAD, 500, 40)
	for dd in (STEP, 0.1):
		grid = _FftGrid('E', PAD, START, END, BinSpec(lpf_lim=500, num_bins=40, display_dt=dd), STEP,
		                bins, n_chan)
		assert grid.display_dt == dd
