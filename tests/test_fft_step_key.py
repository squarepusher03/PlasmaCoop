"""
FFT pickles are keyed on step as well as pad.

spike_biggest_burst_fft builds pad-0.25 pickles at STEP 0.10; plot_biggest_spikes and
event_export read pad 0.25 at 0.05. Keyed on pad alone they shared one pickle, and the
second caller got the first one's step: every other 0.05 s column empty, or two FFTs
averaged into each 0.10 s column (2026-09-29). 0.05 keeps the old folder name so the
existing cache stays valid. Synthetic CDFs in tmp_path only.

Run (the `plasma` conda env):

    python -m pytest -q tests/test_fft_step_key.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLBACKEND', 'Agg')

import ffttest
from short_interval import BinSpec, ShortIntervalPlotter, empty_fill_for
from test_direct_cdf import write_cdf, T

FT = datetime(2021, 2, 23, 10, 0, 0)


def test_default_step_keeps_the_existing_folder():
	path = ffttest._fft_pickle_path('/root', 'scm', 'scb', 0.25, FT, probe=1)
	assert path == ffttest._fft_pickle_path('/root', 'scm', 'scb', 0.25, FT, probe=1, step=0.05)
	assert path.replace(os.sep, '/').endswith('mms/1/scm/scb/fft/pad0.250_full/2021/02/23/100000.pkl')


def test_another_step_gets_its_own_folder():
	path = ffttest._fft_pickle_path('/root', 'scm', 'scb', 0.25, FT, probe=1, step=0.10)
	assert path.replace(os.sep, '/').endswith('mms/1/scm/scb/fft/pad0.250_step0.100_full/2021/02/23/100000.pkl')


def _spacing(fdf):
	t = np.sort(fdf['time'].unique())
	return round(float(np.median(np.diff(t).astype(np.int64)) / 1e9), 4)


def test_two_steps_do_not_share_a_pickle(tmp_path):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	write_cdf(data, 'B', '20210223100000', T(0), 8192 * 12)
	kw = dict(probe=1, download=False, write_manifest=False)
	coarse = ffttest.collect_fft_data(data, cache, 'Bz', 0.25, T(1), T(11), step=0.10, **kw)
	fine = ffttest.collect_fft_data(data, cache, 'Bz', 0.25, T(1), T(11), step=0.05, **kw)
	assert _spacing(coarse) == 0.10
	assert _spacing(fine) == 0.05

	# One FFT per column at 0.05 s: no empty columns.
	mean_g = ShortIntervalPlotter(None, None)._bin_fft_dataframe(
		fine, key='B', pad=0.25, start=T(1), end=T(11),
		binning=BinSpec(lpf_lim=1000, num_bins=40, display_dt=None), step=0.05)[0]
	assert not np.all(mean_g == empty_fill_for('B'), axis=1).any()
