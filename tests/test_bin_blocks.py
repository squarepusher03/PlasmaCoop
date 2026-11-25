"""
Block-size invariance test for ShortIntervalPlotter._bin_fft_dataframe.

The chunks plotting path bins a whole good-data chunk's FFT rows into a
(time-bin, channel) grid. To bound peak memory it accumulates the histogram in
row blocks of FFT_BIN_BLOCK_ROWS at a time (see short_interval.py). Bin edges,
channel count, and per-bin COUNTS are exactly block-size independent; per-bin
power sums (hence mean_g) match a single pass to float64 rounding only, since
block partials are combined in a different summation order. This test proves
that by binning the same synthetic frame at several block sizes.

Run (needs numpy/pandas/matplotlib -- the `plasma` conda env):

    python tests/test_bin_blocks.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

import short_interval
from short_interval import BinSpec, ShortIntervalPlotter


def _synthetic_fft_frame(n_rows: int, start: datetime, span_s: float, seed: int = 0):
	"""A believable FFT frame: (time, frequency, power) with repeated
	frequencies (so log-bin channelization actually collapses rows) and times
	spread across [start, start+span_s]. Includes a frequency==0 row so the
	0->1 guard is exercised."""
	rng = np.random.default_rng(seed)
	freqs = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0])
	t_offsets = np.sort(rng.uniform(0.0, span_s, size=n_rows))
	times = (np.datetime64(start, 'ns')
	         + (t_offsets * 1e9).astype('timedelta64[ns]'))
	return pd.DataFrame({
		'time': times,
		'frequency': rng.choice(freqs, size=n_rows),
		'power': rng.uniform(1e-6, 1e-2, size=n_rows),
	})


def _bin(plotter, fdf, start, end):
	return plotter._bin_fft_dataframe(
		fdf.copy(), key='B', pad=0.25, start=start, end=end,
		binning=BinSpec(lpf_lim=512, num_bins=25, display_dt=0.05), step=0.05,
	)


def test_block_size_invariance():
	plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
	start = datetime(2019, 8, 16, 9, 0, 0)
	span_s = 90.0
	end = start + timedelta(seconds=span_s + 1)
	fdf = _synthetic_fft_frame(50_000, start, span_s)

	orig = short_interval.FFT_BIN_BLOCK_ROWS
	try:
		# Single pass (block far larger than the frame) is the reference.
		short_interval.FFT_BIN_BLOCK_ROWS = 10_000_000
		ref = _bin(plotter, fdf, start, end)
		assert ref is not None
		mean_ref, tb_ref, ye_ref, bins_ref, nchan_ref, vmi, vma, dd = ref

		# Several small block sizes -- awkward divisors of 50_000 on purpose.
		max_rel = 0.0
		for block in (1, 7, 999, 12_345, 49_999, 50_000):
			short_interval.FFT_BIN_BLOCK_ROWS = block
			mean_g, tb, ye, bins_, nchan, *_ = _bin(plotter, fdf, start, end)
			# Grid shape / bin edges / channel count are EXACTLY invariant.
			assert nchan == nchan_ref, f'n_chan changed at block={block}'
			assert np.array_equal(tb, tb_ref), f'time_bins changed at block={block}'
			assert mean_g.shape == mean_ref.shape, f'grid shape changed at block={block}'
			# Means match to float64 rounding (summation-order difference only).
			assert np.allclose(mean_g, mean_ref, rtol=1e-9, atol=0.0), \
				f'mean_g diverged beyond fp rounding at block={block}'
			rel = np.max(np.abs(mean_g - mean_ref) / np.abs(mean_ref))
			max_rel = max(max_rel, float(rel))
		print(f'    max relative mean_g diff across block sizes: {max_rel:.2e}')
	finally:
		short_interval.FFT_BIN_BLOCK_ROWS = orig


def test_single_row_and_tiny_frames_bin():
	# A degenerate 1-row frame and a handful-of-rows frame must not crash and
	# must produce a grid (regression guard for the block loop's edges).
	plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
	start = datetime(2019, 8, 16, 9, 0, 0)
	end = start + timedelta(seconds=5)
	for n in (1, 3, 8):
		fdf = _synthetic_fft_frame(n, start, 4.0, seed=n)
		short_interval.FFT_BIN_BLOCK_ROWS = 2
		out = _bin(plotter, fdf, start, end)
		# May be None only if all rows fall outside the (tiny) window; with the
		# frame built inside [start, end] we expect a real grid here.
		assert out is not None
		assert out[0].shape[1] == out[4]        # mean_g columns == n_chan


def _run():
	tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
	for t in tests:
		t()
		print(f'  ok  {t.__name__}')
	print(f'{len(tests)} passed')




# ---------------------------------------------------------------------------
# _FftGrid: the streaming accumulator behind _stream_binned and _bin_fft_files.
#
# The chunks path stopped materialising a burst's rows (one probe reached ~50M
# and died in pd.concat) and now folds per-file FFT pickles into a fixed grid
# one at a time. That is only valid if folding is SPLIT-INVARIANT: the same
# rows must produce the same grid however they are divided across files.
# ---------------------------------------------------------------------------

def _grid_for(plotter, key, pad, start, end, lpf_lim, num_bins, step):
	bins, n_chan = plotter._analytic_log_bins(pad, lpf_lim, num_bins)
	return short_interval._FftGrid(key, pad, start, end,
	                               BinSpec(lpf_lim=lpf_lim, num_bins=num_bins,
	                                       display_dt=None),
	                               step, bins, n_chan)


def test_fft_grid_split_invariance():
	"""N frames folded separately == the same rows folded as one frame."""
	plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
	start = datetime(2021, 2, 23, 10, 0, 0)
	span_s = 120.0
	end = start + timedelta(seconds=span_s)
	pad, lpf_lim, num_bins, step = 0.25, 512, 20, 0.05

	whole = _synthetic_fft_frame(60_000, start, span_s, seed=11)

	one = _grid_for(plotter, 'B', pad, start, end, lpf_lim, num_bins, step)
	one.add(whole)
	ref = one.result()
	assert ref is not None

	# Split into awkward, unequal pieces -- the way real per-file boundaries fall.
	for n_parts in (2, 3, 7, 13):
		g = _grid_for(plotter, 'B', pad, start, end, lpf_lim, num_bins, step)
		cuts = np.linspace(0, len(whole), n_parts + 1).astype(int)
		for a, b in zip(cuts[:-1], cuts[1:]):
			g.add(whole.iloc[a:b])
		got = g.result()
		assert got is not None, n_parts
		assert np.array_equal(got[1], ref[1]), f'time_bins differ at {n_parts} parts'
		assert got[4] == ref[4], f'n_chan differs at {n_parts} parts'
		# Counts are exact; means differ only by float64 summation order.
		assert np.allclose(got[0], ref[0], rtol=1e-12, atol=0), \
			f'mean grid differs at {n_parts} parts'


def test_fft_grid_ignores_rows_outside_window():
	"""Rows outside [start, end] must not leak into the grid."""
	plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
	start = datetime(2021, 2, 23, 10, 0, 0)
	end = start + timedelta(seconds=30)
	pad, lpf_lim, num_bins, step = 0.25, 512, 20, 0.05

	inside = _synthetic_fft_frame(5_000, start, 30.0, seed=3)
	outside = _synthetic_fft_frame(5_000, start + timedelta(seconds=600), 30.0, seed=4)

	a = _grid_for(plotter, 'B', pad, start, end, lpf_lim, num_bins, step)
	a.add(inside)
	only_inside = a.result()

	b = _grid_for(plotter, 'B', pad, start, end, lpf_lim, num_bins, step)
	b.add(inside)
	b.add(outside)
	with_outside = b.result()

	assert np.array_equal(only_inside[0], with_outside[0]), \
		'rows outside the window changed the grid'


def test_fft_grid_empty_returns_none():
	plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
	start = datetime(2021, 2, 23, 10, 0, 0)
	g = _grid_for(plotter, 'B', 0.25, start, start + timedelta(seconds=30), 512, 20, 0.05)
	assert g.result() is None
	g.add(pd.DataFrame({'time': pd.to_datetime([]), 'frequency': [], 'power': []}))
	assert g.result() is None


def test_bin_fft_files_matches_in_memory_fold():
	"""_bin_fft_files (reads pickles) == _FftGrid fed the same frames directly."""
	import tempfile
	plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
	start = datetime(2021, 2, 23, 10, 0, 0)
	span_s = 90.0
	end = start + timedelta(seconds=span_s)
	pad, lpf_lim, num_bins, step = 0.25, 512, 20, 0.05

	frames = [_synthetic_fft_frame(8_000, start + timedelta(seconds=i * 30), 30.0, seed=20 + i)
	          for i in range(3)]

	mem = _grid_for(plotter, 'B', pad, start, end, lpf_lim, num_bins, step)
	for f in frames:
		mem.add(f)
	ref = mem.result()
	assert ref is not None

	with tempfile.TemporaryDirectory() as d:
		paths = []
		for i, f in enumerate(frames):
			p = os.path.join(d, f'{i}.pkl')
			f.to_pickle(p)
			paths.append(p)
		# A deliberately unreadable path must be skipped, not fatal.
		paths.insert(1, os.path.join(d, 'missing.pkl'))
		got = plotter._bin_fft_files(paths, 'B', pad, start, end,
		                             BinSpec(lpf_lim=lpf_lim, num_bins=num_bins,
		                                     display_dt=None), step)

	assert got is not None
	assert np.array_equal(got[0], ref[0]), 'pickle-read fold != in-memory fold'
	assert np.array_equal(got[1], ref[1])


if __name__ == '__main__':
	_run()
