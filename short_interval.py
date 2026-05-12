"""
Single-threaded spectrogram plotter for short time intervals (seconds to ~10 minutes).

For short intervals you typically have only one or two CDF files, so the overhead
of ProcessPoolExecutor (used by gen_fft) far outweighs any parallel speedup.
This module runs everything synchronously on the calling thread: CDF->pickle
conversion, FFT computation, and drawing all happen in one shot.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

from ffttest import (
	SCB_FS,
	find_cdf_files,
	ensure_data_pickle,
	find_data_pickles,
	_compute_file_ffts,
)


class ShortIntervalPlotter:
	"""
	Computes and plots log-binned power spectrograms for short time windows.

	Supports single-probe single-window via plot(...) and multi-probe x multi-window
	stacked grids via plot_probes(...).
	"""

	def __init__(self, data_dir: str, pickle_root: str):
		self.data_dir    = data_dir
		self.pickle_root = pickle_root

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def plot(
		self,
		key:        str,
		pad:        float,
		start:      datetime,
		end:        datetime,
		lpf_lim:    float = 1000,
		num_bins:   int   = 40,
		display_dt: float = None,
		step:       float = 0.05,
	):
		"""Single-probe (MMS1) single-window wrapper around plot_probes."""
		return self.plot_probes(
			probes=[1], key=key, start=start, end=end,
			pad=pad, lpf_lim=lpf_lim, num_bins=num_bins,
			display_dt=display_dt, step=step,
		)

	def plot_probes(
		self,
		probes:     list,
		key:        str,
		start:      datetime,
		end:        datetime,
		pad:        float = None,
		df:         float = None,
		lpf_lim:    float = 1000,
		num_bins:   int   = 40,
		display_dt = None,
		step            = 0.05,
	):
		"""
		Draw a grid of spectrograms: rows = probes, columns = window choices.

		Exactly one of `pad` or `df` must be supplied (scalar or list). If `df`
		is given, internally pad = 1 / (2 * df) elementwise.

		Memory: per-cell compute -> channelize -> bin -> draw -> discard. Only
		the small binned mean_g persists (inside the pcolormesh).
		"""
		# --- Validate pad/df exclusivity ---
		if (pad is None) == (df is None):
			raise ValueError("Exactly one of `pad` or `df` must be provided.")

		def _as_list(x):
			if isinstance(x, (list, tuple, np.ndarray)):
				return list(x)
			return [x]

		if pad is not None:
			pads = [float(p) for p in _as_list(pad)]
		else:
			pads = [1.0 / (2.0 * float(d)) for d in _as_list(df)]

		n_cols = len(pads)

		# Broadcast step
		steps = _as_list(step)
		if len(steps) == 1:
			steps = steps * n_cols
		elif len(steps) != n_cols:
			raise ValueError(f"step length {len(steps)} must match pad/df length {n_cols}.")

		# Broadcast display_dt
		if display_dt is None:
			display_dts = [None] * n_cols
		else:
			dds = _as_list(display_dt)
			if len(dds) == 1:
				display_dts = dds * n_cols
			elif len(dds) != n_cols:
				raise ValueError(f"display_dt length {len(dds)} must match pad/df length {n_cols}.")
			else:
				display_dts = dds

		n_rows = len(probes)

		sig_char = key[0]
		units = (
			r'\frac{\text{nT}^2}{\text{Hz}}'
			if sig_char == 'B'
			else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'
		)

		fig, axes = plt.subplots(
			nrows=n_rows, ncols=n_cols,
			figsize=(5 * n_cols, 2.5 * n_rows),
			squeeze=False, sharex='col', constrained_layout=True,
		)

		last_mappable = None
		for col, (p, s, dd) in enumerate(zip(pads, steps, display_dts)):
			for row, probe in enumerate(probes):
				ax = axes[row][col]
				result = self._compute_binned(
					key=key, pad=p, start=start, end=end,
					lpf_lim=lpf_lim, num_bins=num_bins,
					display_dt=dd, step=s, probe=probe,
				)
				if result is None:
					ax.text(
						0.5, 0.5, f'No data\nMMS{probe}',
						ha='center', va='center', transform=ax.transAxes,
					)
					ax.set_xticks([])
					ax.set_yticks([])
					if row == 0:
						ax.set_title(fr'$\Delta f = {int(1 / (p * 2))}$ Hz')
					if col == 0:
						ax.set_ylabel(f'MMS{probe}')
					continue

				mean_g, time_bins, ye, bins, n_chan, vmi, vma, dd_used = result
				mappable = self._draw_onto(
					ax=ax, mean_g=mean_g, time_bins=time_bins, ye=ye,
					bins=bins, n_chan=n_chan, vmi=vmi, vma=vma,
					key=key, pad=p, start=start, display_dt=dd_used,
					is_top_row=(row == 0),
					is_bottom_row=(row == n_rows - 1),
					is_left_col=(col == 0),
					probe=probe,
				)
				last_mappable = mappable

				# Free the big binned array ref held locally (matplotlib keeps its own copy inside pcolormesh)
				del mean_g, time_bins, ye, bins, result

		fig.suptitle(fr'$|{sig_char}|$ vs. Frequency vs. Time  $\left[{units}\right]$')

		if last_mappable is not None:
			cbar = fig.colorbar(last_mappable, ax=axes, location='right', shrink=0.9)
			cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')

		plt.show()

	# ------------------------------------------------------------------
	# Compute / draw helpers
	# ------------------------------------------------------------------

	def _compute_binned(
		self,
		key:        str,
		pad:        float,
		start:      datetime,
		end:        datetime,
		lpf_lim:    float,
		num_bins:   int,
		display_dt: float,
		step:       float,
		probe:      int,
	):
		"""
		Run the full CDF -> pickle -> FFT -> channelize -> bin pipeline for one
		(probe, pad) cell. Returns (mean_g, time_bins, ye, bins, n_chan, vmi, vma,
		display_dt_used) or None if no data found.

		The intermediate per-cell FFT DataFrame is dropped before this function
		returns; only the small aggregated mean_g (~10-200 KB) escapes.
		"""
		if display_dt is None:
			display_dt = step

		if key[0] == 'B':
			inst, mode, fs = 'scm', 'scb', SCB_FS
			vmi, vma = 1e-6, 1e-2
		else:
			inst, mode, fs = 'edp', 'dce', SCB_FS
			vmi, vma = 1e-4, 10

		n_expected  = int(fs * pad * 2)
		Df          = 1 / (pad * 2)
		bin_factor  = lpf_lim ** (1 / num_bins)

		# 1. Locate CDF files and convert to per-file pickles
		cdf_files = find_cdf_files(
			self.data_dir, inst, mode,
			start - timedelta(seconds=pad),
			end   + timedelta(seconds=pad),
			probe=probe,
		)
		if not cdf_files:
			return None

		for cdf in cdf_files:
			ensure_data_pickle(cdf, self.pickle_root, key, probe=probe)

		# 2. Discover the converted pickles
		data_files = find_data_pickles(
			self.pickle_root, inst, mode,
			start - timedelta(seconds=pad),
			end   + timedelta(seconds=pad),
			probe=probe,
		)
		if not data_files:
			return None

		# 3. Compute FFTs synchronously
		frames = []
		for i, (file_time, curr_path) in enumerate(data_files):
			next_path      = data_files[i + 1][1] if i + 1 < len(data_files) else None
			next_file_time = data_files[i + 1][0] if i + 1 < len(data_files) else None

			fdf = _compute_file_ffts(
				curr_path, next_path, key, inst, mode, fs,
				pad, n_expected, Df, file_time, next_file_time, lpf_lim, step,
			)
			fdf = fdf[
				(fdf['time'] >= np.datetime64(start, 'ns')) &
				(fdf['time'] <= np.datetime64(end,   'ns'))
			]
			if not fdf.empty:
				frames.append(fdf)

		if not frames:
			return None

		fdf = pd.concat(frames, ignore_index=True)
		fdf.loc[fdf['frequency'] == 0, 'frequency'] = 1

		# 4. Build log-frequency bins and channelize
		bins   = self._make_log_bins(fdf['frequency'].unique(), bin_factor, pad)
		n_chan = len(bins) + 1

		channel_map  = {f: self._channelize(f, bins) for f in fdf['frequency'].unique()}
		fdf['channel'] = fdf['frequency'].map(channel_map)

		st_ns  = np.datetime64(start, 'ns').astype(np.int64)
		t_vals = (fdf['time'].values.astype(np.int64) - st_ns) / 1e9 + 1e-12

		ye = np.arange(n_chan + 1)

		t_min, t_max = t_vals.min(), t_vals.max()
		# Offset by display_dt/2 so t_vals sit at bin centers
		time_bins    = np.arange(t_min - display_dt / 2, t_max + display_dt, display_dt)
		if len(time_bins) < 2:
			return None

		n_time = len(time_bins) - 1
		ix     = np.digitize(t_vals, time_bins) - 1
		iy     = fdf['channel'].values.astype(int)
		valid  = (ix >= 0) & (ix < n_time) & (iy >= 0) & (iy < n_chan)
		flat   = ix[valid] * n_chan + iy[valid]

		pwr    = fdf['power'].values
		sum_g  = np.bincount(flat, weights=pwr[valid], minlength=n_time * n_chan).reshape(n_time, n_chan)
		cnt_g  = np.bincount(flat, minlength=n_time * n_chan).reshape(n_time, n_chan)
		mean_g = np.full((n_time, n_chan), vmi)
		mask   = cnt_g > 0
		mean_g[mask] = sum_g[mask] / cnt_g[mask]

		# Drop the big intermediate DataFrame before returning
		del fdf, frames, ix, iy, valid, flat, sum_g, cnt_g, pwr, t_vals, channel_map, mask

		return (mean_g, time_bins, ye, bins, n_chan, vmi, vma, display_dt)

	def _draw_onto(
		self,
		ax,
		mean_g, time_bins, ye, bins, n_chan, vmi, vma,
		key, pad, start, display_dt,
		is_top_row: bool, is_bottom_row: bool, is_left_col: bool,
		probe: int,
	):
		"""Draw one spectrogram cell onto `ax`. Returns the mappable for colorbar."""
		norm = mpl.colors.LogNorm(vmin=vmi, vmax=vma)
		ax.set_facecolor(plt.cm.jet(norm(vmi)))

		mappable = ax.pcolormesh(time_bins, ye, mean_g.T, norm=norm, cmap='jet')

		# Y-axis: show 10, 100, 1000 Hz ticks where they fall within the bins
		tick_freqs = [f for f in [10, 100, 1000] if f <= bins[-1]]
		tick_chans = [self._channelize(f, bins) for f in tick_freqs]
		ax.set_ylim(bottom=self._channelize(10, bins), top=n_chan - 1)
		ax.yaxis.set_major_locator(FixedLocator(tick_chans))
		ax.set_yticklabels([str(f) for f in tick_freqs])

		if is_left_col:
			ax.set_ylabel(f'MMS{probe}\nFrequency (Hz)')
		if is_bottom_row:
			ax.set_xlabel('Time (s)')
		if is_top_row:
			ax.set_title(
				fr'{pad * 2:.3g} s window, '
				fr'$\Delta f = {int(1 / (pad * 2))}$ Hz'
			)

		return mappable

	# ------------------------------------------------------------------
	# Private helpers (unchanged)
	# ------------------------------------------------------------------

	@staticmethod
	def _make_log_bins(freqs, bin_factor: float, pad: float) -> list:
		"""Build logarithmically-spaced frequency bin edges from unique frequency values."""
		df = 1 / (pad * 2)
		h  = len(freqs) - 1
		b  = []
		while freqs[h] >= df and h > 0:
			b.append(int(freqs[h]))
			h = int(h // bin_factor)
		b.append(int(freqs[0]))
		b.reverse()
		return b

	@staticmethod
	def _channelize(x, bins: list) -> int:
		"""Map a frequency value to its log-bin channel index."""
		if x == bins[0]:
			return 0
		for i in range(1, len(bins)):
			if bins[i - 1] < x <= bins[i]:
				return i
		if x == bins[-1]:
			return len(bins)
		return 0
