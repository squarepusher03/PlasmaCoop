"""
Single-threaded spectrogram plotter for short time intervals (seconds to ~10 minutes).

For short intervals you typically have only one or two CDF files, so the overhead
of ProcessPoolExecutor (used by gen_fft) far outweighs any parallel speedup.
This module runs everything synchronously on the calling thread: CDF reads,
FFT computation, and drawing all happen in one shot.

The one exception is plot_from_fft_data: its per-(interval, probe) cell compute
+ CSV export step is independent per cell (no shared mutable state, no
matplotlib calls), so it runs on a ThreadPoolExecutor -- see that method's
Phase 1/Phase 2 split. Drawing itself stays strictly single-threaded (main
thread only), since matplotlib is not safe to call concurrently against a
shared Figure/Axes.
"""

from datetime import datetime, timedelta

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

from ffttest import (
	SAMPLE_RATE,
	field_burst,
	find_cdf_files,
	ensure_data_pickle,
	find_data_pickles,
	check_cdfs,
	cdf_check,
	find_local_cdfs,
	pair_with_next_cdf,
	_compute_file_ffts,
	cdf_time_bounds,
	get_or_compute_fft,
	_fft_pickle_path, collect_fft_data,
)

from acquisition import acquire

try:
	from spectro.data_intervals import write_request_manifest, merge_bursts, DataInterval
	from spectro.specs import BinSpec, PowerScale, empty_fill_for
except ImportError:
	from src.spectro.data_intervals import write_request_manifest, merge_bursts, DataInterval
	from src.spectro.specs import BinSpec, PowerScale, empty_fill_for

# Cap on display columns for one (probe, interval) cell. ShortIntervalPlotter is
# built for seconds-to-minutes spans; at native `step` resolution a long request
# (e.g. a multi-day capability test) would build a mean_g/cnt_g grid of columns x
# channels that blows past available memory long before matplotlib ever sees it.
# Widening the display column (see _compute_binned) keeps the grid bounded — same
# safeguard as spike_biggest_burst_fft.py's MAX_DISPLAY_COLS/effective_display_dt,
# reused here since short_interval has no capping of its own. Short requests are
# unaffected (the cap never binds at seconds-to-minutes scale).
MAX_DISPLAY_COLS = 200_000

# _bin_fft_dataframe accumulates its (time-bin, channel) histogram in row blocks
# of this many FFT rows at a time. A single long good-data chunk (e.g. a ~10-min
# burst at fine step / high lpf_lim streamed by plot_good_chunks) can be tens of
# millions of rows per probe; binning the whole array at once briefly held ~1 GB
# of per-row int64/bool temporaries and OOM'd. Counts are order-independent and
# the per-bin power sums differ only at float64 rounding (block partials summed
# in a different associativity -- ~1e-14 relative, invisible on the log-scale
# color map), so the blocked pass is numerically identical to a single pass;
# only peak memory changes. Exposed as a module constant so a test can shrink it
# to prove that block-size invariance. The mean_g/cnt_g grid itself is bounded
# by MAX_DISPLAY_COLS above; this bounds the transient per-row arrays instead.
FFT_BIN_BLOCK_ROWS = 2_000_000

# Figure width (inches) for plot_good_chunks' per-qf-chunk images. The 5-inch
# width every other path here uses is a PER-COLUMN width -- fine for the
# multi-interval grid, where N intervals sit side by side and 5*N adds up to a
# wide figure. A chunk image is always exactly one column (one contiguous burst)
# by n_rows probes, so that same 5 produced a 5-wide x 4*n_rows-tall sliver:
# with 4 probes, a 5x16 image where the time axis -- the axis a burst is
# actually read along -- was the short one. Widened here only, so the grid and
# separate-file paths keep their existing per-column sizing.
CHUNK_FIG_W = 16.0


class _FftGrid:
	"""Fixed-geometry (time-bin x channel) power accumulator for ONE cell.

	Rows are folded in one FRAME at a time and each frame is released
	immediately, so peak memory is one file's FFT rows plus the grid -- flat in
	the width of the window. That is what lets a long window be drawn at all:
	the alternative, holding every file's rows until the window is complete,
	grows with the window and reached ~50M rows for a single probe on one long
	quality-good burst (which is where it died, in the pd.concat that joined
	them).

	Geometry is fixed at construction from (start, end, pad, lpf_lim,
	num_bins, display_dt) rather than from the data, because an accumulator
	cannot wait for the last frame to learn how big it should have been. That
	is the one behavioural difference from _bin_fft_dataframe, which sizes its
	time axis to the data's own min/max: this spans the full requested window.
	Where the window IS the data extent (a qf-good chunk), they agree.

	Folding is additive -- per-bin sums and counts -- so the result is
	independent of how the rows were split across frames.
	"""

	def __init__(self, key, pad, start, end, binning, step,
	             bins, n_chan, scale=None):
		self.key = key
		self.pad = pad
		self.start = start
		self.bins = bins
		self.n_chan = n_chan
		# The ceiling is no longer baked into the FFT pickles -- they hold the
		# full spectrum up to Nyquist -- so it has to be applied here, on the way
		# in. It cannot be left implicit: _channelize returns 0 for a frequency
		# above the top bin edge, so an unmasked above-ceiling row would land in
		# the LOWEST channel rather than being dropped, quietly inflating bin 0.
		self.lpf_lim = binning.lpf_lim

		span_s = (end - start).total_seconds()
		display_dt = binning.display_dt
		if display_dt is None:
			display_dt = step
		elif display_dt < step:
			# Never finer than the FFT step -- see _bin_fft_dataframe.
			print(f'display_dt {display_dt} s is finer than the FFT step {step} s; '
			      f'using {step} s so every column has an FFT')
			display_dt = step
		if span_s > 0:
			display_dt = max(display_dt, span_s / MAX_DISPLAY_COLS)
		self.display_dt = display_dt

		# Colour-scale limits. Purely visual, so they are the caller's to pick;
		# scale=None keeps the per-field defaults this pipeline has always used.
		scale = scale if scale is not None else PowerScale.for_field(key)
		self.vmi, self.vma = scale.as_tuple()
		# What a cell with no FFT rows holds -- deliberately NOT self.vmi. The two
		# were the same number while the floor was hardcoded, which quietly made a
		# DISPLAY choice decide which CSV rows count as "no data". They stay apart.
		self.empty_fill = empty_fill_for(key)
		self.n_time = int(np.floor(span_s / display_dt + 1e-9)) + 1

		self.st_ns = np.datetime64(start, 'ns').astype(np.int64)
		self.start_ns64 = np.datetime64(start, 'ns')
		self.end_ns64 = np.datetime64(end, 'ns')

		# Allocated by the first add_partial: a grid built only to compute partials (an FFT pool worker's, see
		# _bin_fft_job) never holds the full window's sums.
		self.p_sum = None
		self.p_cnt = None
		self.saw_data = False

	@property
	def usable(self):
		return self.n_time >= 1

	def add(self, fdf):
		"""Fold one FFT frame (columns time/frequency/power) into the grid."""
		self.add_partial(self.partial(fdf))

	def add_partial(self, part):
		"""Fold one frame's partial (see partial) into the grid: the same += add has always done, so folding
		partials in file order gives the grid add would have built, bit for bit."""
		if part is None:
			return
		lo, p_sum, p_cnt = part
		if self.p_sum is None:
			self.p_sum = np.zeros((self.n_time, self.n_chan), dtype=np.float64)
			self.p_cnt = np.zeros((self.n_time, self.n_chan), dtype=np.int64)
		hi = lo + p_sum.shape[0]
		self.p_sum[lo:hi] += p_sum
		self.p_cnt[lo:hi] += p_cnt
		self.saw_data = True

	def partial(self, fdf):
		"""One FFT frame's contribution to this grid, without touching it: (lo, p_sum, p_cnt), the per-cell power
		sums and counts over the frame's own column span starting at column lo, or None when nothing lands.
		The FFT pool workers return these instead of the frame (2026-11-23, Graham: "Get rid of the binning
		scratch"): a file's spectra never leave its worker, only the cells they fall in."""
		if fdf is None or fdf.empty:
			return None
		fdf = fdf[(fdf['time'] >= self.start_ns64) & (fdf['time'] <= self.end_ns64)
		          & (fdf['frequency'] <= self.lpf_lim)]
		if fdf.empty:
			return None

		fdf = fdf.copy()
		fdf.loc[fdf['frequency'] == 0, 'frequency'] = 1

		channel_map = {f: ShortIntervalPlotter._channelize(f, self.bins)
		               for f in fdf['frequency'].unique()}
		iy = fdf['frequency'].map(channel_map).values.astype(int)

		t_vals = (fdf['time'].values.astype('datetime64[ns]').astype(np.int64)
		          - self.st_ns) / 1e9
		ix = np.floor(t_vals / self.display_dt + 1e-9).astype(int)

		valid = (ix >= 0) & (ix < self.n_time) & (iy >= 0) & (iy < self.n_chan)
		ixv, iyv = ix[valid], iy[valid]
		if ixv.size == 0:
			return None
		pw = fdf['power'].values[valid]

		# Bincount only over this frame's column span; add_partial adds it into
		# the grid slice -- the temporary never grows to the full window, only the
		# grid does (same trick as spike_biggest_burst_fft.stream_probe_matrix).
		lo, hi = int(ixv.min()), int(ixv.max())
		w = hi - lo + 1
		flat = (ixv - lo) * self.n_chan + iyv
		return (lo,
		        np.bincount(flat, weights=pw, minlength=w * self.n_chan).reshape(w, self.n_chan),
		        np.bincount(flat, minlength=w * self.n_chan).reshape(w, self.n_chan))

	def result(self):
		"""The same 8-tuple _bin_fft_dataframe returns, or None if nothing landed."""
		if not self.saw_data:
			return None
		mean_g = np.full((self.n_time, self.n_chan), self.empty_fill)
		mask = self.p_cnt > 0
		mean_g[mask] = self.p_sum[mask] / self.p_cnt[mask]
		time_bins = np.arange(self.n_time + 1) * self.display_dt
		ye = np.arange(self.n_chan + 1)
		return (mean_g, time_bins, ye, self.bins, self.n_chan,
		        self.vmi, self.vma, self.display_dt)


def _bin_fft_job(job):
	"""FFT pool worker (event_export.prepare_export): one file's FFTs, binned into every window that file feeds.

	job is (fft_job, targets): fft_job the tuple ffttest._plan_fft_jobs builds (the one _compute_fft_job takes),
	targets [(target_id, (key, pad, start, end, binning, step)), ...]. The file's spectra are computed in memory by
	the same _compute_file_ffts the pickles were written from, and each target gets _FftGrid.partial of them --
	the cells the rows fall in, which the parent folds into that window's grid. Nothing is written to disk
	(2026-11-23, Graham: "Get rid of the binning scratch"). Returns {target_id: partial} for the targets anything
	landed in."""
	fft_job, targets = job
	file_args, key, inst, mode, fs, pad, n_expected, Df, step, _stream = fft_job
	curr_path, next_path, file_time, next_file_time = file_args[:4]
	prev_path, prev_file_time = file_args[5:7] if len(file_args) >= 7 else (None, None)
	fdf = _compute_file_ffts(curr_path, next_path, key, inst, mode, fs, pad, n_expected, Df, file_time,
	                         next_file_time, step, prev_path=prev_path, prev_file_time=prev_file_time)
	out = {}
	for tid, (t_key, t_pad, start, end, binning, t_step) in targets:
		bins, n_chan = ShortIntervalPlotter._analytic_log_bins(t_pad, binning.lpf_lim, binning.num_bins)
		part = _FftGrid(t_key, t_pad, start, end, binning, t_step, bins, n_chan).partial(fdf)
		if part is not None:
			out[tid] = part
	return out


class ShortIntervalPlotter:
	"""
	Computes and plots log-binned power spectrograms for short time windows.

	Supports single-probe single-window via plot(...) and multi-probe x multi-window
	stacked grids via plot_probes(...).
	"""

	def __init__(self, data_dir: str, pickle_root: str, manifest_dir: str = None):
		self.data_dir    = data_dir
		self.pickle_root = pickle_root
		# None -> collect_fft_data/write_request_manifest fall back to
		# pickle_root (see ffttest.py) -- pass data_paths.INTERVAL_MANIFEST
		# explicitly (see main.py) to write manifests to their own directory
		# instead of mixed in with the FFT/data pickle cache.
		self.manifest_dir = manifest_dir

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
		save_dir:   str   = None,
		download:   bool  = True,
	):
		"""Single-probe (MMS1) single-window wrapper around plot_probes.

		If `save_dir` is given, the figure is saved there instead of shown
		(see plot_probes for the filename convention) and the saved path is returned.
		"""
		return self.plot_probes(
			probes=[1], key=key, start=start, end=end,
			pad=pad, lpf_lim=lpf_lim, num_bins=num_bins,
			display_dt=display_dt, step=step, save_dir=save_dir,
			download=download,
		)

	def plot_batch(
		self,
		intervals: list,
		keys:      list = ('B', 'E'),
		save_dir:  str  = './out/spectrograms',
		probes:    list = None,
		**plot_kwargs,
	):
		"""
		One saved file PER FIELD (not per interval). Within each field's figure,
		every requested interval is its own chart laid out left to right, and
		every probe is its own chart laid out top to bottom — see plot_intervals.

		intervals : list of (start, end) datetime pairs, in the order columns
		            should appear left to right
		keys      : field codes, one saved file per key, e.g. ('B', 'E')
		save_dir  : directory saved figures are written to
		probes    : probe numbers to include as rows (default [1])
		plot_kwargs : forwarded to plot_intervals (pad or df, lpf_lim, num_bins, display_dt, step)

		Returns the list of saved file paths, one per field.
		"""
		probes = probes if probes is not None else [1]
		paths = []
		for key in keys:
			path = self.plot_intervals(
				intervals=intervals, key=key, probes=probes,
				save_dir=save_dir, **plot_kwargs,
			)
			paths.append(path)
		return paths

	def plot_intervals(
		self,
		intervals: list,
		key:       str,
		probes:    list  = None,
		pad:       float = None,
		df:        float = None,
		lpf_lim:   float = 1000,
		num_bins:  int   = 40,
		display_dt       = None,
		step             = 0.05,
		save_dir:  str   = None,
		download:  bool  = True,
	):
		"""
		Draw ONE grid for a single field: columns = requested intervals, left to
		right in request order; rows = probes, top to bottom. One FFT window size
		(pad or df) is shared across every interval so the columns stay comparable.

		If `save_dir` is given, the figure is saved as one file covering every
		interval and probe for this field, and the saved path is returned.
		Otherwise the figure is shown interactively and None is returned.

		download=True: the acquisition phase runs first (see _acquire) for these
		probes over these intervals, and every cell then computes from what it
		left on disk, downloading nothing. download=False: set by callers (see
		plot_biggest_spikes.py) that already ran their own dedicated
		download/prewarm pass, so this never triggers a surprise real network
		call per (probe, interval) cell.
		"""
		if (pad is None) == (df is None):
			raise ValueError("Exactly one of `pad` or `df` must be provided.")
		pad = float(pad) if pad is not None else 1.0 / (2.0 * float(df))

		probes = probes if probes is not None else [1]
		if download:
			self._acquire(key, probes, intervals)
		n_rows = len(probes)
		n_cols = len(intervals)

		sig_char = key[0]
		units = (
			r'\frac{\text{nT}^2}{\text{Hz}}'
			if sig_char == 'B'
			else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'
		)

		fig, axes = plt.subplots(
			nrows=n_rows, ncols=n_cols,
			figsize=(5 * n_cols, 4.0 * n_rows),
			squeeze=False, constrained_layout=True,
		)

		last_mappable = None
		for col, (start, end) in enumerate(intervals):
			for row, probe in enumerate(probes):
				ax = axes[row][col]
				result = self._compute_binned(
					key=key, pad=pad, start=start, end=end,
					binning=BinSpec(lpf_lim=lpf_lim, num_bins=num_bins,
					                display_dt=display_dt),
					step=step, probe=probe,
					download=False,       # acquired above, or by the caller
				)
				if result is None:
					ax.text(
						0.5, 0.5, f'No data\nMMS{probe}',
						ha='center', va='center', transform=ax.transAxes,
					)
					ax.set_xticks([])
					ax.set_yticks([])
					if row == 0:
						ax.set_title(f'{start:%Y-%m-%d %H:%M:%S}')
					if col == 0:
						ax.set_ylabel(f'MMS{probe}')
					continue

				mean_g, time_bins, ye, bins, n_chan, vmi, vma, dd_used = result
				mappable = self._draw_onto(
					ax=ax, mean_g=mean_g, time_bins=time_bins, ye=ye,
					bins=bins, n_chan=n_chan, vmi=vmi, vma=vma,
					key=key, pad=pad, start=start, display_dt=dd_used,
					is_top_row=False,  # column header is the date (set below), not the window size
					is_bottom_row=(row == n_rows - 1),
					is_left_col=(col == 0),
					probe=probe,
				)
				last_mappable = mappable
				if row == 0:
					ax.set_title(f'{start:%Y-%m-%d %H:%M:%S}')

				# Free the big binned array ref held locally (matplotlib keeps its own copy inside pcolormesh)
				del mean_g, time_bins, ye, bins, result

		fig.suptitle(
			fr'$|{sig_char}|$ vs. Frequency vs. Time  $\left[{units}\right]$   '
			fr'({pad * 2:.3g} s window, $\Delta f = {int(1 / (pad * 2))}$ Hz)'
		)

		if last_mappable is not None:
			cbar = fig.colorbar(last_mappable, ax=axes, location='right', shrink=0.9)
			cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')

		if save_dir is not None:
			os.makedirs(save_dir, exist_ok=True)
			# min/max over every interval's start AND end — not just the first/last
			# request's start — so the filename reflects the true covered span even
			# when columns aren't in chronological order (e.g. a long test window
			# whose start predates an earlier-listed short spike).
			span_start = min(iv[0] for iv in intervals)
			span_end   = max(iv[1] for iv in intervals)
			fname = (
				f'{key}_{span_start:%Y%m%d}_to_{span_end:%Y%m%d}'
				f'_n{len(intervals)}.png'
			)
			path = os.path.join(save_dir, fname)
			fig.savefig(path, dpi=150, bbox_inches='tight')
			plt.close(fig)
			return path

		plt.show()
		return None

	def plot_from_fft_data(
		self,
		fft_by_probe: dict,
		intervals:    list,
		key:          str,
		probes:       list  = None,
		pad:          float = None,
		df:           float = None,
		lpf_lim:      float = 1000,
		num_bins:     int   = 40,
		display_dt          = None,
		step                = 0.05,
		save_dir:     str   = None,
		fce_by_col_probe: dict = None,
		separate_files: bool = False,
		export_csv_dir: str = None,
		bins_readme_dir: str = None,
		csv_layout: str = 'wide',
		max_workers: int = 8,
	):
		"""
		STREAMED pipeline: draw the same columns-are-intervals/rows-are-probes
		grid plot_intervals draws, but from FFT data already computed THIS
		runtime -- e.g. a caller that just called ffttest.prewarm() and has
		its {(key, probe): dataframe} return value in hand.

		This is NOT for reading pickles from a prior run -- that's
		plot_intervals/_compute_binned's job (collect_fft_data). For 'stream'
		columns (the default) this method does no file I/O at all: no
		find_cdf_files, no ensure_data_pickle, no collect_fft_data call
		anywhere in it.

		fft_by_probe: {probe: dataframe}, already the correct field's data --
		caller picks the right sub-dict out of prewarm's {(key, probe): df}
		return value first, e.g. `{p: fft_data[(sig_key, p)] for p in probes}`.
		Each dataframe may span more than one interval's window --
		_bin_fft_dataframe does its own final time-filter per cell, so which
		order things were computed in doesn't matter, only the real
		timestamps in the data.

		intervals: list of (start, end) or (start, end, 'full') triples.
		Plain (start, end) (or an explicit 'stream' third element) draws that
		column from fft_by_probe, same as before -- built for narrow,
		already-streamed-back windows (e.g. a spike search's WINDOW_S-second
		result). (start, end, 'full') instead draws the column via
		_stream_binned, which reads per-file FFT pickles straight off disk
		(single process, no pool -- see _stream_binned) and bins as it goes
		-- the only path in this class safe for a WIDE span (hours to
		months): fft_by_probe / _bin_fft_dataframe both
		require the whole span's FFT rows materialized in memory first, which
		is fine at WINDOW_S scale but is exactly what reproduced the
		MemoryError documented in plot_biggest_spikes.py for anything wider.
		'full' columns ignore fft_by_probe entirely and read from the
		on-disk pickle cache instead (download=False -- callers wide enough
		to need this have already run their own prewarm pass).

		fce_by_col_probe: optional {(col_index, probe): (times_ns, fce_hz)}
		-- f_ce (electron cyclotron frequency) curve data, one entry per
		drawn column/probe cell, already time-scoped to that column's own
		[start, end] (see fce.collect_fce). This method does no FGM I/O of
		its own -- same separation of concerns as fft_by_probe -- the
		caller (e.g. plot_biggest_spikes.py) fetches it, after its own
		download pass has completed, and hands it in here. col_index is the
		index into `intervals`, REGARDLESS of separate_files -- see below.

		separate_files: False (default) preserves the original contract --
		one combined figure (rows = probes, columns = every interval) saved
		to (or shown as) a single file, and this method returns that one
		path (or None if save_dir is None). True instead saves ONE file PER
		INTERVAL (rows = probes, single column), named from that interval's
		own [start, end] rather than the combined span, and this method
		returns a LIST of paths (one per interval, in `intervals` order,
		save_dir is None -> shows each figure instead and still returns a
		list, of None per interval). fce_by_col_probe keys are unaffected by
		this -- col_index is always the interval's index into `intervals`,
		never reset per output file, so the same fce data caller built for
		combined mode works unchanged in separate mode.

		Intervals on which EVERY probe came back empty are dropped before
		drawing: no panel, no file, and no entry in the returned list (so a
		separate_files call can return fewer paths than there are intervals,
		or an empty list). An all-'No data' figure is noise, and no CSV was
		written for it either.

		export_csv_dir: optional. When given, this is THIS RUN's batch folder,
		and each drawn interval gets its OWN subfolder under
		export_csv_dir/csv/{key}/{start}-{end}/, holding one power CSV per
		calendar day per probe ('fft_{key}_{YYYYMMDD}_MMS{probe}.csv'). Per
		request so a run's intervals never share a directory, and per day so a
		wide 'full' column spanning weeks doesn't collapse into one unopenable
		file -- the same layout plot_good_chunks writes, so every CSV in a run
		is addressed the same way regardless of which path produced it. An
		interval crossing midnight is split at the boundary with no row
		dropped or duplicated (see fft_excel_export.split_rows_by_day). The
		per-field 'csv/{key}' level keeps |B| and |E| separate; the caller owns
		the per-run batch identity (a fresh folder each run -- see
		plot_biggest_spikes.make_batch_run_dir), so CSVs from different runs
		never mix. None (default) writes no CSVs.

		bins_readme_dir: optional. When given, ALSO writes ONE bin-index ->
		frequency-range README for this call (not per cell) -- reuses
		fft_excel_export.write_bins_readme, same file this project already
		writes from spike_biggest_burst_fft.py, so both scripts' outputs are
		documented the same way. Taken from the FIRST cell this call
		successfully computes; if NO cell had data, the edges are derived from
		(pad, lpf_lim, num_bins) instead of skipping the file, since the
		binning is fully determined by config either way. Bins are already
		built analytically for 'full' columns (_stream_binned) and match
		across probes/intervals under a normal uniform-PAD/lpf_lim/NUM_BINS
		run, so one snapshot per call is documentation, not per-cell
		duplication. Named from the call's overall span (min/max across
		`intervals`), matching the combined PNG's own filename convention.
		None (default) writes nothing.

		csv_layout: 'wide' (default) or 'tall' (~3x faster to_csv on a long
		window -- see fft_excel_export._probe_dataframe). Only matters when
		export_csv_dir is given.

		max_workers: thread count for the per-cell compute+CSV-export pass
		(Phase 1 below). Drawing (Phase 2) always runs single-threaded on
		the calling thread regardless of this value.

		Same grid/draw/save contract as plot_intervals (when separate_files
		is False).
		"""
		if (pad is None) == (df is None):
			raise ValueError("Exactly one of `pad` or `df` must be provided.")
		pad = float(pad) if pad is not None else 1.0 / (2.0 * float(df))

		probes = probes if probes is not None else [1]
		n_rows = len(probes)

		sig_char = key[0]
		units = (
			r'\frac{\text{nT}^2}{\text{Hz}}'
			if sig_char == 'B'
			else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'
		)
		title = (
			fr'$|{sig_char}|$ vs. Frequency vs. Time  $\left[{units}\right]$   '
			fr'({pad * 2:.3g} s window, $\Delta f = {int(1 / (pad * 2))}$ Hz)'
		)

		empty_fft = pd.DataFrame(columns=['time', 'frequency', 'power'])

		span_start = min(iv[0] for iv in intervals)
		span_end   = max(iv[1] for iv in intervals)

		# export_csv_dir is THIS RUN's batch folder -- the caller owns the
		# per-run identity so re-running never overwrites a prior batch (see
		# plot_biggest_spikes.make_batch_run_dir). Every field gets its own
		# subfolder so |B| and |E| CSVs never mix:
		#   <batch>/csv/<field>/fft_<field>_<start>-<end>_MMS<probe>.csv
		batch_csv_dir = None
		if export_csv_dir is not None:
			batch_csv_dir = os.path.join(export_csv_dir, 'csv', key)
			os.makedirs(batch_csv_dir, exist_ok=True)
			print(f'[CSV export] {key} CSV dir: {batch_csv_dir}', flush=True)
		if bins_readme_dir is not None:
			os.makedirs(bins_readme_dir, exist_ok=True)

		# 'full' cells fall through to _stream_binned, which does its own
		# raw-pickle / FFT-pickle check-then-act writes (ensure_data_pickle,
		# get_or_compute_fft in ffttest.py) -- UNLOCKED by design elsewhere
		# in this project (see ffttest.prewarm's own docstring: two workers
		# racing to first-write the same not-yet-cached file is exactly the
		# hazard prewarm's file-level dedup exists to avoid). Normal callers
		# of this method prewarm the whole span first, so in practice these
		# are cache-hit reads -- but nothing here enforces that, so 'full'
		# cells are serialized against each other with one lock rather than
		# trusting caller discipline. 'stream' cells never touch disk (pure
		# fft_by_probe slicing) and are unaffected -- they run fully parallel.
		_full_mode_io_lock = threading.Lock()

		def _compute_cell(col, interval, probe):
			"""Compute one (interval, probe) cell's binned result and, if
			export_csv_dir was given, write its CSV. No matplotlib calls and
			no shared mutable state beyond the disk-write lock above, so this
			is safe to run from a thread pool (see Phase 1 below). Returns
			the same 8-tuple _stream_binned/_bin_fft_dataframe return, or
			None if there's no data for this cell."""
			start, end = interval[0], interval[1]
			col_mode = interval[2] if len(interval) > 2 else 'stream'
			if col_mode == 'full':
				with _full_mode_io_lock:
					result = self._stream_binned(
						key=key, pad=pad, start=start, end=end,
						binning=BinSpec(lpf_lim=lpf_lim, num_bins=num_bins,
						                display_dt=display_dt),
						step=step, probe=probe,
						download=False,
					)
			else:
				fdf = fft_by_probe.get(probe)
				if fdf is None:
					fdf = empty_fft
				result = self._bin_fft_dataframe(
					fdf, key, pad, start, end,
					BinSpec(lpf_lim=lpf_lim, num_bins=num_bins, display_dt=display_dt), step)

			if result is None or batch_csv_dir is None:
				return result

			mean_g, time_bins, ye, bins, n_chan, vmi, vma, dd_used = result
			from fft_excel_export import split_rows_by_day, export_day_to_csv
			# This request gets its own folder, and within it one file per
			# calendar day -- same layout the chunks path writes, so every CSV
			# in a run is addressed the same way regardless of which path
			# produced it. A 'full' column spanning a week is exactly as prone
			# to unopenable single-file blowup as a chunks request is.
			request_dir = os.path.join(
				batch_csv_dir, f'{start:%Y%m%dT%H%M%S}-{end:%Y%m%dT%H%M%S}')
			p_cnt = (mean_g != vmi).astype(int)
			for day, offsets, mean_block, cnt_block in split_rows_by_day(
					start, time_bins, mean_g, p_cnt):
				csv_path = export_day_to_csv(
					request_dir, key, probe, day, offsets, mean_block, cnt_block,
					n_chan, layout=csv_layout)
				print(f'[CSV export] wrote {csv_path}', flush=True)
			return result

		def _draw_cell(col, interval, probe, ax, is_left_col, is_bottom_row, result):
			"""Draw one already-computed (interval, probe) cell onto `ax`.
			col is always the GLOBAL index into `intervals` (for the
			fce_by_col_probe lookup) regardless of which figure/column ax
			physically sits in -- see separate_files docs above. Matplotlib
			is not safe to call concurrently against a shared Figure/Axes,
			so this (Phase 2) stays on the calling thread -- see
			_compute_cell for the parallel half. Returns the mappable, or
			None if there was no data to draw."""
			start = interval[0]
			if result is None:
				ax.text(
					0.5, 0.5, f'No data\nMMS{probe}',
					ha='center', va='center', transform=ax.transAxes,
				)
				ax.set_xticks([])
				ax.set_yticks([])
				if is_left_col:
					ax.set_ylabel(f'MMS{probe}')
				return None

			mean_g, time_bins, ye, bins, n_chan, vmi, vma, dd_used = result
			fce = fce_by_col_probe.get((col, probe)) if fce_by_col_probe else None
			mappable = self._draw_onto(
				ax=ax, mean_g=mean_g, time_bins=time_bins, ye=ye,
				bins=bins, n_chan=n_chan, vmi=vmi, vma=vma,
				key=key, pad=pad, start=start, display_dt=dd_used,
				is_top_row=False,
				is_bottom_row=is_bottom_row,
				is_left_col=is_left_col,
				probe=probe,
				fce=fce,
			)
			return mappable

		# Phase 1 (parallel): compute every (interval, probe) cell's binned
		# result and export its CSV. Each cell is independent (see
		# _compute_cell's own docstring for the one shared resource, the
		# 'full'-mode disk-write lock), so the real per-cell cost -- FFT
		# binning plus a to_csv call, which is the "takes a long time" part
		# for a wide CSV on a long 'full' column -- is spread across
		# max_workers threads instead of running strictly one cell at a
		# time. numpy/pandas and file I/O release the GIL for most of this
		# work, so this is a real wall-clock win despite still being one
		# process.
		cell_specs = [
			(col, interval, probe)
			for col, interval in enumerate(intervals)
			for probe in probes
		]
		with ThreadPoolExecutor(max_workers=max_workers) as pool:
			futures = {
				(col, probe): pool.submit(_compute_cell, col, interval, probe)
				for col, interval, probe in cell_specs
			}
			cell_results = {key_: fut.result() for key_, fut in futures.items()}

		# bins_readme_dir: written ONCE, from the first cell that actually
		# computed data in `intervals`/`probes` order -- the same "first
		# successfully computed" contract this method's docstring promises,
		# resolved here (after Phase 1, not inside a worker) so which thread
		# happens to finish first can never change which cell's bins get
		# documented.
		if bins_readme_dir is not None:
			first_result = None
			for col, _interval in enumerate(intervals):
				for probe in probes:
					r = cell_results.get((col, probe))
					if r is not None:
						first_result = r
						break
				if first_result is not None:
					break
			# No cell had data -> bins/n_chan/dd_used stay None and
			# _write_bins_readme_for derives them from config instead of
			# skipping the file. That skip is what left a run with a B README
			# and no E README whenever E had no good data overlapping the
			# B-selected spike window: the binning was fully known either way.
			bins = n_chan = dd_used = None
			if first_result is not None:
				_, _, _, bins, n_chan, _, _, dd_used = first_result
			self._write_bins_readme_for(
				bins_readme_dir, key=key, pad=pad, lpf_lim=lpf_lim,
				num_bins=num_bins, span_start=span_start, span_end=span_end,
				bins=bins, n_chan=n_chan, display_dt=dd_used, step=step,
			)

		# Intervals where EVERY probe came back empty are dropped before any
		# drawing: an all-'No data' panel is noise, not a result, and it has no
		# CSV behind it either. Kept as (original_col, interval) pairs because
		# `col` is the key into fce_by_col_probe and cell_results -- renumbering
		# would silently re-pair columns with the wrong f_ce overlay.
		drawable = [
			(col, interval) for col, interval in enumerate(intervals)
			if any(cell_results.get((col, probe)) is not None for probe in probes)
		]
		skipped = len(intervals) - len(drawable)
		if skipped:
			print(f'[{key}] skipping {skipped} interval(s) with no data on any probe', flush=True)
		if not drawable:
			return [] if separate_files else None

		# Phase 2 (sequential, main thread): draw every cell from its
		# already-computed result -- see _draw_cell for why this can't run
		# on the thread pool too.
		if not separate_files:
			n_cols = len(drawable)
			fig, axes = plt.subplots(
				nrows=n_rows, ncols=n_cols,
				figsize=(5 * n_cols, 4.0 * n_rows),
				squeeze=False, constrained_layout=True,
			)

			last_mappable = None
			for grid_col, (col, interval) in enumerate(drawable):
				start = interval[0]
				for row, probe in enumerate(probes):
					mappable = _draw_cell(
						col, interval, probe, axes[row][grid_col],
						is_left_col=(grid_col == 0), is_bottom_row=(row == n_rows - 1),
						result=cell_results.pop((col, probe), None),
					)
					if mappable is not None:
						last_mappable = mappable
					if row == 0:
						ax = axes[row][grid_col]
						ax.set_title(f'{start:%Y-%m-%d %H:%M:%S}')

			fig.suptitle(title)
			if last_mappable is not None:
				cbar = fig.colorbar(last_mappable, ax=axes, location='right', shrink=0.9)
				cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')

			if save_dir is not None:
				os.makedirs(save_dir, exist_ok=True)
				span_start = min(iv[0] for _c, iv in drawable)
				span_end   = max(iv[1] for _c, iv in drawable)
				fname = (
					f'{key}_{span_start:%Y%m%d}_to_{span_end:%Y%m%d}'
					f'_n{len(drawable)}.png'
				)
				path = os.path.join(save_dir, fname)
				fig.savefig(path, dpi=150, bbox_inches='tight')
				plt.close(fig)
				return path

			plt.show()
			return None

		# separate_files: one small (probes-tall, 1-wide) figure per interval,
		# each saved (or shown) on its own -- see docstring.
		results = []
		if save_dir is not None:
			os.makedirs(save_dir, exist_ok=True)
		for col, interval in drawable:
			start, end = interval[0], interval[1]
			fig, axes = plt.subplots(
				nrows=n_rows, ncols=1,
				figsize=(5, 4.0 * n_rows),
				squeeze=False, constrained_layout=True,
			)

			last_mappable = None
			for row, probe in enumerate(probes):
				mappable = _draw_cell(
					col, interval, probe, axes[row][0],
					is_left_col=True, is_bottom_row=(row == n_rows - 1),
					result=cell_results.pop((col, probe), None),
				)
				if mappable is not None:
					last_mappable = mappable
				if row == 0:
					axes[row][0].set_title(f'{start:%Y-%m-%d %H:%M:%S}')

			fig.suptitle(title)
			if last_mappable is not None:
				cbar = fig.colorbar(last_mappable, ax=axes, location='right', shrink=0.9)
				cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')

			if save_dir is not None:
				fname = f'{key}_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}.png'
				path = os.path.join(save_dir, fname)
				fig.savefig(path, dpi=150, bbox_inches='tight')
				plt.close(fig)
				results.append(path)
			else:
				plt.show()
				results.append(None)

		return results

	def plot_good_chunks(
		self,
		chunk_stream,
		key:       str,
		probes:    list,
		pad:       float = None,
		df:        float = None,
		lpf_lim:   float = 1000,
		num_bins:  int   = 40,
		display_dt       = None,
		step             = 0.05,
		save_dir:  str   = None,
		export_csv_dir: str = None,
		csv_layout: str = 'wide',
		bins_readme_dir: str = None,
	):
		"""STREAMED per-qf-chunk plotter: draw ONE figure per qf-good data
		chunk yielded by `chunk_stream` -- (chunk_start, chunk_end,
		{probe: [fft_pickle_path, ...]}, n_files) groups, exactly as
		ffttest.stream_good_chunk_ffts produces them. Probes are rows; every row shares the chunk's
		[chunk_start, chunk_end] window with t=0 pinned at chunk_start (the
		earliest good-qf start across the probes in the chunk), so a probe whose
		good data starts later simply begins further right. A probe with no data
		in the chunk draws a 'No data' row.

		This READS the FFT pickles the producer already wrote, one file at a
		time, via _bin_fft_files -- it does not receive the rows in memory. A
		single long quality-good burst is far too large to hold: one probe
		reached ~50M rows, and the parent died concatenating them. Paths cost
		nothing to pass and the accumulator folds each file in and drops it, so
		memory is flat in burst length. Styling still goes through the shared
		_draw_onto, so a per-chunk image looks identical to every other
		spectrogram in this module; nothing about the look is redefined here.
		Because the stream overlaps FFT compute with drawing
		(stream_good_chunk_ffts keeps a ProcessPool warming later files while
		this draws the current chunk), each savefig runs while the next chunk's
		FFTs are already being computed -- the whole point of the streamed path
		over the old compute-all-then-draw-all two-phase wait.

		A chunk on which EVERY probe binned to nothing is skipped entirely --
		no figure, no file, no entry in the returned list. An all-'No data'
		image is noise, and there is no CSV behind it either.

		save_dir given -> one PNG per drawn chunk under a per-date subfolder,
		save_dir/<YYYYMMDD>/{key}_{start}_{end}.png (grouped by the chunk's own
		start date), and returns the list of saved paths. save_dir None -> shows
		each and returns [None, ...].

		export_csv_dir given -> ALSO writes this request's power CSVs there, ONE
		FILE PER CALENDAR DAY PER PROBE (fft_{key}_{YYYYMMDD}_MMS{probe}.csv).
		Per-day rather than per-chunk because a chunks request routinely spans
		weeks: one file for the whole span would be unopenable, while one file
		per chunk would scatter hundreds of tiny files with no natural way to
		scan a date. A burst that crosses midnight is split at the boundary and
		contributes to BOTH days' files, with day N+1 continuing exactly where
		day N stopped -- no row is dropped, duplicated, or re-binned across the
		seam (see fft_excel_export.split_rows_by_day).

		Rows are buffered per (day, probe) and flushed when the stream moves
		past that day, so each file is written once, complete, with its own
		max_power/min_power summary. That is safe precisely because
		stream_good_chunk_ffts yields in strictly ascending chunk_start order:
		once a chunk from a later day arrives, no earlier day can gain rows.
		Buffering is cheap -- empty time slots are dropped, so a day holds only
		the minutes that actually recorded burst data, not 24h of grid.

		bins_readme_dir given -> writes this request's bin-index -> frequency
		map alongside, so the CSVs' bin_N columns are interpretable. Derived
		from lpf_lim/num_bins/pad alone, so it is written even for a request
		whose probes turn out to have no data at all.
		"""
		if (pad is None) == (df is None):
			raise ValueError("Exactly one of `pad` or `df` must be provided.")
		pad = float(pad) if pad is not None else 1.0 / (2.0 * float(df))

		probes = probes if probes is not None else [1]
		n_rows = len(probes)

		sig_char = key[0]
		units = (
			r'\frac{\text{nT}^2}{\text{Hz}}'
			if sig_char == 'B'
			else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'
		)
		title = (
			fr'$|{sig_char}|$ vs. Frequency vs. Time  $\left[{units}\right]$   '
			fr'({pad * 2:.3g} s window, $\Delta f = {int(1 / (pad * 2))}$ Hz)'
		)

		if save_dir is not None:
			os.makedirs(save_dir, exist_ok=True)
		if export_csv_dir is not None:
			os.makedirs(export_csv_dir, exist_ok=True)

		# Per-(day, probe) row buffers awaiting flush, plus the newest day the
		# stream has reached. See the docstring for why buffer-then-flush is
		# both safe (strictly ascending stream) and cheap (empty slots dropped).
		day_buf = {}          # (day_midnight, probe) -> [(offsets_s, mean, cnt)]
		day_n_chan = {}       # (day_midnight, probe) -> n_chan for that buffer
		newest_day = None

		def _flush_days(before_day=None):
			"""Write out every buffered day strictly older than `before_day`
			(all of them when None -- the end-of-stream drain)."""
			if export_csv_dir is None:
				return
			from fft_excel_export import export_day_to_csv
			for dkey in sorted([k for k in day_buf if before_day is None or k[0] < before_day]):
				blocks = day_buf.pop(dkey)
				day, probe = dkey
				n_chan_d = day_n_chan.pop(dkey)
				offsets = np.concatenate([b[0] for b in blocks])
				mean_b  = np.vstack([b[1] for b in blocks])
				cnt_b   = np.vstack([b[2] for b in blocks])
				# Chunks arrive in ascending order, but a day can still collect
				# blocks from several of them -- sort so the file's rows read
				# in time order regardless.
				order = np.argsort(offsets, kind='stable')
				path = export_day_to_csv(
					export_csv_dir, key, probe, day,
					offsets[order], mean_b[order], cnt_b[order], n_chan_d,
					layout=csv_layout,
				)
				print(f'[CSV export] wrote {path}', flush=True)

		results = []
		for chunk_start, chunk_end, probe_paths, _n_files in chunk_stream:
			# A chunk every probe came back empty on is not drawn -- see the
			# same rule in plot_from_fft_data. Binning is what decides "empty"
			# (a probe can appear in `probe_paths` and still bin to nothing), so this
			# is resolved below, after the per-probe pass, by discarding the
			# figure rather than by not creating it.
			fig, axes = plt.subplots(
				nrows=n_rows, ncols=1,
				figsize=(CHUNK_FIG_W, 4.0 * n_rows),
				squeeze=False, constrained_layout=True,
			)
			xspan = (chunk_end - chunk_start).total_seconds()

			last_mappable = None
			for row, probe in enumerate(probes):
				ax = axes[row][0]
				paths = probe_paths.get(probe)
				result = (
					self._bin_fft_files(paths, key, pad, chunk_start, chunk_end,
					                    BinSpec(lpf_lim=lpf_lim, num_bins=num_bins,
					                            display_dt=display_dt), step)
					if paths else None
				)
				if result is None:
					ax.text(
						0.5, 0.5, f'No data\nMMS{probe}',
						ha='center', va='center', transform=ax.transAxes,
					)
					ax.set_xticks([])
					ax.set_yticks([])
					ax.set_ylabel(f'MMS{probe}')
					continue

				mean_g, time_bins, ye, bins, n_chan, vmi, vma, dd_used = result

				# Buffer this probe's rows into their calendar days. p_cnt is
				# reconstructed as (mean_g != vmi) -- vmi is the floor the
				# binner fills every empty cell with, so this is exact, not an
				# approximation (the same reconstruction export_cell_to_csv
				# relies on).
				if export_csv_dir is not None:
					from fft_excel_export import split_rows_by_day
					p_cnt = (mean_g != vmi).astype(int)
					for day, offsets, mean_block, cnt_block in split_rows_by_day(
							chunk_start, time_bins, mean_g, p_cnt):
						dkey = (day, probe)
						day_buf.setdefault(dkey, []).append((offsets, mean_block, cnt_block))
						day_n_chan[dkey] = n_chan
						if newest_day is None or day > newest_day:
							newest_day = day

				mappable = self._draw_onto(
					ax=ax, mean_g=mean_g, time_bins=time_bins, ye=ye,
					bins=bins, n_chan=n_chan, vmi=vmi, vma=vma,
					key=key, pad=pad, start=chunk_start, display_dt=dd_used,
					is_top_row=False,
					is_bottom_row=(row == n_rows - 1),
					is_left_col=True,
					probe=probe,
				)
				# Pin every row to the chunk's shared [0, xspan] window so the
				# per-probe start offsets stay visible and comparable (a probe
				# whose good data begins later sits further right). _draw_onto
				# sets ylim, not xlim; pcolormesh alone would autoscale x to each
				# row's own data extent and silently hide that offset.
				if xspan > 0:
					ax.set_xlim(0, xspan)
				last_mappable = mappable

			# last_mappable stays None only when NO probe produced a drawable
			# grid for this chunk -- an all-'No data' figure. Drop it.
			if last_mappable is None:
				plt.close(fig)
				print(f'Skipped (no data on any probe): {key} '
				      f'{chunk_start:%Y-%m-%d %H:%M:%S}-{chunk_end:%H:%M:%S}', flush=True)
				continue

			fig.suptitle(
				f'{title}\n{chunk_start:%Y-%m-%d %H:%M:%S} – {chunk_end:%H:%M:%S}'
			)
			cbar = fig.colorbar(last_mappable, ax=axes, location='right', shrink=0.9)
			cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')

			if save_dir is not None:
				# One subfolder per calendar day (the chunk's own start date),
				# so a multi-day request's chunks group by date instead of
				# piling every day's images into one flat directory.
				day_dir = os.path.join(save_dir, f'{chunk_start:%Y-%m-%d}')
				os.makedirs(day_dir, exist_ok=True)
				fname = f'{key}_{chunk_start:%Y%m%d-T%H%M%S}_{chunk_end:%Y%m%d-T%H%M%S}.png'
				path = os.path.join(day_dir, fname)
				fig.savefig(path, dpi=150, bbox_inches='tight')
				plt.close(fig)
				results.append(path)
				print(f'Saved: {path}', flush=True)
			else:
				plt.show()
				results.append(None)

			# Every day older than the one this chunk reached is now complete
			# (the stream only moves forward), so write those files and free
			# their buffers before the next chunk's rows arrive.
			_flush_days(before_day=newest_day)

		# End of stream: whatever is still buffered is the final day (or days,
		# if the last chunk straddled midnight).
		_flush_days()

		# Written from lpf_lim/num_bins/pad alone, so a request whose probes
		# had no data at all still documents the binning its (empty) CSVs
		# would have used -- see the same fallback in plot_from_fft_data.
		if bins_readme_dir is not None:
			self._write_bins_readme_for(
				bins_readme_dir, key=key, pad=pad, lpf_lim=lpf_lim,
				num_bins=num_bins, display_dt=display_dt, step=step,
			)

		return results

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
		save_dir:   str  = None,
		download:   bool = True,
	):
		"""
		Draw a grid of spectrograms: rows = probes, columns = window choices.

		Exactly one of `pad` or `df` must be supplied (scalar or list). If `df`
		is given, internally pad = 1 / (2 * df) elementwise.

		Memory: per-cell compute -> channelize -> bin -> draw -> discard. Only
		the small binned mean_g persists (inside the pcolormesh).

		If `save_dir` is given, the figure is saved to
		`{save_dir}/{field}_{start}_{end}.png` (field-and-date differentiated so
		repeat requests never collide) instead of being shown, and the saved path
		is returned. Otherwise the figure is shown interactively and None is returned.

		download=True: the acquisition phase runs first (see _acquire) for these
		probes over [start, end]; every cell then computes without downloading.
		download=False: the caller already acquired -- see plot_intervals for why
		a caller with its own prewarm/download pass needs this.
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

		if download:
			self._acquire(key, probes, [(start, end)])
		n_rows = len(probes)

		sig_char = key[0]
		units = (
			r'\frac{\text{nT}^2}{\text{Hz}}'
			if sig_char == 'B'
			else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'
		)

		fig, axes = plt.subplots(
			nrows=n_rows, ncols=n_cols,
			figsize=(5 * n_cols, 4.0 * n_rows),
			squeeze=False, sharex='col', constrained_layout=True,
		)

		last_mappable = None
		for col, (p, s, dd) in enumerate(zip(pads, steps, display_dts)):
			for row, probe in enumerate(probes):
				ax = axes[row][col]
				result = self._compute_binned(
					key=key, pad=p, start=start, end=end,
					binning=BinSpec(lpf_lim=lpf_lim, num_bins=num_bins, display_dt=dd),
					step=s, probe=probe,
					download=False,       # acquired above, or by the caller
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

		if save_dir is not None:
			os.makedirs(save_dir, exist_ok=True)
			fname = f'{key}_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}.png'
			path = os.path.join(save_dir, fname)
			fig.savefig(path, dpi=150, bbox_inches='tight')
			plt.close(fig)
			return path

		plt.show()
		return None

	# ------------------------------------------------------------------
	# Compute / draw helpers
	# ------------------------------------------------------------------

	def _bin_fft_dataframe(
		self,
		fdf,
		key:        str,
		pad:        float,
		start:      datetime,
		end:        datetime,
		binning:    BinSpec,
		step:       float,
		scale=None,
	):
		"""
		Bin an already-computed FFT DataFrame (columns time/frequency/power)
		into a display-ready channel/time grid for one (probe, pad) cell.
		Returns (mean_g, time_bins, ye, bins, n_chan, vmi, vma,
		display_dt_used), or None if no rows fall in [start, end].

		Pure binning math -- no file I/O, no CDF/pickle knowledge. `fdf` can
		span more than [start, end] (this does its own final time-filter),
		so a caller holding a broader already-computed range doesn't need to
		pre-slice precisely -- only real timestamps matter, not ordering.

		Shared by both data-gathering pipelines: _compute_binned (reads/
		computes via collect_fft_data) and plot_from_fft_data (already-
		computed-this-runtime data, no file access at all).
		"""
		display_dt = binning.display_dt
		if display_dt is None:
			display_dt = step
		elif display_dt < step:
			# A display column narrower than the FFT step can hold no FFT centre in
			# most positions: at 0.02 s over a 0.05 s step, 3 of every 5 columns drew
			# as no-data stripes (main.py's E plot, 2026-09-29). Only a smaller step
			# gives finer time resolution, so the column widens to the step instead.
			print(f'display_dt {display_dt} s is finer than the FFT step {step} s; '
			      f'using {step} s so every column has an FFT')
			display_dt = step
		span_s = (end - start).total_seconds()
		if span_s > 0:
			display_dt = max(display_dt, span_s / MAX_DISPLAY_COLS)

		scale = scale if scale is not None else PowerScale.for_field(key)
		vmi, vma = scale.as_tuple()
		empty = empty_fill_for(key)   # no-data marker; independent of the scale
		bin_factor = binning.bin_factor

		# The frequency mask here is not cosmetic. The bin EDGES below are built
		# from the frequencies actually present in fdf, so an unmasked
		# full-spectrum frame would push the top edge out to Nyquist and quietly
		# rescale the entire y-axis. Masking first reproduces exactly the edges
		# _analytic_log_bins derives from (pad, lpf_lim, num_bins).
		fdf = fdf[
			(fdf['time'] >= np.datetime64(start, 'ns')) &
			(fdf['time'] <= np.datetime64(end, 'ns')) &
			(fdf['frequency'] <= binning.lpf_lim)
		]
		if fdf.empty:
			return None

		fdf = fdf.copy()
		fdf.loc[fdf['frequency'] == 0, 'frequency'] = 1

		# Build log-frequency bins and channelize
		bins   = self._make_log_bins(fdf['frequency'].unique(), bin_factor, pad)
		n_chan = len(bins)

		channel_map  = {f: self._channelize(f, bins) for f in fdf['frequency'].unique()}
		fdf['channel'] = fdf['frequency'].map(channel_map)

		st_ns  = np.datetime64(start, 'ns').astype(np.int64)
		ye = np.arange(n_chan + 1)

		# Time-bin range from the column's min/max directly -- reading the span
		# this way avoids materializing a full-length seconds array just to call
		# .min()/.max() on it (the per-row seconds are computed block-by-block
		# below instead). Monotonic transform, so min/max of the transformed
		# values equal the transform of the min/max -- identical bin edges.
		tmin_ns = int(fdf['time'].min().value)
		tmax_ns = int(fdf['time'].max().value)
		t_min = (tmin_ns - st_ns) / 1e9 + 1e-12
		t_max = (tmax_ns - st_ns) / 1e9 + 1e-12

		# Offset by display_dt/2 so t_vals sit at bin centers
		time_bins    = np.arange(t_min - display_dt / 2, t_max + display_dt, display_dt)
		if len(time_bins) < 2:
			return None

		n_time = len(time_bins) - 1

		# Accumulate the (time-bin, channel) histogram in ROW BLOCKS instead of
		# allocating one set of full-length (one-per-FFT-row) index arrays for
		# the whole chunk at once -- see FFT_BIN_BLOCK_ROWS. Summing per-block
		# counts is exact and per-block power sums match a single global pass to
		# float64 rounding; only the transient footprint changes.
		time_vals = fdf['time'].values      # datetime64[ns] view -- sliced per block
		chan_vals = fdf['channel'].values
		pwr_vals  = fdf['power'].values
		n_rows    = len(fdf)
		sum_flat  = np.zeros(n_time * n_chan, dtype=np.float64)
		cnt_flat  = np.zeros(n_time * n_chan, dtype=np.int64)
		for s in range(0, n_rows, FFT_BIN_BLOCK_ROWS):
			e   = min(s + FFT_BIN_BLOCK_ROWS, n_rows)
			tb  = (time_vals[s:e].astype(np.int64) - st_ns) / 1e9 + 1e-12
			ixb = np.digitize(tb, time_bins) - 1
			iyb = chan_vals[s:e].astype(int)
			vb  = (ixb >= 0) & (ixb < n_time) & (iyb >= 0) & (iyb < n_chan)
			fb  = ixb[vb] * n_chan + iyb[vb]
			sum_flat += np.bincount(fb, weights=pwr_vals[s:e][vb], minlength=n_time * n_chan)
			cnt_flat += np.bincount(fb, minlength=n_time * n_chan)

		sum_g  = sum_flat.reshape(n_time, n_chan)
		cnt_g  = cnt_flat.reshape(n_time, n_chan)
		mean_g = np.full((n_time, n_chan), empty)
		mask   = cnt_g > 0
		mean_g[mask] = sum_g[mask] / cnt_g[mask]

		# Drop the big intermediate DataFrame before returning
		del fdf, sum_flat, cnt_flat, sum_g, cnt_g, time_vals, chan_vals, pwr_vals, channel_map, mask

		return (mean_g, time_bins, ye, bins, n_chan, vmi, vma, display_dt)

	def _acquire(self, key, probes, intervals):
		"""The acquisition phase for a grid (acquisition.acquire), run before any
		cell is computed: every CDF the cells can open is downloaded, validated
		and, if corrupt, fetched again, so the computation after it never
		downloads, never validates and never meets a corrupt file (Graham,
		2026-09-27).

		Only the probes asked for are requested, each in its own SDC call (Graham,
		2026-09-28). Intervals that overlap are acquired once, as one span --
		merged by the same merge_bursts the manifests use -- so no file is fetched
		twice; intervals that don't are acquired apart, so the SDC is never asked
		for the time between them."""
		for span in merge_bursts([DataInterval(start, end, None) for start, end in intervals]):
			acquire(key, probes, span.start, span.end, data_dir=self.data_dir, pickle_root=self.pickle_root)

	def _compute_binned(
		self,
		key:        str,
		pad:        float,
		start:      datetime,
		end:        datetime,
		binning:    BinSpec,
		step:       float,
		probe:      int,
		download:   bool = True,
	):
		"""
		READ-PICKLES pipeline for one (probe, pad) cell: fetch FFT data via
		ffttest.collect_fft_data (reads already-cached pickles, computes only
		what's missing) and bin it. Returns (mean_g, time_bins, ye, bins,
		n_chan, vmi, vma, display_dt_used) or None if no data found.

		Data gathering (CDF discovery, pickling, FFT computation/caching) is
		NOT this method's job -- that's collect_fft_data's. A plotting class
		has no business finding CDFs, writing pickles, or running FFT math
		itself.

		download=False: forwarded through to collect_fft_data's own
		find_cdf_files call -- see plot_intervals.
		"""
		fdf = collect_fft_data(
			self.data_dir, self.pickle_root, key, pad, start, end,
			lpf_lim=binning.lpf_lim, step=step, probe=probe, download=download,
			manifest_dir=self.manifest_dir,
		)
		return self._bin_fft_dataframe(fdf, key, pad, start, end, binning, step)

	def _stream_binned(
		self,
		key:        str,
		pad:        float,
		start:      datetime,
		end:        datetime,
		binning:    BinSpec,
		step:       float,
		probe:      int,
		download:   bool = False,
	):
		"""
		STREAMING pipeline for one (probe, pad) cell, safe for a span of any
		width (hours to months): bins one file's FFT rows into the display
		grid at a time, so peak memory is one file's FFT rows plus the grid
		-- flat in the length of [start, end], a week bins the same as a
		second. Mirrors stream_probe_matrix in spike_biggest_burst_fft.py's
		BINNING discipline, adapted to this class's (mean_g, time_bins, ye,
		bins, n_chan, vmi, vma, display_dt) contract so it can feed the same
		_draw_onto every other compute path here uses. _compute_binned
		(collect_fft_data) and _bin_fft_dataframe's fft_by_probe path both
		fully materialize [start, end]'s FFT rows before binning -- fine at
		WINDOW_S (spike-search) scale, but that's exactly what reproduced the
		MemoryError plot_biggest_spikes.py documents for a wide window.

		UNLIKE stream_probe_matrix (which drives ffttest.gen_fft's
		ProcessPoolExecutor-based generator), this reads per-file FFT
		pickles directly off disk via get_or_compute_fft -- single process,
		no pool -- the same way collect_fft_data does. gen_fft spins up its
		OWN ProcessPoolExecutor (twice: CDF pickling, then FFT computation)
		plus a multiprocessing.Manager() PER CALL; this method is called once
		per (interval, probe) cell (e.g. 2 fields x 4 probes x N 'full'
		intervals), so an earlier version routed through gen_fft and paid
		for that many fresh process-pool spin-ups on top of whatever real
		work they did -- needless, since plot_biggest_spikes.py's Phase 1
		prewarm() already computed and cached every FFT pickle this call
		will ever touch, in ONE batched parallel pass up front. That mismatch
		(pool-per-cell instead of one shared batch) is what made 'full'
		columns "incredibly slow or looping" in practice, especially on
		Windows where process spawn is expensive. Every file here is a cache
		HIT -- get_or_compute_fft just reads it straight off disk.

		write_manifest is always False (this never writes one of its own):
		any caller wide enough to need streaming has already gone through its
		own prewarm/manifest pass, so writing it again per (field, probe)
		cell would be redundant.

		Returns the same 8-tuple _bin_fft_dataframe does, or None if no data.
		"""
		inst, mode, fs = field_burst('B' if key[0] == 'B' else 'E')   # E never follows B's selection

		n_expected = int(fs * (pad * 2))
		Df         = 1.0 / (pad * 2)

		span_s = (end - start).total_seconds()
		display_dt = binning.display_dt
		if display_dt is None:
			display_dt = step
		if span_s > 0:
			display_dt = max(display_dt, span_s / MAX_DISPLAY_COLS)

		# Resolved here and handed to the grid below. This pair used to be
		# computed and then left unused -- _FftGrid derived its own copy.
		scale = PowerScale.for_field(key)

		# Bin edges built analytically -- unlike _bin_fft_dataframe, which
		# derives them from fdf['frequency'].unique(), streaming can't wait for
		# the first file to arrive before it knows how big the accumulator
		# needs to be. See _analytic_log_bins (shared with the bin README).
		bins, n_chan = self._analytic_log_bins(pad, binning.lpf_lim, binning.num_bins)

		n_time = int(np.floor(span_s / display_dt + 1e-9)) + 1
		if n_time < 1:
			return None

		# Validated as discovered, seam neighbours included -- see ffttest.cdf_check.
		check = cdf_check(key, probe, pickle_root=self.pickle_root)
		cdf_files = find_cdf_files(self.data_dir, inst, mode, start, end, probe=probe, download=download,
		                           check=check)
		if not cdf_files:
			return None

		data_files = find_local_cdfs(self.data_dir, inst, mode, start, end, probe=probe,
		                             pickle_root=self.pickle_root, check=check)
		if not data_files:
			return None

		grid = _FftGrid(key, pad, start, end, binning, step,
		                bins, n_chan, scale=scale)

		# Neighbours from disk at both ends -- see pair_with_next_cdf.
		for (file_time, curr_path, prev_file_time, prev_path,
		     next_file_time, next_path) in pair_with_next_cdf(
				data_files, self.data_dir, inst, mode, probe, check=check):
			fft_pkl = _fft_pickle_path(self.pickle_root, inst, mode, pad, file_time, probe=probe, step=step)
			fdf = get_or_compute_fft(
				(curr_path, next_path, file_time, next_file_time, fft_pkl,
				 prev_path, prev_file_time),
				key=key, inst=inst, mode=mode, fs=fs, pad=pad,
				n_expected=n_expected, Df=Df, step=step,
			)
			# No ceiling filter needed on this line: `grid` was constructed with
			# lpf_lim above, and _FftGrid.add masks every frame on the way in.
			grid.add(fdf)
			del fdf

		return grid.result()

	def _bin_fft_files(
		self,
		fft_paths,
		key:        str,
		pad:        float,
		start:      datetime,
		end:        datetime,
		binning:    BinSpec,
		step:       float,
		scale=None,
	):
		"""Bin ALREADY-COMPUTED FFT pickles into the display grid, reading one
		file at a time. Same 8-tuple contract as _bin_fft_dataframe.

		The chunk path's counterpart to _stream_binned: identical accumulation
		(both drive _FftGrid), but the frames come from paths a producer has
		already computed rather than being computed here. That is why
		stream_good_chunk_ffts hands back paths instead of DataFrames -- the
		rows for one long burst do not fit in the parent process, and they are
		already on disk.

		A path that cannot be read is skipped rather than failing the whole
		chunk: one unreadable file should cost its own rows, not the image.
		"""
		bins, n_chan = self._analytic_log_bins(pad, binning.lpf_lim, binning.num_bins)
		grid = _FftGrid(key, pad, start, end, binning, step,
		                bins, n_chan, scale=scale)
		if not grid.usable:
			return None

		for path in fft_paths or ():
			try:
				fdf = pd.read_pickle(path)
			except Exception as exc:
				print(f'[bin] skipping unreadable FFT pickle {path}: '
				      f'{type(exc).__name__}: {exc}', flush=True)
				continue
			grid.add(fdf)
			del fdf

		return grid.result()

	def _draw_onto(
		self,
		ax,
		mean_g, time_bins, ye, bins, n_chan, vmi, vma,
		key, pad, start, display_dt,
		is_top_row: bool, is_bottom_row: bool, is_left_col: bool,
		probe: int,
		fce=None,
	):
		"""Draw one spectrogram cell onto `ax`. Returns the mappable for colorbar.

		fce: optional (times_ns, fce_hz) pair (see fce.collect_fce), already
		scoped to this cell's own [start, end] by the caller -- drawn as a
		curve on top of the pcolormesh via fce.overlay_fce. None/empty draws
		nothing. Imported lazily, only when actually overlaying, so this
		module stays free of any hard dependency on fce.py (matches fce.py's
		own "optional companion" design -- see its module docstring).
		"""
		norm = mpl.colors.LogNorm(vmin=vmi, vmax=vma)
		ax.set_facecolor(plt.cm.jet(norm(vmi)))

		mappable = ax.pcolormesh(time_bins, ye, mean_g.T, norm=norm, cmap='jet')

		if fce is not None and fce[0].size:
			from fce import overlay_fce
			overlay_fce(ax, start, bins, fce[0], fce[1])

		# Y-axis: show 10, 100, 1000 Hz ticks where they fall within the bins.
		# Set AFTER pcolormesh/overlay (not before) -- ax.plot() (the overlay)
		# triggers its own view autoscale on add, which would otherwise widen
		# the view back out to the full [0, n_chan] data extent and silently
		# undo the deliberate bottom crop below.
		tick_freqs = [f for f in [10, 100, 1000] if f <= bins[-1]]
		tick_chans = [self._channelize(f, bins) for f in tick_freqs]
		ax.set_ylim(bottom=self._channelize(10, bins), top=n_chan)
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

	@classmethod
	def _analytic_log_bins(cls, pad: float, lpf_lim: float, num_bins: int):
		"""(bins, n_chan) derived from CONFIG ALONE -- no FFT data required.

		The FFT grid's frequencies are deterministic multiples of Df = 1/(2*pad)
		up to lpf_lim, so the log-bin edges are fully determined by (pad,
		lpf_lim, num_bins) before a single sample is read. _stream_binned has
		always needed that (it must size its accumulator before the first file
		arrives); the bin README needs it for a different reason -- the README
		describes the BINNING, which exists whether or not any probe turned out
		to have data in the window, so gating it on a successfully computed
		cell was wrong. Shared here so both get the same edges by construction.
		"""
		Df = 1.0 / (pad * 2)
		bin_factor = lpf_lim ** (1 / num_bins)
		grid_freqs = np.unique((np.round(np.arange(0, lpf_lim / Df + 1)) * Df).astype(int))
		grid_freqs = grid_freqs[(grid_freqs >= 0) & (grid_freqs <= lpf_lim)]
		grid_freqs[grid_freqs == 0] = 1
		bins = cls._make_log_bins(grid_freqs, bin_factor, pad)
		return bins, len(bins)

	def _write_bins_readme_for(self, bins_readme_dir, key, pad, lpf_lim, num_bins,
	                           span_start=None, span_end=None,
	                           bins=None, n_chan=None, display_dt=None, step=0.05):
		"""Write this request's bin-index -> frequency-range README.

		bins/n_chan/display_dt come from an actually-computed cell when there
		was one (so the file reports the exact grid that was drawn/exported);
		otherwise they are derived from config via _analytic_log_bins, which
		is what makes the README appear even for a field whose probes had no
		data in the window. Before that fallback existed, a request where (say)
		E had no overlapping good data silently produced no E README at all,
		while B -- same run, same code path -- produced one.

		Filename carries the span when given; callers writing into a folder
		already named for the request (the per-request CSV folders) omit it.
		"""
		os.makedirs(bins_readme_dir, exist_ok=True)
		if bins is None or n_chan is None:
			bins, n_chan = self._analytic_log_bins(pad, lpf_lim, num_bins)
		from fft_excel_export import write_bins_readme
		meta = {
			'field': key,
			'inst': 'scm' if key[0] == 'B' else 'edp',
			'day': span_start,
			'start': span_start, 'end': span_end, 'pad': pad,
			'df': 1.0 / (2 * pad), 'lpf_lim': lpf_lim, 'num_bins': num_bins,
			'display_dt': display_dt if display_dt is not None else step,
		}
		if span_start is not None and span_end is not None:
			name = f'{key}_{span_start:%Y%m%d}_to_{span_end:%Y%m%d}_bins_README.md'
		else:
			name = f'{key}_bins_README.md'
		path = os.path.join(bins_readme_dir, name)
		write_bins_readme(path, bins, n_chan, meta)
		return path

	@staticmethod
	def _make_log_bins(freqs, bin_factor: float, pad: float) -> list:
		"""Build logarithmically-spaced frequency bin edges from unique frequency values.

		freqs is sorted ascending first: the index walk below assumes freqs[-1] is
		the top frequency. pandas' .unique() returns values in order of appearance,
		not sorted, so passing it unsorted would make the top edge some arbitrary
		mid-band frequency (non-monotonic edges that never reach lpf_lim)."""
		freqs = np.unique(np.asarray(freqs))
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
			return len(bins) - 1
		return 0
