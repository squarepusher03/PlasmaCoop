from datetime import datetime, timedelta

import os
import glob
import re
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from multiprocessing import Manager

_corrupted_log_lock = threading.Lock()

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import log

import scipy
from scipy.signal import windows
from matplotlib.ticker import FuncFormatter, MultipleLocator, FixedLocator, \
	FormatStrFormatter
from scipy.stats import binned_statistic_2d

SCB_FS = 8_192
SCHB_FS = 16_384


def find_cdf_files(data_dir, inst, mode, start, end, probe: int = 1):
	"""
	Return a sorted list of CDF paths under data_dir/mms{probe}/{inst}/brst/l2/{mode}
	covering [start, end].

	Scans all files, sorts by timestamp, then anchors to the rightmost file whose
	start time is <= start (that file contains data at the beginning of the range).
	All files from that anchor up through end are returned.
	"""
	pattern = os.path.join(data_dir, f'mms{probe}/{inst}/brst/l2/{mode}/**/*.cdf')
	all_files = []
	for path in glob.glob(pattern, recursive=True):
		m = re.search(r'_(\d{14})_', path)
		if m:
			ftime = datetime.strptime(m.group(1), '%Y%m%d%H%M%S')
			all_files.append((ftime, path))
	all_files.sort()

	if not all_files:
		return []

	anchor = 0
	for i in range(len(all_files) - 1, -1, -1):
		if all_files[i][0] <= start:
			anchor = i
			break

	return [path for ftime, path in all_files[anchor:] if ftime <= end]


def find_data_pickles(pickle_root, inst, mode, start, end, probe: int = 1):
	"""
	Same discovery logic as find_cdf_files but for per-file data pickles.
	Returns sorted list of (file_start_time, path) pairs.
	Pickle filenames follow the convention: {YYYY}/{MM}/{DD}/{HHMMSS}.pkl
	"""
	pattern = os.path.join(pickle_root, f'mms/{probe}/{inst}/{mode}/*/*/*/*.pkl')
	all_files = []
	for path in glob.glob(pattern):
		m = re.search(r'(\d{4})[/\\](\d{2})[/\\](\d{2})[/\\](\d{6})\.pkl$', path)
		if m:
			ftime = datetime.strptime(''.join(m.groups()), '%Y%m%d%H%M%S')
			all_files.append((ftime, path))
	all_files.sort()

	if not all_files:
		return []

	anchor = 0
	for i in range(len(all_files) - 1, -1, -1):
		if all_files[i][0] <= start:
			anchor = i
			break

	return [(ftime, path) for ftime, path in all_files[anchor:] if ftime <= end]


def _cdf_to_df(cdf_path, key, probe: int = 1):
	"""Load a single CDF file into a DataFrame with columns time, {f}x, {f}y, {f}z."""
	from spacepy import pycdf
	f = key[0]
	with pycdf.CDF(cdf_path) as cdf:
		if f == 'B':
			var  = cdf[f'mms{probe}_scm_acb_gse_scb_brst_l2'][:]
			time = pd.to_datetime(cdf['Epoch'][:])
		else:
			var  = cdf[f'mms{probe}_edp_dce_gse_brst_l2'][:]
			time = pd.to_datetime(cdf[f'mms{probe}_edp_epoch_brst_l2'][:])
	return pd.DataFrame({
		'time': time,
		f'{f}x': var[:, 0].copy(),
		f'{f}y': var[:, 1].copy(),
		f'{f}z': var[:, 2].copy(),
	})


def ensure_data_pickle(cdf_path, pickle_root, key, corrupted_log=None, lock=None, probe: int = 1):
	"""
	Ensure a per-file data pickle exists for cdf_path.
	If not, generate it from the CDF and save it.
	Pickle path: {pickle_root}/mms/{probe}/{inst}/{mode}/{14-digit-timestamp}.pkl
	Returns the pickle path.
	"""
	m = re.search(r'_(\d{4})(\d{2})(\d{2})(\d{6})_', cdf_path)
	if not m:
		raise ValueError(f'Cannot parse timestamp from CDF path: {cdf_path}')
	timestamp = m.group(4)

	f    = key[0]
	inst = 'scm' if f == 'B' else 'edp'
	mode = 'scb' if f == 'B' else 'dce'

	pickle_path = os.path.join(pickle_root,
	                           f'mms/{probe}/{inst}/{mode}/{m.group(1)}/{m.group(2)}/{m.group(3)}/{timestamp}.pkl')

	if not os.path.exists(pickle_path):
		print(f'Generating data pickle from {os.path.basename(cdf_path)}...')
		try:
			df = _cdf_to_df(cdf_path, key, probe=probe)
		except Exception as e:
			size = os.path.getsize(cdf_path)
			print(f'Bad CDF, deleting: {os.path.basename(cdf_path)} — {size} bytes, {type(e).__name__}: {e}')
			try:
				os.remove(cdf_path)
			except PermissionError:
				print(f'Cannot delete locked file: {os.path.basename(cdf_path)}')
			if corrupted_log:
				with (lock or _corrupted_log_lock):
					with open(corrupted_log, 'a') as f:
						f.write(cdf_path + '\n')
			return None
		os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
		df.to_pickle(pickle_path)
		print(f'Saved: {pickle_path}')

	return pickle_path


def _fft_pickle_path(pickle_root, inst, mode, pad, file_time, probe: int = 1):
	"""
	Path for the per-file FFT result pickle.
	{pickle_root}/mms/{probe}/{inst}/{mode}/fft/pad{pad:.3f}/{YYYY}/{MM}/{DD}/{HHMMSS}.pkl
	Each pad value gets its own subdirectory since it determines frequency resolution.
	"""
	return os.path.join(
		pickle_root,
		f'mms/{probe}/{inst}/{mode}/fft/pad{pad:.3f}',
		file_time.strftime('%Y/%m/%d/%H%M%S.pkl'),
	)


def _compute_file_ffts(curr_path, next_path, key, inst, mode, fs, pad, n_expected, Df, file_time, next_file_time, lpf_lim, step=0.05):
	"""
	Compute FFT windows whose centers lie in [file_time, next_file_time).
	Uses data from curr_path, with next_path loaded on demand for forward boundary overlap.

	Per-window policy:
	  n == 0                          skip entirely (no data in window)
	  next file available in window:
	    combined n >= n_expected      compute FFT
	    combined n < n_expected       store zeros
	  no next file in window:
	    n >= n_expected // 2          compute FFT with frequency snapping
	    n <  n_expected // 2          store zeros
	"""
	curr_df      = pd.read_pickle(curr_path)
	if curr_df.empty:
		return pd.DataFrame(columns=['time', 'frequency', 'power'])
	curr_time_ns = curr_df['time'].values.astype(np.int64)
	curr_x_arr   = curr_df[key[0] + 'x'].values
	curr_y_arr   = curr_df[key[0] + 'y'].values
	curr_z_arr   = curr_df[key[0] + 'z'].values

	next_df           = None
	next_time_ns      = None
	next_x_arr        = None
	next_y_arr        = None
	next_z_arr        = None

	step_ns           = int(step * 1e9)
	pad_ns            = int(pad  * 1e9)
	file_time_ns      = np.datetime64(file_time, 'ns').astype(np.int64).item()
	start_ns          = max(file_time_ns, curr_time_ns[0] - pad_ns)
	next_file_time_ns = (np.datetime64(next_file_time, 'ns').astype(np.int64).item()
	                     if next_file_time is not None else None)
	data_end_ns = int(curr_time_ns[-1]) + pad_ns + step_ns
	end_ns      = min(next_file_time_ns, data_end_ns) if next_file_time_ns is not None else data_end_ns

	freq_grid = (np.round(np.arange(0, lpf_lim / Df + 1)) * Df).astype(int)

	n_steps      = (end_ns - start_ns) // step_ns + 2
	max_per_step = len(freq_grid)
	max_rows     = n_steps * max_per_step
	t_arr = np.empty(max_rows, dtype='datetime64[ns]')
	f_arr = np.empty(max_rows, dtype=np.int16)
	p_arr = np.empty(max_rows, dtype=np.float32)
	row   = 0

	center_ns = start_ns
	print(f'Computing FFTs for {file_time}...')

	while center_ns < end_ns:
		fft_start_ns = center_ns - pad_ns
		fft_end_ns   = center_ns + pad_ns

		lo  = np.searchsorted(curr_time_ns, fft_start_ns, side='left')
		hi  = np.searchsorted(curr_time_ns, fft_end_ns,   side='right')
		n   = hi - lo
		center_dt64 = np.datetime64(center_ns, 'ns')

		if n == 0:
			center_ns += step_ns
			continue

		if n < n_expected:
			next_in_window = (next_file_time_ns is not None and next_file_time_ns <= fft_end_ns)

			if next_in_window:
				if next_df is None:
					next_df      = pd.read_pickle(next_path)
					next_time_ns = next_df['time'].values.astype(np.int64)
					next_x_arr   = next_df[key[0] + 'x'].values
					next_y_arr   = next_df[key[0] + 'y'].values
					next_z_arr   = next_df[key[0] + 'z'].values
				last_time_ns = curr_time_ns[hi - 1]
				n_needed     = n_expected - n
				lo_n = np.searchsorted(next_time_ns, last_time_ns, side='right')
				hi_n = np.searchsorted(next_time_ns, fft_end_ns,   side='right')
				hi_n_clip = min(lo_n + n_needed, hi_n)
				if lo_n < hi_n_clip:
					x_win = np.concatenate([curr_x_arr[lo:hi], next_x_arr[lo_n:hi_n_clip]])
					y_win = np.concatenate([curr_y_arr[lo:hi], next_y_arr[lo_n:hi_n_clip]])
					z_win = np.concatenate([curr_z_arr[lo:hi], next_z_arr[lo_n:hi_n_clip]])
					n = len(x_win)
				if n < n_expected:
					k = len(freq_grid)
					t_arr[row:row+k] = center_dt64
					f_arr[row:row+k] = freq_grid
					p_arr[row:row+k] = 0.0
					row += k
					center_ns += step_ns
					continue
			else:
				if n < n_expected // 2:
					k = len(freq_grid)
					t_arr[row:row+k] = center_dt64
					f_arr[row:row+k] = freq_grid
					p_arr[row:row+k] = 0.0
					row += k
					center_ns += step_ns
					continue

		if n == hi - lo:  # no concatenation happened, use direct slices
			x_win = curr_x_arr[lo:hi]
			y_win = curr_y_arr[lo:hi]
			z_win = curr_z_arr[lo:hi]
		signal = np.sqrt(z_win**2 + y_win**2 + x_win**2)
		signal = signal - np.mean(signal)

		w = np.hanning(n)
		u = np.mean(w**2)  # mean square of Hann window

		signal_fft      = np.fft.rfft(signal * w)
		signal_fft_freq = np.fft.rfftfreq(n, d=1/fs)
		signal_fft_freq = (np.round(signal_fft_freq / Df) * Df).astype(int)

		signal_psd     = (np.abs(signal_fft)**2) / (fs * u * n)
		signal_psd[1:] = signal_psd[1:] * 2  # one-sided correction

		mask  = (signal_fft_freq >= 0) & (signal_fft_freq <= lpf_lim)
		f_sel = signal_fft_freq[mask]
		P_sel = signal_psd[mask]

		k = len(f_sel)
		t_arr[row:row+k] = center_dt64
		f_arr[row:row+k] = f_sel
		p_arr[row:row+k] = P_sel
		row += k

		center_ns += step_ns

	if row == 0:
		return pd.DataFrame(columns=['time', 'frequency', 'power'])

	return pd.DataFrame({
		'time':      t_arr[:row].copy(),
		'frequency': f_arr[:row].copy(),
		'power':     p_arr[:row].copy(),
	})


def _compute_and_save_fft(file_args, key, inst, mode, fs, pad, n_expected, Df, lpf_lim, step=0.05):
	"""
	Worker function for parallel FFT computation.
	file_args: (curr_path, next_path, file_time, next_file_time, fft_pkl)
	Computes FFTs, saves to fft_pkl, returns fft_pkl path or None if empty.
	"""
	curr_path, next_path, file_time, next_file_time, fft_pkl = file_args
	if os.path.exists(fft_pkl):
		return fft_pkl
	file_fdf = _compute_file_ffts(
		curr_path, next_path, key, inst, mode, fs,
		pad, n_expected, Df, file_time, next_file_time, lpf_lim, step
	)
	if not file_fdf.empty:
		os.makedirs(os.path.dirname(fft_pkl), exist_ok=True)
		file_fdf.to_pickle(fft_pkl)
		print(f'Saved FFT pickle: {fft_pkl}')
		return fft_pkl
	return None


def gen_fft(data_dir, pickle_root, key, pad, st, ts, lpf_lim=1000, corrupted_log=None, step=0.05, probe: int = 1):
	"""
	Compute a sliding-window PSD spectrogram over [st, ts].

	Parameters
	----------
	data_dir    : str      Root data directory, e.g. './pydata'
	pickle_root : str      Root cache directory, e.g. './.serialized'
	key         : str      Field prefix: 'B' for magnetic (SCM/SCB), 'E' for electric (EDP/DCE)
	pad         : float    Half-window in seconds; full window = 2*pad s, Df = 1/(2*pad) Hz
	st          : datetime Start time (inclusive)
	ts          : datetime End time (inclusive)
	lpf_lim     : float    Upper frequency cutoff (Hz)

	Returns
	-------
	Generator of pd.DataFrames, one per raw data file, with columns:
	    time      (datetime64[ns])  absolute timestamp
	    frequency (int16)           Hz, snapped to nearest Df multiple
	    power     (float64)         nT²/Hz (or mV²/m²/Hz for E-field)

	Caching
	-------
	Raw data: each CDF is converted to a per-file pickle on first use.
	    Path: {pickle_root}/mms/{probe}/{inst}/{mode}/{YYYY}/{MM}/{DD}/{HHMMSS}.pkl

	FFT results: stored per raw data file, keyed by pad value.
	    Path: {pickle_root}/mms/{probe}/{inst}/{mode}/fft/pad{pad:.3f}/{YYYY}/{MM}/{DD}/{HHMMSS}.pkl
	    On-disk time column: datetime64[ns] (absolute, 8 bytes).
	    Filtered to [st, ts] on load; downstream accumulation code unchanged.
	"""
	if key[0] == 'B':
		inst, mode, fs = 'scm', 'scb', SCB_FS
	else:
		inst, mode, fs = 'edp', 'dce', SCB_FS  # TODO: confirm EDP sample rate

	n_expected = int(fs * (pad * 2))
	Df         = 1 / (pad * 2)

	cdf_files = find_cdf_files(data_dir, inst, mode,
	                           st - timedelta(seconds=pad),
	                           ts + timedelta(seconds=pad),
	                           probe=probe)
	if not cdf_files:
		raise FileNotFoundError(
			f'No CDF files found under {data_dir}/mms{probe}/{inst}/brst/l2/{mode} '
			f'for the range {st} to {ts}.'
		)

	with Manager() as manager:
		lock = manager.Lock()
		fn = partial(ensure_data_pickle, pickle_root=pickle_root, key=key, corrupted_log=corrupted_log, lock=lock, probe=probe)
		with ProcessPoolExecutor(max_workers=8) as pool:
			list(pool.map(fn, cdf_files))

	data_files = find_data_pickles(pickle_root, inst, mode,
	                               st - timedelta(seconds=pad),
	                               ts + timedelta(seconds=pad),
	                               probe=probe)
	if not data_files:
		return

	file_args = [
		(file_path, data_files[i + 1][1] if i + 1 < len(data_files) else None,
		 file_time, data_files[i + 1][0] if i + 1 < len(data_files) else None,
		 _fft_pickle_path(pickle_root, inst, mode, pad, file_time, probe=probe))
		for i, (file_time, file_path) in enumerate(data_files)
	]

	worker = partial(_compute_and_save_fft, key=key, inst=inst, mode=mode, fs=fs,
	                 pad=pad, n_expected=n_expected, Df=Df, lpf_lim=lpf_lim, step=step)

	with ProcessPoolExecutor(max_workers=8) as pool:
		for fft_pkl in pool.map(worker, file_args):
			if not fft_pkl:
				continue

			file_fdf = pd.read_pickle(fft_pkl)
			file_fdf = file_fdf[
				(file_fdf['time'] >= np.datetime64(st, 'ns')) &
				(file_fdf['time'] <= np.datetime64(ts, 'ns'))
			]

			if file_fdf.empty:
				continue

			yield file_fdf


paper = 'mms_data/mms1/scm/brst/l2/schb/2019/08/16/mms1_scm_brst_l2_schb_20190816093103_v2.2.0.cdf'
pkl_path_p = '.serialized/mms/1/scm/schb/20190816093145.pkl'

paperb = 'pydata/mms1/scm/brst/l2/scb/2019/08/16/mms1_scm_brst_l2_scb_20190816093103_v2.2.1.cdf'
pkl_path_b = '../.serialized/mms/1/scm/scb/20190816093145.pkl'

pkl_path_pr = '.serialized/mms/1/scm/fft/scb/20190816093145.pkl'
if __name__ == "__main__":
#	rcParams['path.simplify'] = True
#	rcParams['path.simplify_threshold'] = 0.2
	#rcParams['savefig.format'] = 'svg'
	mpl.use('Qt5Agg')

	lpf_lim = 1000
	hpf_lim = 10
	num_bins = 40
	bin_factor = (lpf_lim) ** (1 / num_bins)

	sig_key = 'Bz'
	inst = 'scm' if sig_key[0] == 'B' else 'edp'
	units = r'\frac{\text{nT}^2}{\text{Hz}}' if sig_key[0] == 'B' else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'

	if sig_key[0] == 'B':
		cdf_path = f'pydata/mms1/{inst}/brst/l2/scb/2019/08/16/mms1_{inst}_brst_l2_scb_20190816093103_v2.2.1.cdf'
		vmi = 1e-6
		vma = 1e-2
	else:
		cdf_path = f'pydata/mms1/{inst}/brst/l2/dce/2019/08/16/mms1_{inst}_brst_l2_dce_20190816093103_v3.0.1.cdf'
		vmi = 1e-4
		vma = 10

	data_dir    = 'E:/PlasmaCoop/pydata'
	pickle_root = 'E:/PlasmaCoop/.serialized'
	save = True
	plot = False

	# all scm caches for graphing all files
	# sfc = glob.glob('mms_data/mms1/scm/brst/l2/schb/2020/09/02/*')
	# pkl_paths = glob.glob('.serialized/scm/schb/*')

	datefmt = '%m/%d/%Y-%H:%M:%S'
	start = datetime.strptime('08/01/2019-00:00:00', datefmt)
	#start = datetime.strptime('08/16/2019-09:31:58.45', datefmt)
	end = datetime.strptime('02/01/2020-00:00:00', datefmt)

	if plot:
		# Create two axes: top for Ez vs time, bottom for FFT/PSD
		fig, axes = plt.subplots(
			1, 1, figsize=(8, 4), sharex=True, constrained_layout=True
		)
		axes = list([axes,])

	bins = list()

	def make_log_bins(freqs, period):
		h = len(freqs) - 1
		df = 1 / period

		b = list()
		while freqs[h] >= df and h > 0:
			b.append(int(freqs[h]))
			h = int(h // bin_factor)

		b.append(int(freqs[0]))

		b.reverse()

		return b

	def channelize(x):
		global bins

		if x == bins[0]:
			return 0
		else:
			for i in range(1, len(bins)):
				if bins[i - 1] < x <= bins[i]:
					return i

		if x == bins[-1]:
			return len(bins)

		print(f'wtf: {x}')
		return None


#		 def display_logbin_legend(ax: plt.Axes):
#	        lg_lbls = [f'Channel {i}: [{bins[i - 1]} - {bins[i]}) Hz' for i in range(channelize(10), len(bins) - 1)]
#	        lg_lbls.append(f'Channel {len(bins) - 1}: [{bins[len(bins) - 2]} - {lpf_lim}] Hz')
#	        proxies = [Patch(color='none') for _ in range(len(lg_lbls))]
#	        ax.legend(proxies, lg_lbls, handlelength=0, handletextpad=0)


	vfunc = np.vectorize(channelize)

	#clear_fft(date=start) # deletes all pkl files for start date, must be adjusted for other

	name = './out/sheet.xlsx'
	if os.path.exists(name):
		os.remove(name)

# 	writer = pd.ExcelWriter(name, engine='xlsxwriter', mode='w')

# # 	book = writer.book

# 	sci_notate = book.add_format()
#	sci_notate.set_num_format(11)

#	print('open excel')

	f = lambda x: int(x)
	vf = np.vectorize(f)
	logbf = lambda x: log(x, bin_factor)
	vlogbf = np.vectorize(logbf)

	if plot:
		display_dt = 0.10  # seconds per display bin — reduce for more detail, increase for speed

		fig.suptitle(
			fr'$|{sig_key[0]}|$ Frequency vs. Time vs. $|{sig_key[0]}|$ Power ${units}$ starting from {start.hour:02d}:{start.minute:02d}:{start.second:02d}')

		methods = False # ~~~~~~~~~~~~~~~~~~~~~~~~~~ TRUE IF PLOTTING SPECTRA OF BOTH, FALSE IF PLOTTING POWER OF BOTH TODO:

	for j, i in enumerate([0.1]): #, 0.25, 0.5]):
		print(i*2)

		if not plot:
			a = datetime.now()
			print(f"Started generating FFTs at @ {a.strftime('%H:%M:%S.%f')[:-3]}")

			for _ in gen_fft(data_dir, pickle_root, sig_key, i, start, end, lpf_lim):
				pass
			b = datetime.now()
			print(f"Ended generating FFTs at @ {b.strftime('%H:%M:%S.%f')[:-3]}")
			print(f"Duration {str(b - a)}")
		else:
			ax = axes[j]

			if start == end:
				def make_fft(start, end, method, i, key):
					# method parameter retained for signature compatibility; gen_fft is always used
					fdf = pd.concat(list(gen_fft(data_dir, pickle_root, key, i, start, end, lpf_lim)), ignore_index=True)

					fdf.loc[fdf['frequency'] == 0, 'frequency'] = 1

					print('load')

			#		fq = fdf['frequency'].unique() # TODO: delete when done debugging
					global bins
					bins = make_log_bins(fdf['frequency'].unique(), i * 2)
					print('bins')
					fdf['channel'] = vfunc(fdf['frequency'])
					print('channelize')

					# Convert absolute datetime64[ns] → float seconds since start
					st_ns = np.datetime64(start, 'ns').astype(np.int64)
					fdf['time'] = (fdf['time'].astype(np.int64) - st_ns) / 1e9 + 1e-12
					fdf['frequency'] = fdf['frequency'].astype(float)
					fdf['power'] = fdf['power'].astype(float)

					return fdf

				fdf = make_fft(start, end, gen_fft, i, sig_key)

				if methods:
					ax.plot(fdf['frequency'], fdf['power'], color='red', label='Numpy method')

					ax.xaxis.set_major_locator(FixedLocator([10, 100, 300, 1000]))
					ax.set_xticklabels([r'$10^1$', r'$10^2$', r'$3 \cdot 10^2$', r'$10^3$'])
					ax.set_xlim(left=1, right=300)
					ax.set_yscale('log')
				else:
					ax.plot(fdf['frequency'], fdf['power'], color='red', label='Numpy method')
					ax.tick_params(labelbottom=True)
					ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, -3))
					ax.set_ylim(top=1.2 * 1e-3 + 1e-4)

				ax.grid(linewidth=0.25)
			else:
				st_ns      = np.datetime64(start, 'ns').astype(np.int64)
				norm       = mpl.colors.LogNorm(vmin=vmi, vmax=vma)
				ax.set_facecolor(plt.cm.jet(norm(vmi)))
				cbar_added = False
				n_chan     = None

				for day_fdf in gen_fft(data_dir, pickle_root, sig_key, i, start, end, lpf_lim):
					day_fdf.loc[day_fdf['frequency'] == 0, 'frequency'] = 1

					if n_chan is None:
						bins   = make_log_bins(day_fdf['frequency'].unique(), i * 2)
						n_chan = len(bins) + 1
						ye     = np.arange(n_chan + 1)
						print('load')

					channel_map = {f: channelize(f) for f in day_fdf['frequency'].unique()}
					day_fdf['channel'] = day_fdf['frequency'].map(channel_map)
					t_vals = (day_fdf['time'].values.astype(np.int64) - st_ns) / 1e9 + 1e-12

					t_min = t_vals.min()
					t_max = t_vals.max()
					chunk_time_bins = np.arange(t_min, t_max + display_dt, display_dt)
					if len(chunk_time_bins) < 2:
						continue

					n_time_chunk = len(chunk_time_bins) - 1
					ix    = np.digitize(t_vals, chunk_time_bins) - 1
					iy    = day_fdf['channel'].values.astype(int)
					valid = (ix >= 0) & (ix < n_time_chunk) & (iy >= 0) & (iy < n_chan)
					flat  = ix[valid] * n_chan + iy[valid]

					sum_chunk   = np.bincount(flat, weights=day_fdf['power'].values[valid], minlength=n_time_chunk * n_chan).reshape(n_time_chunk, n_chan)
					count_chunk = np.bincount(flat, minlength=n_time_chunk * n_chan).reshape(n_time_chunk, n_chan)

					mean_chunk = np.full((n_time_chunk, n_chan), vmi)
					mask = count_chunk > 0
					mean_chunk[mask] = sum_chunk[mask] / count_chunk[mask]

					mappable = ax.pcolormesh(chunk_time_bins, ye, mean_chunk.T, norm=norm, cmap='jet')

					if not cbar_added:
						cbar = plt.colorbar(mappable=mappable, ax=ax)
						cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')
						cbar_added = True

					print('plot chunk')

				if n_chan is not None:
					ax.set_xlabel('Time (s)')
					ax.set_ylabel('Frequency (Hz)')

					ax.set_ylim(bottom=channelize(10), top=n_chan - 1)
					ax.yaxis.set_major_locator(FixedLocator(vfunc([10, 100, 1000])))
					ax.set_yticklabels(['10','100','1000'])

			ax.set_title(rf'FFT taken with {i * 2} sec window / $\Delta f = {int((i * 2) ** -1)}$')
			print('stylize\n')

	if plot:
		if start == end:
			if methods:
				plt.legend(loc='upper right')
			else:
				plt.xlabel('Scipy method Power')
				plt.ylabel('Numpy method Power')

		pngpth = './out/fft/'
		os.makedirs(os.path.dirname(pngpth), exist_ok=True)
		if save:
			if sig_key[0] == 'B':
				plt.savefig(pngpth + 'bfield.pdf', format='pdf')
			else:
				plt.savefig(pngpth + 'efield.pdf', format='pdf')
		else:
			plt.show()
			print('\nshow')


# \twriter.close()
