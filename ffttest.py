from datetime import datetime, timedelta

import os
from time import strftime

import numpy as np
from scipy import fft
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams
from scipy.signal.windows import hann
from matplotlib.widgets import Slider

SCHB_FS = 16_384

def timegraph(ax_time):
	# Top: time-domain Ez with x-limits from start to end and ticks every 0.2 s
	ax_time.set_xlim(start, end)
	ax_time.axvspan(xmin=fft_start, xmax=fft_end, color='maroon', alpha=0.5)
	
	def dformat(x, pos):
		"""Format tick labels as seconds since `start`.

		We avoid timezone/naive-datetime issues by working in Matplotlib's
		float date units (days) and converting the delta to seconds.
		"""
		secs = (x - mdates.date2num(start)) * 86400.0
		return f"{secs:.1f}"
	
	ax_time.xaxis.set_major_formatter(FuncFormatter(dformat))
	
	#		# Tick every 0.2 seconds, labels show time in seconds only
	ax_time.set_xticks(pd.date_range(start, end, freq='500ms')[:-1])
	ax_time.set_xticks(pd.date_range(start, end, freq='050ms'), minor=True)
	ax_time.tick_params(axis='x', labelrotation=90)
	ax_time.set_xlabel('Time (s)')
	ax_time.set_ylabel('Ez (nT)')
	ax_time.set_title(f'Ez vs Time starting @ {datetime.strftime(start, "%H:%M:%S.%f")[:-4]} s\n')
	ax_time.grid(True)

def load_pickle_safe(pickle_path, cdf_path):
	"""Load pickle with fallback to regenerate from CDF if incompatible."""
	try:
		if os.path.exists(pickle_path):
			return pd.read_pickle(pickle_path)
	except (ModuleNotFoundError, AttributeError, ImportError) as e:
		print(f"Warning: Pickle file incompatible ({e}). Regenerating from CDF...")
	
	# Regenerate from CDF
	try:
		from spacepy import pycdf
		with pycdf.CDF(cdf_path) as cdf:
			var = cdf['mms1_scm_acb_gse_schb_brst_l2'][:]
			time = pd.to_datetime(cdf['Epoch'])
		
		df = pd.DataFrame({'time': time, 'Bx': var[:, 0], 'By': var[:, 1], 'Ez': var[:, 2]})
		
		# Save the regenerated pickle
		os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
		df.to_pickle(pickle_path)
		print(f"Successfully regenerated pickle file: {pickle_path}")
		
		return df
	except ImportError:
		print("Error: spacepy not available. Cannot regenerate from CDF.")
		print("Please install spacepy or delete the .cache directory and regenerate pickle files.")
		raise
	
sfn = 'mms_data/mms1/edp/brst/l2/dce/2020/09/02/mms1_edp_brst_l2_dce_20200902062933_v3.0.1.cdf'
if __name__ == "__main__":
	rcParams['path.simplify'] = True
	rcParams['path.simplify_threshold'] = 0.2
	mpl.use('Qt5Agg')
	
	pkl_path = '.cache/edp/dce/20200902062933.pkl'
	df = load_pickle_safe(pkl_path, sfn)
	df = df.dropna().reset_index(drop=True)

	start = datetime(2020, 9, 2, 6, 30, 54)
	end = datetime(2020, 9, 2, 6, 31, 6)
	
	# Create two axes: top for Ez vs time, bottom for FFT/PSD
	fig, axes = plt.subplots(
		6, 1, figsize=(10, 10), sharex=False, constrained_layout=True
	)
	
	for i, ax in enumerate(axes):
		fft_start = datetime(2020, 9, 2, 6, 30, 56) + i * timedelta(seconds=0.05)
		fft_end = fft_start + pd.Timedelta(seconds=1)
		print(f'start: {(fft_start - start).total_seconds()}, end: {(fft_end - start).total_seconds()}')
		#print(f'centered @ {(fft_start - start).total_seconds() + 0.5}s')
		# time_s = (td - start).dt.total_seconds()
		df = df[(df['time'] > fft_start) & (df['time'] < fft_end)]
		
		Ez = df['Ez']
		Ez = Ez - Ez.mean()
		time = df['time'] - start
		
		# Length and window
		n = len(Ez)
		# Apply a Hann window (FFT-oriented, periodic form) to reduce spectral leakage
		# Use SciPy's hann(..., sym=False) to match periodogram's default
		w = hann(n, sym=False)
		Ez_win = Ez * w
		
		# FFT and frequency axis (one-sided)
		Ez_fft = fft.rfft(Ez_win)
		Ez_fft_freq = np.fft.rfftfreq(n, d=1 / SCHB_FS)  # Hz
		
		# Hann window power normalization for PSD (units: nT^2/Hz)
		U = np.sum(w ** 2)  # window power of Hann
		Ez_psd = (np.abs(Ez_fft) ** 2) / (SCHB_FS * U)
		
		# One-sided correction (conserve variance)
		if n % 2 == 0:
			# even n: DC at 0, Nyquist present at last index
			if Ez_psd.size > 2:
				Ez_psd[1:-1] *= 2
		else:
			# odd n: no Nyquist bin
			if Ez_psd.size > 1:
				Ez_psd[1:] *= 2
		
		# Frequency mask — keep only (0, 2000] Hz and apply to BOTH arrays
		#mask = (Ez_fft_freq > 0) & (Ez_fft_freq <= 2000)
		#f_sel = Ez_fft_freq[mask]
		#P_sel = Ez_psd[mask]
	
		ticks = [10 * 10 ** (-x) for x in range(10, 0, -3)]
		# Bottom: PSD (FFT) plot
		ax.plot(Ez_fft_freq, Ez_psd)
		ax.set_yscale('log')
		ax.set_yticks(ticks)
		ax.set_xlim(0, 2000)
		ax.grid(True)
		
		labels = [item.get_text() for item in ax.get_yticklabels()]
		labels[-1] = 1
		ax.set_yticklabels(labels)
	
	axes[0].set_ylabel(r'Power $\frac{(\text{mV})^2}{\text{m}^2 \cdot \text{Hz}}$')
	axes[0].set_title(r'$E_z$ Power Spectrum')
	axes[-1].set_xlabel('Frequency (Hz)')
	
	# Layout: add extra bottom margin so there's vertical space for the slider
	
	plt.tight_layout()
	plt.show()

#	sl_start = (td - (start + timedelta(seconds=0.5))).abs().argmin()
#	sl_end = (td - (end - timedelta(seconds=0.5))).abs().argmin()
#
#	mpltime = mdates.date2num(td)
#	major = list(ax_time.xaxis.get_majorticklocs())
#	minor = list(ax_time.xaxis.get_minorticklocs())
#
#	major.extend(minor)
#	major.sort()
#	ticks = major
#
#	tick_idx = [
#		np.argmin(np.abs(mpltime - tick))
#		for tick in ticks
#	]
#	tick_idx = sorted(list(set(tick_idx)))[2:-2]
#
#	help = plt.text(0.2, 0.2, 'start, end')
#
#	# Slider axes: leave vertical space between slider, the edge, and the lower axes
#	slider_ax = fig.add_axes([0.12, 0.08, 0.55, 0.04])
#	index_slider = Slider(
#		ax=slider_ax,
#		label='Timestamp the FFT is centered around',
#		valmin=float(tick_idx[0]),
#		valmax=float(tick_idx[-1]),
#		valinit=float(len(tick_idx) // 2 + 1),
#		valstep=tick_idx
#	)
#
#	# Initial highlight window corresponding to the slider position (store in a list for mutability)
#	dt = td.iloc[int(index_slider.val)]
	
	#mid = len(tick_idx) // 2 + 1
	#ts = td.iloc[tick_idx[mid - 2]]
	#te = td.iloc[tick_idx[mid + 2]]
	
	#highlight = [ax_time.axvspan(ts, te,
#	                             color='maroon', alpha=0.5)]
#
#	def update(val):
#		dt = td.iloc[int(index_slider.val)]
#		idx = tick_idx.index(val)
#		ts = td.iloc[tick_idx[idx - 2]]
#		te = td.iloc[tick_idx[idx + 2]]
#		help.set_text(str(ts) + ', ' + str(te))
#
#		highlight[0].remove()
#		highlight[0] = ax_time.axvspan(ts, te, color='maroon', alpha=0.5)
#		fig.canvas.draw_idle()

	# Connect slider to update callback
#	index_slider.on_changed(update)
