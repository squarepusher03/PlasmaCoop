import glob
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
	
sfn = 'mms_data/mms1/edp/brst/l2/dce/2020/09/02/mms1_edp_brst_l2_dce_20200902062933_v3.0.1.cdf' # edp source
pkl_path_e = '.cache/edp/dce/20200902062933.pkl'

sfc = 'mms_data/mms1/scm/brst/l2/schb/2020/09/02/mms1_scm_brst_l2_schb_20200902062933_v3.0.1.cdf' # scm source
pkl_path = '.cache/scm/schb/20200902062933.pkl'
if __name__ == "__main__":
	rcParams['path.simplify'] = True
	rcParams['path.simplify_threshold'] = 0.2
	mpl.use('Qt5Agg')
	
	# all scm caches for graphing all files
	#sfc = glob.glob('mms_data/mms1/scm/brst/l2/schb/2020/09/02/*')
	#pkl_paths = glob.glob('.cache/scm/schb/*')

	df = load_pickle_safe(pkl_path_e, sfn)
	df = df.dropna().reset_index(drop=True)

	start = datetime(2020, 9, 2, 6, 30, 54)
	end = datetime(2020, 9, 2, 6, 31, 6)
	
	# Create two axes: top for Ez vs time, bottom for FFT/PSD
	fig, ax = plt.subplots(
		1, 1, figsize=(8, 6), sharex=False, constrained_layout=True
	)
	
	sig_key = 'Ez'
	
	st = datetime(2020, 9, 2, 6, 30, 56)
	ts = st + timedelta(seconds=1.25)
	times = [np.arange((st - start).total_seconds(), (ts - start).total_seconds(), 0.05)]
	
	fdf = pd.DataFrame(columns=['time', 'frequency', 'power'])

	for i in np.arange(0, 0.75, 0.05):
		center = st + timedelta(seconds=(float(i)))
		fft_start = center - timedelta(seconds=(float(0.5)))
		fft_end = center + timedelta(seconds=(float(0.5)))
		
		tdf = df[(df['time'] >= fft_start) & (df['time'] <= fft_end)].reset_index(drop=True)
		signal = tdf[sig_key]
		signal = signal - signal.mean()
		
		n = len(signal)
		w = hann(n, sym=False)
		signal_win = signal * w
		
		# FFT and frequency axis (one-sided)
		signal_fft = fft.rfft(signal_win)
		signal_fft_freq = np.fft.rfftfreq(n, d=1 / SCHB_FS)  # Hz
		
		# Hann window power normalization for PSD (units: nT^2/Hz)
		U = np.sum(w ** 2)  # window power of Hann
		signal_psd = (np.abs(signal_fft) ** 2) / (SCHB_FS * U)
		
		# One-sided correction (conserve variance)
		if n % 2 == 0:
			# even n: DC at 0, Nyquist present at last index
			if signal_psd.size > 2:
				signal_psd[1:-1] *= 2
		else:
			# odd n: no Nyquist bin
			if signal_psd.size > 1:
				signal_psd[1:] *= 2
		
		# Frequency mask — keep only (0, 2000] Hz and apply to BOTH arrays
		mask = (signal_fft_freq > 0) & (signal_fft_freq <= 2000)
		f_sel = signal_fft_freq[mask]
		P_sel = signal_psd[mask]
		fdf = pd.concat([fdf, pd.DataFrame({
			'time': [(center - start).total_seconds()] * len(f_sel),
			'frequency': f_sel,
			'power': P_sel})], ignore_index=True)

	# Bottom: PSD (FFT) plot
	time_bins = np.arange(fdf['time'].min(), fdf['time'].max() + 0.06, 0.05)
	freq_bins = np.linspace(0, 2000, 40)
	ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05, 2))
	
	ax.hist2d(fdf['time'], fdf['frequency'], weights=fdf['power'], bins=[time_bins, freq_bins],
	          cmap='viridis', norm=mpl.colors.LogNorm())
	ax.set_ylim(0, 2000)
	
	ax.set_ylabel(r'Frequency (Hz)')
	ax.set_title(r'$E_z$ Frequency vs. Time vs. $E_z$ Power $\frac{\text{mV}^2}{\text{mV}^2 \cdot \text{Hz}}$')
	ax.set_xlabel('Time (s)')
	
	plt.tight_layout()
	plt.show()
	
#	sn = f'@ {((fft_start - start) + timedelta(seconds=0.5)).total_seconds()}s'
#	with pd.ExcelWriter('power_bfield.xlsx', mode='w') as writer:
#		pdf.to_excel(writer, sheet_name=sn, index=False)

#	with pd.ExcelWriter('power_bfield.xlsx', engine='openpyxl', mode='a',
#						if_sheet_exists='replace') as writer:
#		pdf.to_excel(writer, sheet_name=sn, index=False)


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
