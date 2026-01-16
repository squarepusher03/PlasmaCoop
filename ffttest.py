import glob
from datetime import datetime, timedelta

import os

import numpy as np
from scipy import fft
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import log
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator, MultipleLocator, LogLocator, FixedLocator
from scipy.signal.windows import hann
from scipy.stats import binned_statistic_2d

from caching import load_pickle_safe, regen_cdf

SCB_FS = 8_192
SCHB_FS = 16_384


def gen_fft(pickle_path, cdf_path, key, *args):
	st = args[0][1]
	ts = args[0][2]
	inst = args[0][3].lower()
	pad = args[0][0]

	pickle_path = f'./.cache/mms/1/scm/scb/{st.year}{st.month:02d}{st.day:02d}{st.hour:02d}{st.minute:02d}{st.second:02d}.pkl'

	df = load_pickle_safe(pickle_path, cdf_path, key, regen_cdf)

	vr = np.vectorize(lambda x: round(x))
	Df = 1 / (pad * 2)

	match inst:
		case 'schb':
			fs = SCHB_FS
		case 'scb':
			fs = SCB_FS
		case _:
			fs = SCB_FS

	fdf = pd.DataFrame(columns=['time', 'frequency', 'power'])

	for i in np.arange(0, (ts - st).total_seconds() + 0.01, 0.05):
		center = st + timedelta(seconds=(float(i)))
		fft_start = center - timedelta(seconds=(float(pad)))
		fft_end = center + timedelta(seconds=(float(pad)))

		tdf = df[(df['time'] >= fft_start) & (df['time'] <= fft_end)].reset_index(drop=True)
		# signal = tdf[sig_key]
		signal = np.sqrt(tdf[key[0] + 'z'] ** 2 + tdf[key[0] + 'y'] ** 2 + tdf[key[0] + 'x'] ** 2)
		signal = signal - np.mean(signal)

		n = len(signal)
		w = hann(n, sym=False)
		u = np.mean(w ** 2) # "mean square" of hann window, differs by window

		# FFT and frequency axis (one-sided)
		signal_fft = fft.rfft(signal * w)
		signal_fft_freq = np.fft.rfftfreq(n, d=1/SCB_FS)  # Hz
		#signal_fft_freq = vr(signal_fft_freq / Df) * Df # TODO: fix leakage with subtracted series

		# Hann window power normalization for PSD (units: nT^2/Hz) ONLY FOR POWER SPECTRA
		signal_psd = (np.abs(signal_fft) ** 2) / (fs * u * n)
		signal_psd[1:] = signal_psd[1:] * 2

		mask = (signal_fft_freq >= 0) & (signal_fft_freq <= lpf_lim)

		f_sel = signal_fft_freq[mask]
		P_sel = signal_psd[mask]
		fdf = pd.concat([fdf, pd.DataFrame({
			'time': [(center - st).total_seconds()] * len(f_sel),
			'frequency': f_sel,
			'power': P_sel})], ignore_index=True)

	os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
	fdf.to_pickle(pickle_path)

	print(f"Successfully generated pickle file: {pickle_path}")

	return fdf


paper = 'mms_data/mms1/scm/brst/l2/schb/2019/08/16/mms1_scm_brst_l2_schb_20190816093103_v2.2.0.cdf'
pkl_path_p = '.cache/mms/1/scm/schb/20190816093145.pkl'

paperb = 'mms_data/mms1/scm/brst/l2/scb/2019/08/16/mms1_scm_brst_l2_scb_20190816093103_v2.2.1.cdf'
pkl_path_b = '.cache/mms/1/scm/scb/20190816093145.pkl'

pkl_path_pr = '.cache/mms/1/scm/fft/scb/20190816093145.pkl'
if __name__ == "__main__":
#	rcParams['path.simplify'] = True
#	rcParams['path.simplify_threshold'] = 0.2
	#rcParams['savefig.format'] = 'svg'
	mpl.use('Qt5Agg')

	lpf_lim = 1000
	hpf_lim = 10
	num_bins = 40
	bin_factor = (lpf_lim) ** (1 / num_bins)

	cdf_path = paperb

	# all scm caches for graphing all files
	# sfc = glob.glob('mms_data/mms1/scm/brst/l2/schb/2020/09/02/*')
	# pkl_paths = glob.glob('.cache/scm/schb/*')

	sig_key = 'Bz'
	units = r'\frac{\text{nT}^2}{\text{Hz}}' if sig_key[0] == 'B' else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'

	datefmt = '%m/%d/%Y-%H:%M:%S'
	start = datetime.strptime('08/16/2019-09:31:56', datefmt)
	end = datetime.strptime('08/16/2019-09:32:00', datefmt)

	# Create two axes: top for Ez vs time, bottom for FFT/PSD
	fig, axes = plt.subplots(
		3, 1, figsize=(8, 12), sharex=True, constrained_layout=True
	)

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

		if x == lpf_lim + 1:
			return len(bins)

		print(f'wtf: {x}')
		return None

	vfunc = np.vectorize(channelize)

	#clear_fft(date=start) # deletes all pkl files for start date, must be adjusted for other

	name = './out/sheet.xlsx'
	if os.path.exists(name):
		os.remove(name)

	writer = pd.ExcelWriter(name, mode='w')

	print('open excel')

	f = lambda x: int(x)
	vf = np.vectorize(f)
	logbf = lambda x: log(x, bin_factor)
	vlogbf = np.vectorize(logbf)

	time_bins = np.arange(0, (end - start).total_seconds() + 0.06, 0.05)

	fig.suptitle(
		fr'$|{sig_key[0]}|$ Frequency vs. Time vs. $|{sig_key[0]}|$ Power ${units}$ starting from {start.hour:02d}:{start.minute:02d}:{start.second:02d}')

	for j, i in enumerate([0.1, 0.25, 0.5]):
		print(i*2)
		ax = axes[j]
		pkl_path_real = f'./.cache/mms1/scm/scb/{start.year}{start.month:02d}{start.day:02d}{start.hour:02d}{start.minute:02d}{start.second:02d}.pkl'
		fdf = load_pickle_safe(pkl_path_real, cdf_path, sig_key, gen_fft, i, start, end, 'scb')
		fdf.loc[fdf['frequency'] == 0, 'frequency'] = 1

		print('load')

		fq = fdf['frequency'].unique() # TODO: delete when done debugging
		bins = make_log_bins(fdf['frequency'].unique(), i * 2)
		print('bins')
		fdf['channel'] = vfunc(fdf['frequency'])
		print('channelize')

		fdf.to_excel(writer, sheet_name=f'{i * 2} sec interval')
		print('write excel')

		# Bottom: PSD (FFT) plot
		fdf['time'] += 1e-12

#        print(str(2 * i))
#        print(fdf[:10])

		stat, xe, ye, bn = binned_statistic_2d(fdf['time'], fdf['channel'], fdf['power'],
											   statistic='mean', bins=[time_bins, fdf['channel'].unique()])
		print('plot')

		# debug plot stuff
		ax.yaxis.set_major_locator(MultipleLocator(1))
		#ax.tick_params('x', rotation=90)

		ax.xaxis.set_major_locator(MultipleLocator(0.5, 1))
		ax.yaxis.set_major_locator(FixedLocator(vfunc([10, 100, 1000])))
		ax.set_yticklabels([r'$10^1$', r'$10^2$', r'$10^3$'])

		mappable = ax.pcolormesh(xe, ye, stat.T,
								 norm=mpl.colors.LogNorm(vmin=1e-6, vmax=1e-2), cmap='jet')
		cbar = plt.colorbar(mappable=mappable, ax=ax)
		cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')

		ax.set_ylim(bottom=channelize(10), top=channelize(lpf_lim))
		ba = ax.get_yticklabels()
		ba[-1].set_text(r'$10^3$')
		ba[-2].set_text('')

#        lg_lbls = [f'Channel {i}: [{bins[i - 1]} - {bins[i]}) Hz' for i in range(channelize(10), len(bins) - 1)]
#        lg_lbls.append(f'Channel {len(bins) - 1}: [{bins[len(bins) - 2]} - {lpf_lim}] Hz')
#        proxies = [Patch(color='none') for _ in range(len(lg_lbls))]
#        ax.legend(proxies, lg_lbls, handlelength=0, handletextpad=0)

		#ax.grid()
		ax.set_title(rf'FFT taken with {i * 2} sec window / $\Delta f = {int((i * 2) ** -1)}$')
		ax.set_ylabel('Frequency (Hz)')
		ax.set_xlabel('Time (s)')
		print('stylize\n')

	plt.show()
	print('\nshow')

	pngpth = './out/fft/scb.png'
	os.makedirs(os.path.dirname(pngpth), exist_ok=True)
	#plt.savefig(gen_path(ext='png'))

	writer.close()

#    ytx = ax.get_yticks()
#    ytl = [str(int(i)) for i in ytx[:-2]] + ['']
# ax.set_yticklabels(ytl)

#	fdf = tdf[fdf['time'] == 2.45].drop(columns='time')
#	ax1.plot(fdf['frequency'], fdf['power'], color='red')

#	fdf = load_pickle_safe(pkl_path_real, cache_path, sig_key, gen_fft, 0.25, start, end)
#	fdf = tdf[tdf['time'] == 2.45].drop(columns='time')
#	ax1.plot(fdf['frequency'], fdf['power'], color='blue')
#
#	ax1.set_xticks([10,] + list(range(100, 1001, 100)))
#	ax1.set_xlim(left=10, right=1000)
#	ax1.set_title(fr'$|{sig_key[0]}|$ Power Spectra at 09:31:58.45')
#	ax1.set_xlabel('Frequency (Hz)')
#	ax1.set_ylabel(rf'Power $\left({units}\right)$')
#	ax1.set_yscale('log')
#	ax1.legend(labels=['0.2 second window', '0.5 second window'])

