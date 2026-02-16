from datetime import datetime, timedelta

import os

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

from caching import load_pickle_safe, regen_cdf

SCB_FS = 8_192
SCHB_FS = 16_384


def scipy_gen_fft(pickle_path, cdf_path, key, *args):
	st = args[0][1]
	ts = args[0][2]
	inst = args[0][3].lower()
	pad = args[0][0]
	
	pickle_path = f'./.cache/mms/1/scm/scb/{st.year}{st.month:02d}{st.day:02d}{st.hour:02d}{st.minute:02d}{st.second:02d}.pkl'
	
	df = load_pickle_safe(pickle_path, cdf_path, key, regen_cdf)
	
	vr = np.vectorize(lambda x: round(x))

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
		fft_start = (center - timedelta(seconds=(float(pad))))
		fft_end = (center + timedelta(seconds=(float(pad))))
		
		tdf = df[(df['time'] >= fft_start) & (df['time'] <= fft_end)].reset_index(drop=True)
		# signal = tdf[sig_key]
		signal = np.sqrt(tdf[key[0] + 'z'] ** 2 + tdf[key[0] + 'y'] ** 2 + tdf[key[0] + 'x'] ** 2)
		signal = signal - np.mean(signal)
		
		n = len(signal)
		w = windows.hann(n)
		u = np.mean(w ** 2)  # "mean square" of hann window, differs by window
		
		# FFT and frequency axis (one-sided)
		signal_fft = scipy.fft.rfft(signal * w)
		signal_fft_freq = scipy.fft.rfftfreq(n, d=1 / SCB_FS)  # Hz
		signal_fft_freq = vr(signal_fft_freq)
		# signal_fft_freq = vr(signal_fft_freq / Df) * Df # TODO: fix leakage with subtracted series
		
		# Hann window power normalization for PSD (units: nT^2/Hz) ONLY FOR POWER SPECTRA
		signal_psd = (np.abs(signal_fft) ** 2) / (fs * u * n)
		signal_psd[1:] = signal_psd[1:] * 2
		
		mask = (signal_fft_freq >= 0) & (signal_fft_freq <= lpf_lim)
		
		f_sel = signal_fft_freq[mask]
		P_sel = signal_psd[mask]
		fdf = pd.concat([fdf, pd.DataFrame({
			'time': [(center - st).total_seconds()] * len(f_sel),  # converting to seconds since start
			'frequency': f_sel,
			'power': P_sel})], ignore_index=True)
	
	pickle_path = f'./.cache/mms/1/scm/scb/fft/s{pad:.2f}p{st.year}{st.month:02d}{st.day:02d}{st.hour:02d}{st.minute:02d}{st.second:02d}.pkl'
	os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
	fdf.to_pickle(pickle_path)
	
	print(f"Successfully generated pickle file: {pickle_path}")
	
	return fdf


def gen_fft(pickle_path, cdf_path, key, *args):
	st = args[0][1]
	ts = args[0][2]
	inst = args[0][3].lower()
	pad = args[0][0]

	if key[0] == 'B':
		inst = 'scm'
		mode = 'scb'
	else:
		inst = 'edp'
		mode = 'dce'

	pickle_path = f'./.cache/mms/1/{inst}/{mode}/{st.year}{st.month:02d}{st.day:02d}{st.hour:02d}{st.minute:02d}{st.second:02d}.pkl'

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
	
	f = df['time']

	for i in np.arange(0, (ts - st).total_seconds() + 0.01, 0.05):
		center = st + timedelta(seconds=(float(i)))
		fft_start = (center - timedelta(seconds=(float(pad))))
		fft_end = (center + timedelta(seconds=(float(pad))))

		tdf = df[(df['time'] >= fft_start) & (df['time'] <= fft_end)].reset_index(drop=True)
		# signal = tdf[sig_key]
		signal = np.sqrt(tdf[key[0] + 'z'] ** 2 + tdf[key[0] + 'y'] ** 2 + tdf[key[0] + 'x'] ** 2)
		signal = signal - np.mean(signal)

		n = len(signal)
		w = np.hanning(n)
		u = np.mean(w ** 2) # "mean square" of hann window, differs by window

		# FFT and frequency axis (one-sided)
		signal_fft = np.fft.rfft(signal * w)
		signal_fft_freq = np.fft.rfftfreq(n, d=1/SCB_FS)  # Hz
		signal_fft_freq = vr(signal_fft_freq)
		#signal_fft_freq = vr(signal_fft_freq / Df) * Df # TODO: fix leakage with subtracted series

		# Hann window power normalization for PSD (units: nT^2/Hz) ONLY FOR POWER SPECTRA
		signal_psd = (np.abs(signal_fft) ** 2) / (fs * u * n)
		signal_psd[1:] = signal_psd[1:] * 2

		mask = (signal_fft_freq >= 0) & (signal_fft_freq <= lpf_lim)

		f_sel = signal_fft_freq[mask]
		P_sel = signal_psd[mask]
		fdf = pd.concat([fdf, pd.DataFrame({
			'time': [(center - st).total_seconds()] * len(f_sel), # converting to seconds since start
			'frequency': f_sel,
			'power': P_sel})], ignore_index=True)
	
	pickle_path = f'./.cache/mms/1/{inst}/scb/fft/{pad:.2f}p{st.year}{st.month:02d}{st.day:02d}{st.hour:02d}{st.minute:02d}{st.second:02d}.pkl'
	os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
	fdf.to_pickle(pickle_path)

	print(f"Successfully generated pickle file: {pickle_path}")

	return fdf


paper = 'mms_data/mms1/scm/brst/l2/schb/2019/08/16/mms1_scm_brst_l2_schb_20190816093103_v2.2.0.cdf'
pkl_path_p = '.cache/mms/1/scm/schb/20190816093145.pkl'

paperb = 'pydata/mms1/scm/brst/l2/scb/2019/08/16/mms1_scm_brst_l2_scb_20190816093103_v2.2.1.cdf'
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

	save = True

	# all scm caches for graphing all files
	# sfc = glob.glob('mms_data/mms1/scm/brst/l2/schb/2020/09/02/*')
	# pkl_paths = glob.glob('.cache/scm/schb/*')

	datefmt = '%m/%d/%Y-%H:%M:%S'
	start = datetime.strptime('08/16/2019-09:31:45', datefmt)
	#start = datetime.strptime('08/16/2019-09:31:58.45', datefmt)
	end = datetime.strptime('08/16/2019-09:32:15', datefmt)

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

	writer = pd.ExcelWriter(name, engine='xlsxwriter', mode='w')

	book = writer.book

	sci_notate = book.add_format()
	sci_notate.set_num_format(11)

	print('open excel')

	f = lambda x: int(x)
	vf = np.vectorize(f)
	logbf = lambda x: log(x, bin_factor)
	vlogbf = np.vectorize(logbf)

	time_bins = np.arange(0, (end - start).total_seconds() + 0.06, 0.05)

	fig.suptitle(
		fr'$|{sig_key[0]}|$ Frequency vs. Time vs. $|{sig_key[0]}|$ Power ${units}$ starting from {start.hour:02d}:{start.minute:02d}:{start.second:02d}')

	methods = False # ~~~~~~~~~~~~~~~~~~~~~~~~~~ TRUE IF PLOTTING SPECTRA OF BOTH, FALSE IF PLOTTING POWER OF BOTH TODO:

#	if methods:
#		fig.suptitle(fr'{sig_key[0]}-field Frequency vs. Power ${units}$ @ {start.strftime(datefmt)}')
#	else:
#		fig.suptitle(fr'Numpy method vs. Scipy method of Power ${units}$ @ {start.strftime(datefmt)}')

	for j, i in enumerate([0.1, 0.25, 0.5]):
		print(i*2)
		ax = axes[j]
		def make_fft(start, end, method, i, key):
			#pkl_path_real = f'./.cache/mms1/scm/scb/fft/{i:.2f}p{start.year}{start.month:02d}{start.day:02d}{start.hour:02d}{start.minute:02d}{start.second:02d}.pkl'
			mode = 'dce' if inst == 'edp' else 'scb'
			pkl_path_real = f'./.cache/mms/1/{inst}/{mode}/fft/{i:.2f}p{start.year}{start.month:02d}{start.day:02d}{start.hour:02d}{start.minute:02d}{start.second:02d}.pkl'
			fdf = load_pickle_safe(pkl_path_real, cdf_path, sig_key, method, i, start, end, mode)
			fdf.loc[fdf['frequency'] == 0, 'frequency'] = 1

			print('load')

	#		fq = fdf['frequency'].unique() # TODO: delete when done debugging
			global bins
			bins = make_log_bins(fdf['frequency'].unique(), i * 2)
			print('bins')
			fdf['channel'] = vfunc(fdf['frequency'])
			print('channelize')
			
			fdf['time'] = fdf['time'].astype(float) + 1e-12
			fdf['frequency'] = fdf['frequency'].astype(float)
			fdf['power'] = fdf['power'].astype(float)
			
			return fdf

		fdf = make_fft(start, end, gen_fft, i, sig_key)

		fdf.to_excel(writer, sheet_name=f'{i * 2} sec interval')
		print('write excel')

		# Bottom: PSD (FFT) plot

#        print(str(2 * i))
#        print(fdf[:10])
		
		if start == end:
			fdf = make_fft(start, end, gen_fft, i, sig_key)
			sdf = make_fft(start, end, scipy_gen_fft, i, sig_key)
			
			if methods:
				ax.plot(sdf['frequency'], sdf['power'], color='blue', label='Scipy method')
				ax.plot(fdf['frequency'], fdf['power'], color='red', label='Numpy method')

				ax.xaxis.set_major_locator(FixedLocator([10, 100, 300, 1000]))
				ax.set_xticklabels([r'$10^1$', r'$10^2$', r'$3 \cdot 10^2$', r'$10^3$'])
				ax.set_xlim(left=1, right=300)
				ax.set_yscale('log')
			else:
				ax.plot(sdf['power'], fdf['power'])
				ax.tick_params(labelbottom=True)
				ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, -3))
				ax.set_ylim(top=1.2 * 1e-3 + 1e-4)

			ax.grid(linewidth=0.25)
		
			#ax.yaxis.set_major_locator(FixedLocator(fdf['channel'].unique()))
			#tl = bins
			#ax.set_yticklabels(tl)
		else:
			stat, xe, ye, bn = binned_statistic_2d(fdf['time'], fdf['channel'], fdf['power'],
			                                       statistic='mean', bins=[time_bins, fdf['channel'].unique()])
			mappable = ax.pcolormesh(xe, ye, stat.T,
									 norm=mpl.colors.LogNorm(vmin=vmi, vmax=vma), cmap='jet')
			cbar = plt.colorbar(mappable=mappable, ax=ax)
			cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')
			
			print('plot')

			# debug plot stuff
			#ax.yaxis.set_major_locator(MultipleLocator(1))
			#ax.tick_params('x', rotation=90)
			
			ax.xaxis.set_major_locator(MultipleLocator(0.5, 1))
			ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
			ax.set_xlabel('Time (s)')
			ax.set_ylabel('Frequency (Hz)')

			#ax.set_ylim(bottom=channelize(10), top=channelize(lpf_lim))
			ax.set_ylim(bottom=channelize(10), top=fdf['channel'].unique()[-1])
			ax.yaxis.set_major_locator(FixedLocator(vfunc([10, 100, 1000])))
			ax.set_yticklabels(['10','100','1000'])
			#ax.yaxis.set_major_locator(FixedLocator(fdf['channel'].unique()))
			#tl = bins
			#ax.set_yticklabels(tl)

		ax.set_title(rf'FFT taken with {i * 2} sec window / $\Delta f = {int((i * 2) ** -1)}$')
		print('stylize\n')

	if start == end:
		if methods:
			plt.legend(loc='upper right')
		else:
			plt.xlabel('Scipy method Power')
			plt.ylabel('Numpy method Power')

	for sheet in writer.sheets.values():
		sheet.set_column('D:D', 10, sci_notate)

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


	writer.close()
