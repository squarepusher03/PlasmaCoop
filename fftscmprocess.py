import glob
import os

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import MultipleLocator, FuncFormatter

def major_multiple_formatter(val, pos):
	if val < 1000:
		return f'{int(val)}'
	if val % 1000 == 0:
		return f"{int(val) / 1000}k"
	return ""

def minor_multiple_formatter(val, pos):
	if val <= 500:
		return f'{int(val)}'
	if val % 500 == 0:
		return f"{int(val) / 1000}k"
	return ""

RE = 6300  # km
THRESHOLD = 1e4
MAX_WORKERS = lambda _ : max(1, (os.cpu_count() or 1) - 2)

SCHB_FS = 16_384

def cache_path(file):
	"""Create a unique pickle path for a given fgm file name."""
	base = os.path.basename(file)
	name, _ = os.path.splitext(base)
	return f'./.cache/scm/schb/{base[-25:-11]}.pkl'

def spectra(df, fs):
	df = df.dropna().reset_index(drop=True)
	ft = np.fft.rfft(df)
	amp = np.abs(ft)
	ftq = np.fft.rfftfreq(len(df), d=1/fs)
	
	return ftq, amp

def read_data(fileperinst):
	from spacepy import pycdf

	scm_file = fileperinst[0]

	pkl = cache_path(scm_file)

	if False:#os.path.exists(pkl):
		row = pd.read_pickle(pkl)
	else:
		with pycdf.CDF(scm_file) as cdf:
			var = cdf['mms1_scm_acb_gse_schb_brst_l2'][:]
			time = pd.to_datetime(cdf['Epoch'])

		row = pd.DataFrame({'time': time, 'Bx': var[:, 0], 'By': var[:, 1], 'Bz': var[:, 2]})
	
	row.to_pickle(cache_path(scm_file))

	return row

if __name__ == '__main__':
	from spacepy import pycdf
	matplotlib.rcParams['path.simplify'] = True
	matplotlib.rcParams['path.simplify_threshold'] = 0.2

	scm_files = sorted(glob.glob('./mms_data/mms1/scm/brst/l2/schb/2020/09/02/*'))
	
	figs = []
	axes = []
	for file in scm_files:
		fig, ax = plt.subplots(figsize=(12, 8))
		figs.append(fig)
		axes.append(ax)
		df = read_data([file])
		
		start = datetime(2020, 9, 2, 6, 30, 46)
		end = start + pd.Timedelta(seconds=20)
		if df['time'][0] < end:
			df = df[(df['time'] > start) & (df['time'] < end)].reset_index(drop=True)
		else:
			continue
		
		name = f'B-field vectors spectra on {df['time'][0]} to {df['time'].iat[-1]}'
		fig.suptitle(name)
		comps = {'Bx': 'tab:blue', 'By': 'tab:red', 'Bz': 'tab:green'}
		# Track global y-range across components for this figure
		gmin = np.inf
		gmax = -np.inf
		
		ax.set_ylabel('Amplitude of B-field (nT)')
		ax.set_xlabel('Frequency of B-field (Hz)')
		ax.grid(which='major', axis='both', linestyle='-', color='gray', linewidth=1)
		ax.grid(which='minor', axis='both', linestyle=':', color='gray', linewidth=0.6)
		ax.margins(x=0.01, y=0)
		
		ax.xaxis.set_minor_locator(MultipleLocator(100))
		ax.xaxis.set_major_formatter(FuncFormatter(major_multiple_formatter))
		ax.xaxis.set_minor_formatter(FuncFormatter(minor_multiple_formatter))
		
		for i, b in enumerate(comps.keys()):
			freq, amp = spectra(df[b], SCHB_FS)
			ax.plot(freq, amp, color=comps.get(b), label=b)
			
		fig.legend()
		fig.savefig(f'./charts/fft/scm/{name.replace(' ','_').replace(':', ',')}.png')
