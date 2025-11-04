import glob

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os

from matplotlib.ticker import FuncFormatter, FixedLocator, MaxNLocator

def _mmssms_formatter(x, pos=None):
	"""Format matplotlib date tick value as HH:MM:SS.
	x is a Matplotlib date (float days), pos is tick position (unused).
	"""
	dt = mdates.num2date(x)
	return f"{dt:%H:%M:%S}"

RE = 6300  # km
THRESHOLD = 1e4
MAX_WORKERS = lambda _ : max(1, (os.cpu_count() or 1) - 2)

SCHB_FS = 16_384
SCHB_N = 65_536

def cache_path(file):
	"""Create a unique pickle path for a given fgm file name."""
	base = os.path.basename(file)
	name, _ = os.path.splitext(base)
	return f'./.cache/scm/schb/{base[-25:-11]}.pkl'

def spectra(df, n, fs):
	ft = np.fft.rfft(df)
	amp = np.abs(ft)
	ftq = np.fft.rfftfreq(n, d=1/fs)
	
	return ftq, amp

def read_data(fileperinst):
	from spacepy import pycdf

	scm_file = fileperinst[0]

	pkl = cache_path(scm_file)

	if os.path.exists(pkl):
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
		
		#start = df.iloc[0, 'time']
		# TODO: successfully got here, check if start + end work or try to find string rep of time for chart
		# TODO: ask for bin count
		#end = df.iloc[:-1, 'time']
		name = f'B-field vectors spectra on {df['time'][0]} to {df['time'].iat[-1]}'
		fig.suptitle(name)
		comps = {'Bx': 'tab:blue', 'By': 'tab:red', 'Bz': 'tab:green'}
		# Track global y-range across components for this figure
		gmin = np.inf
		gmax = -np.inf
		
		ax.set_ylabel('Amplitude of B-field (nT)')
		ax.set_xlabel('Frequency of B-field (Hz)')
		
		ax.set_xlim(mdates.date2num(start), mdates.date2num(end))
		
		ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
		ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
		
		ax.xaxis.set_major_formatter(FuncFormatter(_mmssms_formatter))
		ax.xaxis.set_minor_formatter(FuncFormatter(_mmssms_formatter))
		
		for i, b in enumerate(comps.keys()):
			
			ax.scatter(df['time'], df[b], color=comps.get(b), label=b, s=0.4)
			# Rotate x-axis labels by -45 degrees for readability
			for label in ax.get_xticklabels():
				label.set_rotation(-45)
				label.set_horizontalalignment('left')
				label.set_rotation_mode('anchor')
			# Also rotate minor tick labels
			for label in ax.get_xminorticklabels():
				label.set_rotation(-45)
				label.set_horizontalalignment('left')
				label.set_rotation_mode('anchor')
			
				dmin = np.nanmin(df[b].values)
				dmax = np.nanmax(df[b].values)
				gmin = min(gmin, dmin)
				gmax = max(gmax, dmax)

		#ftq, amp = spectra(df[b], SCHB_N, SCHB_FS)
			#plt.bar(ftq, amp, width=ftq[1]-ftq[0], label="")
			
		# Build common Y-axis major ticks across subplots in this figure
		locator = MaxNLocator(nbins=6)
		ticks = locator.tick_values(gmin, gmax)
		ticks = np.asarray(ticks, dtype=float)
		if ticks.size < 2:
			span = 1.0 if not (np.isfinite(gmin) and np.isfinite(gmax)) else max(1e-6, abs(gmax - gmin))
			ticks = np.array([gmin - span/2, gmax + span/2], dtype=float)
		for ax in fig.axes:
			ax.yaxis.set_major_locator(FixedLocator(ticks))
			ax.set_ylim(ticks[0], ticks[-1])
		fig.subplots_adjust(hspace=0.65)

		fig.legend()
		fig.savefig(f'./charts/scm/{name.replace(' ','_').replace(':', ',')}.png')
