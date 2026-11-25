import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MultipleLocator
from scipy.stats import binned_statistic_2d

import request
from fftcalculator import FFTCalculator as calc


class Drawer:
	def __init__(self, n_requests):
		self._fig, self._axes = plt.subplots(n_requests, 1, figsize=(8, 4 * n_requests))
		self._fmt = None
		self.freq_bins = []
		self._time_bins = []


	def draw(self, req: request.Request, values: pd.DataFrame, axes_index: int):
		ax = self._axes[axes_index]

		if req.freq_opts.is_log_scaled:
			calc.make_channels(values['frequency'].unique())
			self.freq_bins = range(1, len(req.freq_opts.channels) + 1)
		else:
			self.freq_bins = values['frequency'].unique()

		stat, xe, ye, bn = binned_statistic_2d(values['time'], values['channel'], values['power'],
		                                       statistic=req.power_opts.statistic,
		                                       bins=[self._time_bins, self.freq_bins])

		mappable = ax.pcolormesh(xe, ye, stat.T,
		                         norm=mpl.colors.LogNorm(vmin=req.power_opts.cmin, vmax=req.power_opts.cmax),
		                         cmap=req.power_opts.cmap)

		cbar = plt.colorbar(mappable=mappable, ax=ax)
		cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')

		self.set_xfmt(ax)
		self.set_yfmt(ax, req)


	def set_xfmt(self, ax: plt.Axes):
		ax.xaxis.set_major_locator(MultipleLocator(0.5, 1))
		ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		ax.set_xlabel('Time (s)')


	def set_yfmt(self, ax: plt.Axes, req: request.Request):
		channels = self.freq_bins if req.freq_opts.is_log_scaled else None
		req.freq_opts.tick_fmt.value(ax, channels)
		ax.set_ylabel('Frequency (Hz)')


	def show(self):
		plt.show()

