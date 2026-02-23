import pandas as pd
from pathlib import Path

import request


class FFTCalculator():
	channels = list([int()])

	def __init__(self):
		self.values = pd.DataFrame()
		self.data = pd.DataFrame()
		self.files = list([tuple()])


	def make_channels(self, freqs: list, req: request.Request):
		opts = req.freq_opts
		bin_factor = opts.upper_lim ** (1 / opts.n_bins)

		h = len(freqs) - 1
		df = 1 / req.time_opts.period

		opts.channels = list()
		while freqs[h] >= df and h > 0:
			opts.channels.append(int(freqs[h]))
			h = int(h // bin_factor)

		opts.channels.append(int(freqs[0]))

		opts.channels.reverse()
		self.channels = opts.channels


	def _find_valid_data(self, req: request.Request):



	@staticmethod
	def channelize(x):
		channels = FFTCalculator.channels

		if x == channels[0]:
			return 0
		else:
			for i in range(1, len(channels)):
				if channels[i - 1] < x <= channels[i]:
					return i

		if x == channels[-1]:
			return len(channels)

		print(f'wtf: {x}')
		return 0

