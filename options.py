from matplotlib.axes import Axes
from datetime import datetime
from enum import Enum
from matplotlib.ticker import FixedLocator

from fftcalculator import FFTCalculator as calc


class FrequencyOptions:
	def __init__(self):
		self.upper_limit = 1e9  # arbitrary large number
		self.lower_limit = 0
		self.n_bins = 0
		self.is_log_scaled = False
		self.tick_fmt = None
		self.channels = list([int()])


	def format(self, ax: Axes, *args):
		self.tick_fmt.value(ax, self.channels, args)


	def base_multiple_format(self, ax: Axes, base=10):
		if base is list:
			base = base[0]

		ticks = ax.get_yticks()
		labels = lambda x: str(x)

		tix = []
		for x in ticks:
			if x % base == 0:
				tix.append(x)

		if len(self.channels) > 1:
			b = self.channelize(tix[0])
			t = self.channelize(tix[-1])
			tix = (lambda x: self.channelize(x))(tix)
		else:
			b = tix[0]
			t = tix[-1]

		ax.set_ylim(bottom=b, top=t)
		ax.set_yticklabels(labels(tix))
		ax.yaxis.set_major_locator(FixedLocator(tix))

	def make_channels(self, freqs: list[int], period: float):
		bin_factor = self.upper_limit ** (1 / self.n_bins)

		h = len(freqs) - 1
		df = 1 / period

		while freqs[h] >= df and h > 0:
			self.channels.append(int(freqs[h]))
			h = int(h // bin_factor)

		self.channels.append(int(freqs[0]))

		self.channels.reverse()


	def channelize(self, x):
		if x == self.channels[0]:
			return 0
		else:
			for i in range(1, len(self.channels)):
				if self.channels[i - 1] < x <= self.channels[i]:
					return i

		if x == self.channels[-1]:
			return len(self.channels)

		print(f'wtf: {x}')
		return 0


class FrequencyTickFormat(Enum):
	DEFAULT = ''
	DF_MULTIPLE = ''
	DF_RANGE = ''
	BASE_MULTIPLE = FrequencyOptions.base_multiple_format


class FrequencyScale(Enum):
	LINEAR = 'linear'
	LOG = 'log'
	LOG10 = 'log10'


class SourceOptions:
	def __init__(self):
		self.satellite = 1
		self.sample_rate = 0
		self.sample_rate_type = ''
		self.project = ''
		self.instrument = ''

		if self.project == 'mms':
			if self.sample_rate_type == 'scb':
				self.sample_rate = 0
			elif self.sample_rate_type == 'schb':
				self.sample_rate = 1


class TimeOptions:
	def __init__(self):
		self.fft_period = 0.2
		self.fft_incr = 0.5
		self.major_denom = ''
		self.minor_denom = ''
		self.start = None
		self.end = None


	# only accepts ISO standard dates in 'YYYY-MM-DD HH:MM:SS.ssssss'
	def set_start(self, date: str):
		self.start = datetime.fromisoformat(date)


	# only accepts ISO standard dates in 'YYYY-MM-DD HH:MM:SS.ssssss'
	def set_end(self, date: str):
		self.end = datetime.fromisoformat(date)


class PowerOptions:
	def __init__(self):
		self.cmin = 0
		self.cmax = 1
		self.cmap = 'jet'
		self.statistic = 'mean'
		self.float_precision = 0
		self.field = ''
		self.unit = ''

