from matplotlib.axes import Axes
from datetime import datetime
from enum import Enum
from matplotlib.ticker import FixedLocator

from fftcalculator import FFTCalculator as calc


def base_multiple_format(ax: Axes, base=10):
	channels = FrequencyOptions.channels

	if base is list:
		base = base[0]

	ticks = ax.get_yticks()
	labels = lambda x: str(x)

	tix = []
	for x in ticks:
		if x % base == 0:
			tix.append(x)

	if channels:
		b = calc.channelize(tix[0], channels)
		t = calc.channelize(tix[-1], channels)
		tix = (lambda x: calc.channelize(x, channels))(tix)
	else:
		b = tix[0]
		t = tix[-1]

	ax.set_ylim(bottom=b, top=t)
	ax.set_yticklabels(labels(tix))
	ax.yaxis.set_major_locator(FixedLocator(tix))


class FrequencyOptions:
	channels = list([int()])

	def __init__(self):
		self.is_log_scaled = False
		self.upper_limit = 1e9 # arbitrary large number
		self.lower_limit = 0
		self.tick_fmt = None


	def format(self, ax: Axes, *args):
		self.tick_fmt.value(ax, self.channels, args)



	class FrequencyTickFormat(Enum):
		DEFAULT = ''
		DF_MULTIPLE = ''
		DF_RANGE = ''
		BASE_MULTIPLE = staticmethod(base_multiple_format)

	class FrequencyScale(Enum):
		LINEAR = 'linear'
		LOG = 'log'
		LOG10 = 'log10'

		def make_channels(self, freqs, period):
			bin_factor = opts.upper_lim ** (1 / opts.n_bins)

			h = len(freqs) - 1
			df = 1 / period

			opts.channels = list()
			while freqs[h] >= df and h > 0:
				opts.channels.append(int(freqs[h]))
				h = int(h // bin_factor)

			opts.channels.append(int(freqs[0]))

			opts.channels.reverse()
			self.channels = opts.channels



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

