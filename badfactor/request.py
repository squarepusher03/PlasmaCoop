import options as opts

class Request:
	def __init__(self):
		self.freq_opts = opts.FrequencyOptions()
		self.power_opts = opts.PowerOptions()
		self.time_opts = opts.TimeOptions()
		self.source_opts = opts.SourceOptions()

		self.power_opts.float_precision = 2


class BFieldRequest(Request):
	def __init__(self):
		super().__init__()
		self.time_opts.unit = r'\frac{\text{nT}^2}{\text{Hz}}'
		self.power_opts.cmin = 1e-6
		self.power_opts.cmax = 1e-2


class EFieldRequest(Request):
	def __init__(self):
		super().__init__()
		self.time_opts.unit = r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'
		self.power_opts.cmin = 1e-4
		self.power_opts.cmax = 10

