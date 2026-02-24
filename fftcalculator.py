from datetime import datetime

import pandas as pd
import re
from pathlib import Path

import request


class FFTCalculator():
	def __init__(self, req: request.Request):
		self.values = pd.DataFrame()
		self.data = pd.DataFrame()
		self.files = list(str())
		self.data_root = './pydata/'
		self.path = ''


	def _find_valid_data(self, req: request.Request):
		st = req.time_opts.start
		end = req.time_opts.end
		interval = end - st

		if self.path != '':
			self.path = self.data_root

			sr = req.time_opts.sample_rate_type
			pr = req.source_opts.project.lower()
			sat = req.source_opts.satellite.lower()
			inst = req.source_opts.instrument.lower()

			self.path += pr + sat + f'/{inst}/brst/l2/{sr}/{st.year}/'

			if interval.weeks <= 4:
				self.path += f'{st.strftime("%m")}/'
			if interval.hours <= 24:
				self.path += f'{st.strftime("%d")}/'

		for file in Path('.').rglob(self.path + '**/*.cdf'):
			match = re.search(r'\D+(\d{14})\D+', str(file))
			if match:
				ftime = datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
				if st <= ftime <= end:
					self.files.append(str(file))

