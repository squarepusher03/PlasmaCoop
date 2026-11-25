from abc import ABC, abstractmethod
from datetime import datetime
from pytplot import del_data

from ffttest import find_cdf_files



class DownloadRequest(ABC):
	RESET = '\033[0m'

	# Subclasses must set these (scm/edp, B/E).
	inst:      str
	field:     str

	probe:     str
	trange:    list | None
	datatype:  str
	data_rate: str

	@abstractmethod
	def __init__(self, probe: str, trange: list | None, datatype: str, data_rate: str):
		self.probe = probe
		self.trange = trange
		self.datatype = datatype
		self.data_rate = data_rate


	def find_cdfs(self, data_dir: str) -> list[str]:
		"""CDFs under data_dir whose coverage window overlaps this request's trange."""
		if not self.trange:
			return []
		start = datetime.strptime(self.trange[0], '%Y-%m-%d')
		end   = datetime.strptime(self.trange[1], '%Y-%m-%d')
		return find_cdf_files(data_dir, self.inst, self.datatype, start, end, probe=int(self.probe))


	def download(self, color: str, verbose: bool = False):
		if self.trange:
			for attempt in range(1, 4):
				print(f'{color}Downloading {self.trange[0]} (attempt {attempt})...{DownloadRequest.RESET}')
				try:
					self._download()
					del_data()
				except Exception as e:
					print(f'{color}Retry {attempt}/3 for {self.trange[0]}: {e}{DownloadRequest.RESET}')
			print(f'{color}FAILED {self.trange[0]} after 3 attempts{DownloadRequest.RESET}')
		else:
			print(f"No range set for {type(self).__name__} {self.datatype} {self.data_rate} request")


	@abstractmethod
	def _download(self):
		...


	@classmethod
	def from_existing(cls, request: 'DownloadRequest'):
		return cls(request.probe, request.trange, request.datatype, request.data_rate)


	def set_trange(self, start: datetime, end: datetime):
		self.trange = [start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')]

