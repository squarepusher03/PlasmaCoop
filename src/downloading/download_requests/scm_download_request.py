
from pyspedas import mms

from download_request import DownloadRequest

class SCMDownloadRequest(DownloadRequest):
	inst  = 'scm'
	field = 'B'

	def __init__(self, probe, trange, datatype, data_rate):
		super().__init__(probe, trange, datatype, data_rate)


	def _download(self):
		mms.scm(probe=self.probe, trange=self.trange, time_clip=True, datatype=self.datatype, data_rate=self.data_rate,
				latest_version=True, no_update=False)

