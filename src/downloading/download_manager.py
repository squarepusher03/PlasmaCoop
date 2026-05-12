import os, re, threading

from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Manager
from multiprocessing.synchronize import Lock

from pandas import read_sql_table

from download_requests.download_request import DownloadRequest
from download_requests.verbose_severity import VerboseSeverity
from ffttest import _cdf_to_df


class DownloadManager:
	WORKER_COUNT = 6
	COLORS = ['\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m']

	_requests: list[DownloadRequest]
	_thread_colors: dict[int, str]
	_data_dir: str
	_pickle_root: str
	_corrupted_log: str


	def __init__(self, download_root: str):
		self._requests = []
		self._thread_colors = {}
		self._data_dir = download_root + '/pydata'
		self._pickle_root = download_root + '/.serialized'
		self._corrupted_log = download_root + '/corrupted.txt'


	def get_color(self):
		tid = threading.get_ident()
		with (threading.Lock()):
			if tid not in self._thread_colors:
				self._thread_colors[tid] = \
					DownloadManager.COLORS[len(self._thread_colors) % len(DownloadManager.COLORS)]
		return self._thread_colors[tid]


	def _ensure_data_pickle(self, cdf_path: str, key: str, lock: Lock, probe: int = 1,
	                        verbose: VerboseSeverity = VerboseSeverity.NONE):
		"""
		Ensure a per-file data pickle exists for cdf_path.
		If not, generate it from the CDF and save it. If the CDF is unreadable,
		delete it, log it to the corrupted log under lock, and return None.
		Pickle path: {pickle_root}/mms/{probe}/{inst}/{mode}/{YYYY}/{MM}/{DD}/{HHMMSS}.pkl
		Returns the pickle path, or None if the CDF was corrupt.
		"""
		m = re.search(r'_(\d{4})(\d{2})(\d{2})(\d{6})_', cdf_path)
		if not m:
			raise ValueError(f'Cannot parse timestamp from CDF path: {cdf_path}')
		timestamp = m.group(4)

		field = key[0]
		inst  = 'scm' if field == 'B' else 'edp'
		mode  = 'scb' if field == 'B' else 'dce'

		pickle_path = os.path.join(
			self._pickle_root,
			f'mms/{probe}/{inst}/{mode}/{m.group(1)}/{m.group(2)}/{m.group(3)}/{timestamp}.pkl',
		)

		if os.path.exists(pickle_path):
			return pickle_path

		if verbose >= VerboseSeverity.DEBUG:
			print(f'Generating data pickle from {os.path.basename(cdf_path)}...')

		try:
			df = _cdf_to_df(cdf_path, key, probe=probe)
		except Exception as e:
			size = os.path.getsize(cdf_path)

			if verbose >= VerboseSeverity.WARN:
				print(f'Bad CDF, deleting: {os.path.basename(cdf_path)} — {size} bytes, {type(e).__name__}: {e}')

			try:
				os.remove(cdf_path)
			except PermissionError:
				if verbose >= VerboseSeverity.ERROR:
					print(f'Cannot delete locked file: {os.path.basename(cdf_path)}')
			with lock:
				with open(self._corrupted_log, 'a') as log:
					log.write(cdf_path + '\n')
			return None

		os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
		df.to_pickle(pickle_path)

		if verbose >= VerboseSeverity.DEBUG:
			print(f'Saved: {pickle_path}')

		return pickle_path


	def add_request(self, request: DownloadRequest):
		if request.trange[1] - request.trange[0] > timedelta(days=1):
			self._add_ranged_request(request, request.trange[0], request.trange[1])
		else:
			self._requests.append(request)

	def add_range_of_requests(self, ):


	def _add_ranged_request(self, request: DownloadRequest, start: datetime, end: datetime):
		day = start

		while day < end:
			new_request = DownloadRequest.from_existing(request)
			new_request.set_trange(day, day + timedelta(days=1))
			self._requests.append(new_request)

			day = day + timedelta(days=1)


	def process_requests(self, verbose: VerboseSeverity = VerboseSeverity.NONE):
		if verbose is VerboseSeverity.INFO:
			dl_start = datetime.now()
			print(f'Starting download at {dl_start.strftime("%Y-%m-%d %H:%M:%S")}...')

		with ProcessPoolExecutor(max_workers=DownloadManager.WORKER_COUNT) as pool:
			pool.map(lambda x: x.download(self.get_color(), verbose), self._requests)

		if verbose is VerboseSeverity.INFO:
			dl_end = datetime.now()
			print(f'Download complete at {dl_end.strftime("%Y-%m-%d %H:%M:%S")} taking {dl_end - dl_start}.')

			conv_start = datetime.now()
			print(f'Starting conversion at {conv_start.strftime("%Y-%m-%d %H:%M:%S")}...')
			print('Converting CDFs to pickles...')

		with Manager() as manager:
			lock = manager.Lock()
			with ProcessPoolExecutor(max_workers=DownloadManager.WORKER_COUNT) as pool:
				for req in self._requests:
					cdfs = req.find_cdfs(self._data_dir)
					if not cdfs:
						continue
					fn = partial(self._ensure_data_pickle, key=req.key, lock=lock, probe=int(req.probe))
					list(pool.map(fn, cdfs))

		if verbose is VerboseSeverity.INFO:
			conv_end = datetime.now()
			print(f'Conversion complete at {conv_end.strftime("%Y-%m-%d %H:%M:%S")} taking {conv_end - conv_start}.')


	def redownload_corrupted_data(self, verbose: VerboseSeverity = VerboseSeverity.NONE,):
		if corrupted_log := os.path.abspath(self._corrupted_log):
			with open(corrupted_log) as f:
				corrupted_days = {datetime.strptime(re.search(r'_(\d{8})', line).group(1), '%Y%m%d')
								  for line in f if re.search(r'_(\d{8})', line)}

			if verbose is VerboseSeverity.INFO:
				print(f'Redownloading {len(corrupted_days)} days with corrupted files...')

			os.remove(corrupted_log)



