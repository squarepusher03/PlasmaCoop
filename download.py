#!.venv/bin/python3

import os
import re
import threading
import logging

import pandas as pd

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from abc import ABC, abstractmethod

from pyspedas import mms
from pytplot import del_data

from ffttest import find_cdf_files, ensure_data_pickle

os.chdir("E:/PlasmaCoop/")

data_dir      = 'E:/PlasmaCoop/pydata'
pickle_root   = 'E:/PlasmaCoop/.serialized'
corrupted_log = 'E:/PlasmaCoop/corrupted.txt'

start   = datetime(2019, 8, 16)
end     = datetime(2019, 8, 17)
workers = 6

COLORS = ['\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m']
RESET  = '\033[0m'
thread_colors = {}
color_lock    = threading.Lock()

for _log in ['pyspedas', 'pytplot', 'requests', 'urllib3', '']:
	logging.getLogger(_log).setLevel(logging.CRITICAL)

def get_color():
	tid = threading.get_ident()
	with color_lock:
		if tid not in thread_colors:
			thread_colors[tid] = COLORS[len(thread_colors) % len(COLORS)]
	return thread_colors[tid]

def download_day(day):
	color = get_color()
	nxt  = day + timedelta(days=1)
	rnge = [day.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d')]
	for attempt in range(1, 4):
		try:
			print(f'{color}Downloading {rnge[0]} (attempt {attempt})...{RESET}')
			mms.edp(probe='1', trange=rnge, time_clip=True, datatype='scb', data_rate='brst', latest_version=True, no_update=False)
			del_data()
			print(f'{color}Done {rnge[0]}{RESET}')
			return
		except Exception as e:
			print(f'{color}Retry {attempt}/3 for {rnge[0]}: {e}{RESET}')
	print(f'{color}FAILED {rnge[0]} after 3 attempts{RESET}')


if __name__ == '__main__':
	days = []
	cur = start
	while cur < end:
		days.append(cur)
		cur += timedelta(days=1)


	with ThreadPoolExecutor(max_workers=workers) as pool:
		list(pool.map(download_day, days))

	dl_end = datetime.now()
	print(f'Download complete at {dl_end.strftime("%Y-%m-%d %H:%M:%S")} taking {dl_end - dl_start}.')

	conv_start = datetime.now()
	print(f'Starting conversion at {conv_start.strftime("%Y-%m-%d %H:%M:%S")}...')
	print('Converting CDFs to pickles...')

	cdfs = find_cdf_files(data_dir, 'scm', 'scb', days[0], days[-1] + timedelta(days=1))
	with Manager() as manager:
		lock = manager.Lock()
		fn = partial(ensure_data_pickle, pickle_root=pickle_root, key='B', corrupted_log=corrupted_log, lock=lock)

		with ProcessPoolExecutor(max_workers=8) as pool:
			list(pool.map(fn, cdfs))

	if os.path.exists(corrupted_log):
		with open(corrupted_log) as f:
			corrupted_days = {datetime.strptime(re.search(r'_(\d{8})', line).group(1), '%Y%m%d')
			                  for line in f if re.search(r'_(\d{8})', line)}

		if corrupted_days:
			print(f'Redownloading {len(corrupted_days)} days with corrupted files...')

			os.remove(corrupted_log)

			with ThreadPoolExecutor(max_workers=workers) as pool:
				list(pool.map(download_day, sorted(corrupted_days)))

			cdfs = find_cdf_files(data_dir, 'scm', 'scb', days[0], days[-1] + timedelta(days=1))

			with Manager() as manager:
				lock = manager.Lock()
				fn = partial(ensure_data_pickle, pickle_root=pickle_root, key='B', corrupted_log=corrupted_log, lock=lock)
				with ProcessPoolExecutor(max_workers=8) as pool:
					list(pool.map(fn, cdfs))

	conv_end = datetime.now()
	print(f'Conversion complete at {conv_end.strftime("%Y-%m-%d %H:%M:%S")} taking {conv_end - conv_start}.')
	print(f'Download time: {dl_end - dl_start}\tConversion time: {conv_end - conv_start}')
