#!.venv/bin/python3

import os
import sys
import re
import threading
import logging

import pandas as pd

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

try:
	from pyspedas import mms          # pyspedas < ~2.x: mms at the top level
except ImportError:
	from pyspedas.projects import mms  # pyspedas >= ~2.x: moved under .projects
try:
	from pytplot import del_data      # older releases: separate pytplot package
except ImportError:
	from pyspedas import del_data      # newer releases: bundled into pyspedas itself

from ffttest import find_cdf_files, ensure_data_pickles, check_cdfs, field_burst
from acquisition import acquire
import cdf_validation
from data_paths import DATA_ROOT, DATA_DIR as data_dir, PICKLE_ROOT as pickle_root, CORRUPTED_LOG as corrupted_log

os.chdir(DATA_ROOT)  # per-platform E:/ vs /mnt/e


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

def download_day(day, probe):
	color = get_color()
	nxt  = day + timedelta(days=1)
	rnge = [day.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d')]
	for attempt in range(1, 4):
		try:
			print(f'{color}Downloading {rnge[0]} probe {probe} (attempt {attempt})...{RESET}')
			# datatype=mode ties the download to the same mode value used for
			# discovery/conversion below -- without it, pyspedas falls back to
			# its own default datatype, which for SCM brst data turned out to
			# be schb (high-burst, 16384 Hz), not scb (burst, 8192 Hz).
			# The settings file's source (CDAWeb, 8 files at a time, by default) through acquisition's own
			# fetch, paced and limit-aware, like every other download (Graham, 2026-10-06: "Switch ALL
			# downloads to CDAWeb at 8 at a time"). With source 'sdc' this is the pyspedas download again.
			import acquisition
			letter = {'scm': 'B', 'edp': 'E', 'fgm': 'F'}[inst]
			result = acquisition.Acquired(letter, (int(probe),), day, nxt)
			status, _arrived = acquisition._fetch(acquisition._default_downloader(), inst, mode, (int(probe),), day,
			                                      nxt, data_dir, result, lambda: False, print)
			if status != acquisition.OK:
				raise RuntimeError(f'download {status}')
			print(f'{color}Done {rnge[0]} probe {probe}{RESET}')
			return
		except Exception as e:
			print(f'{color}Retry {attempt}/3 for {rnge[0]} probe {probe}: {e}{RESET}')
	print(f'{color}FAILED {rnge[0]} probe {probe} after 3 attempts{RESET}')


def _download_task(args):
	day, probe = args
	download_day(day, probe)


def _pickle_jobs(probes, key, inst, mode, days):
	"""Combined (cdf_path, key, probe) job list across every requested probe.
	Each probe's CDFs live under their own mms{probe}/... directory, so the
	set is already disjoint across probes -- no dedup needed, just a single
	list to hand to one check_cdfs() call."""
	jobs = []
	for probe in probes:
		cdfs = find_cdf_files(data_dir, inst, mode, days[0], days[-1] + timedelta(days=1), probe=probe, download=False)
		jobs.extend((cdf, key, probe) for cdf in cdfs)
	return jobs


if __name__ == '__main__':
	probes = ["1"]
	start = datetime(2022, 1, 1)
	end = datetime(2022, 1, 2)
	field = 'B'

	if sys.argv[1].lower() == 'help':
		print('usage: download probe#[,probe#...] start end fieldtype\n'
		      'probes: 1..4, comma-separated for multiple (e.g. 1,2,3,4), dates: YYYY-MM-DD, fields: B, E, or FGM')
		sys.exit(0)

	probes = [p.strip() for p in sys.argv[1].split(',')]
	start = datetime.strptime(sys.argv[2], "%Y-%m-%d")
	end   = datetime.strptime(sys.argv[3], "%Y-%m-%d")
	field = sys.argv[4]

	# FGM supplies the background |B| for f_ce; it has no mode subdir under
	# pydata/mms{p}/fgm/brst/l2, so its discovery mode is ''. The conversion key
	# 'F' routes _cdf_to_df / check_cdfs to the FGM branch.
	if field in ('FGM', 'F'):
		inst, mode, key = 'fgm', '', 'F'
	elif field == 'E':
		inst, mode, key = 'edp', 'dce', 'E'
	else:
		(inst, mode), key = field_burst('B')[:2], 'B'     # scb or schb: B's burst selection

	rate = 'brst'
	workers = 6

	print(f"Probes: {probes}")
	print(f"Start time: {start}")
	print(f"End time: {end}")

	days = []
	cur = start
	while cur < end:
		days.append(cur)
		cur += timedelta(days=1)

	dl_start = datetime.now()
	print(f'Download start at {dl_start.strftime("%Y-%m-%d %H:%M:%S")}')
	# The acquisition phase (acquisition.acquire), one day at a time so no single
	# SDC call runs into the download timeout: each probe in its own paced call
	# (Graham, 2026-09-28: 20 s apart, +15 s per 429, capped at 90 s), then every
	# CDF the day can open validated and a corrupt one fetched again, up to 5
	# downloads. It replaces the six parallel pyspedas threads, which ignored the
	# gap between SDC calls; the check and redownload loop below then find
	# nothing left to do.
	for day in days:
		acquire(key, probes, day, day + timedelta(days=1), data_dir=data_dir, pickle_root=pickle_root)

	dl_end = datetime.now()
	print(f'Download complete at {dl_end.strftime("%Y-%m-%d %H:%M:%S")} taking {dl_end - dl_start}.')

	conv_start = datetime.now()
	print(f'Starting CDF check at {conv_start.strftime("%Y-%m-%d %H:%M:%S")}...')
	print('Checking CDFs (the pipeline reads them directly; unreadable ones are deleted and logged)...')

	# download_day() already fetched each day above with latest_version=True,
	# no_update=False -- that's the one freshness/SITL-reprocessing check we
	# need. find_cdf_files defaults to re-checking on its own too, so skip
	# that here to avoid asking the SDC the same question twice.
	jobs = _pickle_jobs(probes, key, inst, mode, days)
	check_cdfs(jobs, corrupted_log=corrupted_log, pickle_root=pickle_root)

	# Retry a corrupted file until it comes down intact or has failed
	# GIVE_UP_AFTER times, rather than the single retry this used to do. The
	# attempt count lives in the validation database keyed on the FILENAME, which
	# is why it survives check_cdfs deleting the file in order to refetch it
	# (Graham, 2026-09-25: "try my redownload loop on the corrupted files and if a
	# file has been attempted to be downloaded 5 times, delete the corrupted file
	# and leave some section in the manifest that holds a list of the corrupted
	# files"). Names that run out of attempts are reported here and written into
	# the interval manifest's corrupted section by the manifest step.
	for _round in range(cdf_validation.GIVE_UP_AFTER):
		if not os.path.exists(corrupted_log):
			break
		corrupted, retry = set(), 0
		with open(corrupted_log) as f:
			for line in f:
				path = line.strip()
				if not path:
					continue
				md = re.search(r'_(\d{8})', path)
				mp = re.search(r'mms(\d+)[/\\]', path)
				if not (md and mp):
					continue
				# Out of attempts: stop asking the SDC for it.
				if cdf_validation.give_up(path, key, pickle_root=pickle_root):
					continue
				retry += 1
				corrupted.add((datetime.strptime(md.group(1), '%Y%m%d'), mp.group(1)))

		os.remove(corrupted_log)
		if not corrupted:
			break
		print(f'Redownloading {len(corrupted)} (day, probe) pairs '
		      f'covering {retry} corrupted files (round {_round + 1})...')

		with ThreadPoolExecutor(max_workers=workers) as pool:
			list(pool.map(_download_task, sorted(corrupted)))

		check_cdfs(jobs, corrupted_log=corrupted_log, pickle_root=pickle_root)

	given_up = cdf_validation.corrupted_names(key, pickle_root=pickle_root)
	if given_up:
		print(f'{len(given_up)} file(s) failed {cdf_validation.GIVE_UP_AFTER} times and '
		      f'were given up on; they are listed in the interval manifests:')
		for name in given_up:
			print(f'  {name}')

	conv_end = datetime.now()
	print(f'CDF check complete at {conv_end.strftime("%Y-%m-%d %H:%M:%S")} taking {conv_end - conv_start}.')
	print(f'Download time: {dl_end - dl_start}\tCheck time: {conv_end - conv_start}\n')
	print(f"Probes: {probes}")
	print(f"Start time: {start}")
	print(f"End time: {end}")
