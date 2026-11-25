"""CDAWeb (NASA SPDF) as the download source for MMS burst CDFs.

Graham, 2026-10-06: "Switch ALL downloads to CDAWeb at 8 at a time." Measured that evening on the same 100 files
the SDC had delivered: CDAWeb gave 6.3 MB/s one file at a time, 12.7 MB/s four at a time and 20.0 MB/s eight at a
time, against the SDC's 0.3-0.9 MB/s. Every file was byte-identical to the SDC's copy, and no request was refused
or limited in any form (_wsl_scratch/cdaweb_parallel_timing.py).

download() has acquisition's downloader signature and gives its outcomes (OK, RATE_LIMITED, FAILED). It fetches
the files pyspedas would have fetched from the SDC for the same request -- pyspedas' burst rules, reproduced from
pyspedas/projects/mms/mms_load_data.py and mms_files_in_interval.py, so everything that relies on them (the
census's call windows, acquire's look-back, the refetch tail) keeps working:

  * the listing runs from the start of t0's day -- from 23:50 the day before when t0 is within 600 s of midnight
    -- to t1 - 0.1 s at whole seconds, by each file's start stamp (listing_window)
  * of the listed files: every one from the last stamped before t0 onward (select)
  * a file already on disk under the same name and listed size is not fetched again
  * files land in pyspedas' layout: <data_dir>/mms<p>/<inst>/brst/l2/[<mode>/]YYYY/MM/DD/<name>

CDAWeb lists a dataset by each file's data span (orig_data, by StartTime/EndTime), the SDC by the stamp in the
file's name, and the two differ: across all 1,517,824 MMS burst files StartTime runs from 3 s before the name's
stamp to 463 s after it (2026-10-06). A tight query missed a call's last file twice in 19 (a file named 18:10:00
whose data starts at 18:10:03). So the query reaches QUERY_MARGIN past both ends, and the answer is cut back to the
name stamps the SDC would list.

Up to `concurrent` files download at once. Each goes to <name>.part beside its destination and is moved into place
only when its size matches the listing and it starts with a CDF's magic number: a rate-limit or error page can
never land under a .cdf name, the way the SDC's 429 pages once did. A 429 or 503, or any Retry-After, stops new
files from starting and makes the call RATE_LIMITED; files that finished stay, and the retry skips them by their
size. A cancel stops the files under way at their next block and deletes their .part files (2026-11-24); the
files already downloaded stay. A 404 for a listed file is logged and left out: it is not there to fetch, so retrying cannot help.
Connection errors, timeouts, other 5xx and bad transfers are retried FILE_TRIES times, then the call is FAILED.
"""
import os
import threading
import time
from bisect import bisect_left
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import app_config

OK, RATE_LIMITED, FAILED = 'ok', 'rate_limited', 'failed'      # acquisition's outcomes, same strings

ORIG_DATA = 'https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/{ds}/orig_data/{t0},{t1}'
DATASETS = {('scm', 'scb'): 'MMS{p}_SCM_BRST_L2_SCB', ('scm', 'schb'): 'MMS{p}_SCM_BRST_L2_SCHB',
            ('edp', 'dce'): 'MMS{p}_EDP_BRST_L2_DCE', ('fgm', ''): 'MMS{p}_FGM_BRST_L2'}
LIMIT_STATUSES = (429, 503)
TIMEOUT = (30, 300)              # connect, read (seconds without a byte)
FILE_TRIES = 3
CHUNK = 1 << 20
CDF_MAGIC = (b'\xcd\xf3', b'\xcd\xf2')     # acquisition._CDF_MAGIC
REPLACE_TRIES = 5                # a virus scanner or indexer can hold a fresh file for a moment
QUERY_MARGIN = timedelta(days=1)   # 186x the widest StartTime/name-stamp offset in the archive (463 s)

Listed = namedtuple('Listed', 'stamp name url size')


class _Limited(Exception):
	def __init__(self, why, retry_after=None):
		super().__init__(why)
		self.retry_after = retry_after


def _session():
	import requests
	return requests.Session()


# Tests swap this for a fake; nothing else in this module touches the network.
session_factory = _session


def _stamp(name):
	return datetime.strptime(name.rsplit('_v', 1)[0][-14:], '%Y%m%d%H%M%S')


def listing_window(t0, t1):
	"""(first, last) stamp the SDC listing pyspedas asks for would hold, both inclusive.

	pyspedas formats the request range to whole seconds, lists from the start of t0's day (600 s earlier when t0
	is within 600 s of midnight -- its "kludge" for bursts that start the previous evening), and ends the listing
	at t1 - 0.1 s, again at whole seconds."""
	t0, t1 = t0.replace(microsecond=0), t1.replace(microsecond=0)
	day = datetime(t0.year, t0.month, t0.day)
	first = day - timedelta(seconds=600) if (t0 - day).total_seconds() <= 600 else day
	last = (t1 - timedelta(seconds=0.1)).replace(microsecond=0)
	return first, last


def select(listed, t0):
	"""mms_files_in_interval: from the last file stamped before t0 to the end of the listing."""
	listed = sorted(listed)
	idx = bisect_left([f.stamp for f in listed], t0.replace(microsecond=0))
	return listed[max(idx - 1, 0):]


def local_path(data_dir, inst, mode, probe, name):
	"""Where pyspedas puts a burst file (mms_load_data: ..., drate, lvl, dtype, YYYY, MM, DD)."""
	t = _stamp(name)
	return os.path.join(data_dir, f'mms{probe}', inst, 'brst', 'l2', *([mode] if mode else []),
	                    f'{t:%Y}', f'{t:%m}', f'{t:%d}', name)


def dataset(inst, mode, probe):
	try:
		return DATASETS[(inst, mode)].format(p=probe)
	except KeyError:
		raise ValueError(f'no CDAWeb dataset for {inst} {mode!r}: only SCM scb/schb, EDP dce and FGM burst '
		                 f'are mapped') from None


def list_files(session, ds, first, last):
	"""[Listed] for every file CDAWeb has in `ds` stamped (by name) in [first, last]."""
	url = ORIG_DATA.format(ds=ds, t0=(first - QUERY_MARGIN).strftime('%Y%m%dT%H%M%SZ'),
	                       t1=(last + QUERY_MARGIN).strftime('%Y%m%dT%H%M%SZ'))
	r = session.get(url, headers={'Accept': 'application/json'}, timeout=TIMEOUT)
	if r.status_code in LIMIT_STATUSES or r.headers.get('Retry-After'):
		raise _Limited(f'listing answered HTTP {r.status_code}', r.headers.get('Retry-After'))
	r.raise_for_status()
	out = []
	for d in r.json().get('FileDescription', []):
		name = d['Name'].rsplit('/', 1)[-1]
		try:
			stamp = _stamp(name)
		except ValueError:
			continue
		if first <= stamp <= last:
			out.append(Listed(stamp, name, d['Name'], int(d['Length'])))
	return sorted(out)


def _replace(src, dst):
	for attempt in range(REPLACE_TRIES):
		try:
			os.replace(src, dst)
			return
		except PermissionError:
			if attempt == REPLACE_TRIES - 1:
				raise
			time.sleep(0.2 * (attempt + 1))


def _remove_quietly(path):
	try:
		os.remove(path)
	except OSError:
		pass


# Cancelling (Graham, 2026-11-24: "the only files that should be deleted are ones being currently downloaded and
# output artifacts currently being generated, the rest should persist"): a download under way stops at its next
# block and its .part is deleted; files already downloaded stay. The .part files open now are listed here so a force
# quit can delete them too (discard_partial_downloads); a stale .part in a folder a download uses is deleted then.
_downloading = set()
_downloading_lock = threading.Lock()


def discard_partial_downloads():
	"""Delete every .part a download of this process has open now (force quit). The paths not deleted (a file
	still held open on Windows)."""
	with _downloading_lock:
		paths = list(_downloading)
	left = []
	for p in paths:
		try:
			os.remove(p)
		except FileNotFoundError:
			pass
		except OSError:
			left.append(p)
	return left


def _sweep_stale_parts(folder):
	"""Delete .part files left in `folder` by a process that is gone (a crash or a force quit), not ours open now."""
	try:
		names = [n for n in os.listdir(folder) if n.endswith('.part')]
	except OSError:
		return
	with _downloading_lock:
		ours = set(_downloading)
	for n in names:
		p = os.path.join(folder, n)
		if p not in ours:
			_remove_quietly(p)


def _one(session, f, dest, stop, cancelled=lambda: False):
	"""Download one file. ('ok' | 'limited' | 'not_found' | 'failed' | 'skipped' | 'cancelled', detail)."""
	import requests
	part = dest + '.part'
	why = ''
	for attempt in range(1, FILE_TRIES + 1):
		if attempt > 1:
			time.sleep(2 * (attempt - 1))
		if stop.is_set() or cancelled():
			return 'skipped', 'stopped before it started'
		try:
			with session.get(f.url, stream=True, timeout=TIMEOUT, headers={'Accept-Encoding': 'identity'}) as r:
				if r.status_code in LIMIT_STATUSES or r.headers.get('Retry-After'):
					stop.set()
					return 'limited', (f'HTTP {r.status_code}', r.headers.get('Retry-After'))
				if r.status_code == 404:
					return 'not_found', 'HTTP 404'
				if r.status_code != 200:
					why = f'HTTP {r.status_code}'
					if r.status_code >= 500:
						continue                 # retried, after the wait at the top
					return 'failed', why
				os.makedirs(os.path.dirname(dest), exist_ok=True)
				n, head, stopped = 0, b'', False
				with _downloading_lock:
					_downloading.add(part)
				try:
					with open(part, 'wb') as fh:
						for block in r.iter_content(CHUNK):
							if cancelled():
								stopped = True
								break
							if len(head) < 4096:
								head += block[:4096 - len(head)]
							fh.write(block)
							n += len(block)
					if stopped:
						_remove_quietly(part)          # closed above; a partial file never stays
						return 'cancelled', n
				finally:
					with _downloading_lock:
						_downloading.discard(part)
			if head.startswith(CDF_MAGIC) and n == f.size:
				_replace(part, dest)
				return 'ok', n
			_remove_quietly(part)
			text = head[:512].decode('latin-1').lower()
			if not head.startswith(CDF_MAGIC) and ('429' in text or 'too many requests' in text):
				stop.set()
				return 'limited', ('a rate-limit page', None)
			why = f'{n} bytes, listed {f.size}' + ('' if head.startswith(CDF_MAGIC) else f', starts {head[:40]!r}')
		except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError, OSError) as exc:
			_remove_quietly(part)
			why = f'{type(exc).__name__}: {str(exc)[:120]}'
	return 'failed', why


def _archive_held(data_dir, inst, mode, probe):
	"""{name: CDF size} of the B and E files the preprocessed archive holds (preproc_archive.held): once a CDF is
	archived and deleted, the same name at the same size is not downloaded again. {} while the archive is off."""
	if inst not in ('scm', 'edp'):
		return {}
	import preproc_archive
	return preproc_archive.held(data_dir, inst, mode, probe)


def download(inst, mode, probes, t0, t1, data_dir, *, concurrent=None, cancelled=None, log=print):
	"""Fetch what pyspedas would fetch for [t0, t1], `concurrent` files at a time. OK, RATE_LIMITED or FAILED."""
	app_config.require_drive(data_dir)
	concurrent = concurrent or app_config.download_concurrent()
	cancelled = cancelled or (lambda: False)
	first, last = listing_window(t0, t1)
	limited = failed = False
	for probe in probes:
		ds = dataset(inst, mode, probe)
		session = session_factory()
		try:
			listed = list_files(session, ds, first, last)
		except _Limited as exc:
			log(f'CDAWeb refused the {ds} listing ({exc}); waiting before trying again.')
			return RATE_LIMITED
		except Exception as exc:  # noqa: BLE001 -- no listing, no download; the caller decides what next
			log(f'CDAWeb listing for {ds} failed: {type(exc).__name__}: {str(exc)[:200]}')
			return FAILED
		todo = []
		held = _archive_held(data_dir, inst, mode, probe)
		for f in select(listed, t0):
			dest = local_path(data_dir, inst, mode, probe, f.name)
			if held.get(f.name) == f.size:
				continue                         # the preprocessed archive holds this file at its listed size
			try:
				if os.path.getsize(dest) == f.size:
					continue                     # pyspedas' rule: same name, same size -- already here
			except OSError:
				pass
			others = _other_versions(dest)
			if others:
				log(f'  {f.name}: other version(s) of this file already on disk: {", ".join(others)}')
			todo.append((f, dest))
		if not todo:
			continue
		for folder in {os.path.dirname(d) for _f, d in todo}:
			_sweep_stale_parts(folder)
		total = sum(f.size for f, _d in todo)
		log(f'Downloading {len(todo)} {inst.upper()} file(s) for MMS{probe} from CDAWeb, {total / 1e6:.0f} MB, '
		    f'{min(concurrent, len(todo))} at a time ...')
		stop = threading.Event()
		local = threading.local()

		def work(item):
			if not hasattr(local, 'session'):
				local.session = session_factory()
			return item, _one(local.session, item[0], item[1], stop, cancelled)

		t_start = time.monotonic()
		got = 0
		with ThreadPoolExecutor(max_workers=min(concurrent, len(todo))) as pool:
			futures = [pool.submit(work, item) for item in todo]
			for fut in as_completed(futures):
				(f, dest), (outcome, detail) = fut.result()
				if outcome == 'ok':
					got += detail
					log(f'  {f.name} ({detail / 1e6:.1f} MB) -> {os.path.dirname(dest)}')
				elif outcome == 'limited':
					limited = True
					log(f'  {f.name}: CDAWeb is limiting requests ({detail[0]}'
					    f'{", Retry-After " + str(detail[1]) if detail[1] else ""}); no new files start.')
				elif outcome == 'not_found':
					log(f'  {f.name}: listed by CDAWeb but not found there (404); left out.')
				elif outcome == 'failed':
					failed = True
					log(f'  {f.name}: download failed after {FILE_TRIES} tries ({detail}).')
				elif outcome == 'cancelled':
					log(f'  {f.name}: stopped by the cancel after {detail / 1e6:.1f} MB; its partial file is deleted.')
				if cancelled() and not stop.is_set():
					stop.set()
					failed = True
					log('Cancelled: no new files start, the ones under way stop and their partial files are deleted; '
					    'the files already downloaded stay.')
		el = time.monotonic() - t_start
		log(f'CDAWeb: {got / 1e6:.0f} MB in {el:.0f} s ({got / 1e6 / max(el, 1e-6):.1f} MB/s).')
	if limited:
		return RATE_LIMITED
	return FAILED if failed else OK


def _other_versions(dest):
	"""Names of other versions of dest's file in its folder (same stamp, another _vX.Y.Z)."""
	folder, name = os.path.split(dest)
	stem = name.rsplit('_v', 1)[0]
	try:
		return sorted(n for n in os.listdir(folder) if n != name and n.endswith('.cdf') and n.rsplit('_v', 1)[0] == stem)
	except OSError:
		return []
