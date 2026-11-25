"""Post-process verification's own downloads (post_verify.py), 2026-11-04.

Shares no code with the app's downloader (cdaweb.py, acquisition.py): its own CDAWeb listing, its own choice of
version, its own transfer and checks. So a fault in the app's download path cannot hide itself by giving the
verifier the same wrong file.

  listing   CDAWeb's orig_data web service, one call per dataset and time range; one file per start stamp, the
            latest version, as the app keeps
  download  a few files at a time into a staging folder in the app's own layout
            (mms<p>/<inst>/brst/l2/<mode>/YYYY/MM/DD/<name>), written as .part and renamed only once its size is the
            listed size and it starts with the CDF magic number; up to ATTEMPTS tries per file

The staging folder is post_verify's, never the app's CDF folder: the app's archive never sees these files.
"""
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

LIST_URL = 'https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/{ds}/orig_data/{t0},{t1}'
USER_AGENT = 'plasmacoop-post-verify'
CDF_MAGIC = b'\xcd\xf3\x00\x01'
ATTEMPTS = 4
RETRY_WAIT_S = 10
INSTRUMENT = {'B': 'scm', 'E': 'edp'}


def dataset(probe, field, mode):
	"""CDAWeb's dataset id: MMS1_SCM_BRST_L2_SCB, MMS1_SCM_BRST_L2_SCHB, MMS1_EDP_BRST_L2_DCE."""
	return f'MMS{probe}_{INSTRUMENT[field].upper()}_BRST_L2_{mode.upper()}'


def stamp_of(name):
	"""The 14-digit start stamp in a burst file name, as a datetime."""
	for part in name.split('_'):
		if len(part) == 14 and part.isdigit():
			return datetime.strptime(part, '%Y%m%d%H%M%S')
	raise ValueError(f'no start stamp in {name}')


def listing(probe, field, mode, t0, t1, timeout=180):
	"""[{'name', 'url', 'size', 'stamp'}] of the burst files CDAWeb lists between t0 and t1 (datetimes), one per
	start stamp (the latest version), in stamp order."""
	url = LIST_URL.format(ds=dataset(probe, field, mode), t0=t0.strftime('%Y%m%dT%H%M%SZ'),
	                      t1=t1.strftime('%Y%m%dT%H%M%SZ'))
	req = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': USER_AGENT})
	for attempt in range(ATTEMPTS):
		try:
			with urllib.request.urlopen(req, timeout=timeout) as r:
				body = json.load(r)
			break
		except Exception:  # noqa: BLE001 -- retried, then raised
			if attempt == ATTEMPTS - 1:
				raise
			time.sleep(RETRY_WAIT_S)
	latest = {}
	for f in body.get('FileDescription') or []:
		name = f['Name'].rsplit('/', 1)[-1]
		if not name.endswith('.cdf'):
			continue
		st = stamp_of(name)
		if st not in latest or name > latest[st]['name']:
			latest[st] = {'name': name, 'url': f['Name'], 'size': int(f['Length']), 'stamp': st}
	return [latest[k] for k in sorted(latest)]


def local_path(stage, probe, field, mode, f):
	st = f['stamp']
	return os.path.join(stage, f'mms{probe}', INSTRUMENT[field], 'brst', 'l2', mode, f'{st:%Y}', f'{st:%m}',
	                    f'{st:%d}', f['name'])


def _fetch_one(url, dst, size):
	part = dst + '.part'
	os.makedirs(os.path.dirname(dst), exist_ok=True)
	req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
	with urllib.request.urlopen(req, timeout=600) as r, open(part, 'wb') as fh:
		while True:
			b = r.read(1 << 20)
			if not b:
				break
			fh.write(b)
	with open(part, 'rb') as fh:
		magic = fh.read(4)
	got = os.path.getsize(part)
	if got != size or magic != CDF_MAGIC:
		os.remove(part)
		raise IOError(f'{os.path.basename(dst)}: {got} bytes against {size} listed, magic {magic!r}')
	os.replace(part, dst)


def download(jobs, concurrent=8, cancelled=lambda: False, log=print):
	"""jobs: [(url, dst, size)]. Each file is fetched unless dst already holds the listed size. Returns
	{dst: error text} for the files that failed every attempt."""
	failed = {}

	def one(job):
		url, dst, size = job
		if os.path.exists(dst) and os.path.getsize(dst) == size:
			return
		for attempt in range(ATTEMPTS):
			if cancelled():
				failed[dst] = 'cancelled'
				return
			try:
				_fetch_one(url, dst, size)
				return
			except Exception as exc:  # noqa: BLE001 -- retried, then reported
				if attempt == ATTEMPTS - 1:
					failed[dst] = f'{type(exc).__name__}: {exc}'
					log(f'download failed after {ATTEMPTS} tries: {os.path.basename(dst)} ({exc})')
				else:
					time.sleep(RETRY_WAIT_S)

	with ThreadPoolExecutor(max_workers=max(1, int(concurrent))) as pool:
		list(pool.map(one, jobs))
	return failed
