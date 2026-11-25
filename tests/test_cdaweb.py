"""The CDAWeb downloader (cdaweb.py) against a fake CDAWeb: pyspedas' file choice, pyspedas' layout, nothing but
a whole CDF ever under a .cdf name, limits and failures, 8 at a time."""
import os
import random
import re
import threading
import time
from datetime import datetime, timedelta

import pytest
import requests

import cdaweb

MAGIC = b'\xcd\xf3\x00\x01'


def body(name, size=3000):
	seed = sum(map(ord, name))
	return MAGIC + bytes((seed + i) % 251 for i in range(size - len(MAGIC)))


class Resp:
	def __init__(self, status=200, content=b'', headers=None, json_data=None, delay=0.0, fail_after=None):
		self.status_code, self.content, self.headers = status, content, headers or {}
		self._json, self.delay, self.fail_after = json_data, delay, fail_after
		self.on_close = None

	def __enter__(self):
		return self

	def __exit__(self, *exc):
		if self.on_close:
			self.on_close()
		return False

	def json(self):
		return self._json

	def raise_for_status(self):
		if self.status_code >= 400:
			raise requests.HTTPError(f'HTTP {self.status_code}')

	def iter_content(self, n):
		for i in range(0, len(self.content), 1000):
			if self.delay:
				time.sleep(self.delay)
			if self.fail_after is not None and i >= self.fail_after:
				raise requests.exceptions.ChunkedEncodingError('connection broken')
			yield self.content[i:i + 1000]


class FakeCdaweb:
	"""files: ds -> {name: bytes}. Listing by time overlap, as orig_data answers; each file covers 120 s."""

	ORIG = re.compile(r'/datasets/([^/]+)/orig_data/(\d{8}T\d{6}Z),(\d{8}T\d{6}Z)$')

	def __init__(self, files, behaviour=None, listing_status=200, delay=0.0, start_lag=None):
		self.files, self.behaviour = files, behaviour or {}
		self.start_lag = start_lag or {}             # name -> seconds its data starts after its name stamp
		self.listing_status, self.delay = listing_status, delay
		self.gets, self.lock = [], threading.Lock()
		self.in_flight = self.max_in_flight = 0

	def session(self):
		server = self

		class Session:
			def get(self, url, **kw):
				return server.get(url, **kw)
		return Session()

	def get(self, url, stream=False, **kw):
		m = self.ORIG.search(url)
		if m:
			if self.listing_status != 200:
				return Resp(self.listing_status, headers={'Retry-After': '30'} if self.listing_status == 429 else {})
			ds, t0, t1 = m.group(1), datetime.strptime(m.group(2), '%Y%m%dT%H%M%SZ'), \
				datetime.strptime(m.group(3), '%Y%m%dT%H%M%SZ')
			out = []
			for name, data in self.files.get(ds, {}).items():
				start = cdaweb._stamp(name) + timedelta(seconds=self.start_lag.get(name, 0))
				if start <= t1 and start + timedelta(seconds=120) >= t0:
					out.append({'Name': f'https://cdaweb.example/{ds}/{name}', 'Length': len(data),
					            'StartTime': start.strftime('%Y-%m-%dT%H:%M:%S.000Z')})
			return Resp(json_data={'FileDescription': out})
		name = url.rsplit('/', 1)[1]
		with self.lock:
			self.gets.append(name)
			self.in_flight += 1                       # until the response closes: the whole transfer
			self.max_in_flight = max(self.max_in_flight, self.in_flight)
			n = len([g for g in self.gets if g == name])
		step = self.behaviour.get(name)
		if callable(step):
			resp = step(n)
		elif isinstance(step, Resp):
			resp = step
		else:
			resp = Resp(content=next(d[name] for d in self.files.values() if name in d), delay=self.delay)
		resp.on_close = self._closed
		return resp

	def _closed(self):
		with self.lock:
			self.in_flight -= 1


def names(probe, inst_mode, stamps, version='v2.2.0'):
	tag = {'scb': 'scm_brst_l2_scb', 'schb': 'scm_brst_l2_schb', 'dce': 'edp_brst_l2_dce', 'fgm': 'fgm_brst_l2'}[inst_mode]
	return [f'mms{probe}_{tag}_{s}_{version}.cdf' for s in stamps]


@pytest.fixture
def no_sleep(monkeypatch):
	monkeypatch.setattr(cdaweb.time, 'sleep', lambda s: None)


def use(monkeypatch, server):
	monkeypatch.setattr(cdaweb, 'session_factory', server.session)


def on_disk(root):
	out = set()
	for d, _sub, files in os.walk(root):
		out.update(os.path.relpath(os.path.join(d, f), root).replace(os.sep, '/') for f in files)
	return out


# ---------------------------------------------------------------------------
# pyspedas' rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('t0, first', [
	(datetime(2019, 8, 16, 7, 40), datetime(2019, 8, 16)),                    # mid-day: from the day's start
	(datetime(2019, 8, 16, 0, 5), datetime(2019, 8, 15, 23, 50)),             # within 600 s: 10 min earlier
	(datetime(2019, 8, 16, 0, 10), datetime(2019, 8, 15, 23, 50)),            # exactly 600 s still counts
	(datetime(2019, 8, 16, 0, 10, 1), datetime(2019, 8, 16)),
	(datetime(2019, 8, 16, 0, 0), datetime(2019, 8, 15, 23, 50)),
])
def test_the_listing_starts_where_pyspedas_starts_it(t0, first):
	assert cdaweb.listing_window(t0, t0 + timedelta(hours=1))[0] == first


@pytest.mark.parametrize('t1, last', [
	(datetime(2019, 8, 16, 7, 50), datetime(2019, 8, 16, 7, 49, 59)),         # t1 - 0.1 s, whole seconds
	(datetime(2019, 8, 16, 7, 50, 0, 900000), datetime(2019, 8, 16, 7, 49, 59)),   # pyspedas drops the fraction
])
def test_the_listing_ends_where_pyspedas_ends_it(t1, last):
	assert cdaweb.listing_window(datetime(2019, 8, 16, 7, 40), t1)[1] == last


def test_select_is_pyspedas_mms_files_in_interval():
	mfi = pytest.importorskip('pyspedas.projects.mms.mms_files_in_interval').mms_files_in_interval
	rng = random.Random(7)
	for _case in range(300):
		day = datetime(2019, 8, 16)
		stamps = sorted({day + timedelta(seconds=rng.randrange(0, 86400)) for _ in range(rng.randrange(0, 12))})
		listed = [cdaweb.Listed(s, n, n, 100) for s, n in zip(stamps, names(1, 'scb', [f'{s:%Y%m%d%H%M%S}' for s in stamps]))]
		t0 = day + timedelta(seconds=rng.randrange(0, 86400))
		ours = [f.name for f in cdaweb.select(listed, t0)]
		theirs = [f['file_name'] for f in mfi([{'file_name': f.name, 'timetag': f'{f.stamp:%Y-%m-%dT%H:%M:%S}',
		                                         'file_size': 100} for f in listed],
		                                       [t0.strftime('%Y-%m-%d/%H:%M:%S'), '2019-08-17/00:00:00'])]
		assert ours == theirs, (t0, stamps)


@pytest.mark.parametrize('inst, mode, expected', [
	('scm', 'scb', 'mms2/scm/brst/l2/scb/2019/08/16/'),
	('scm', 'schb', 'mms2/scm/brst/l2/schb/2019/08/16/'),
	('edp', 'dce', 'mms2/edp/brst/l2/dce/2019/08/16/'),
	('fgm', '', 'mms2/fgm/brst/l2/2019/08/16/'),
])
def test_files_land_in_pyspedas_layout(inst, mode, expected):
	name = names(2, mode or 'fgm', ['20190816074533'])[0]
	assert cdaweb.local_path('/d', inst, mode, 2, name).replace(os.sep, '/') == '/d/' + expected + name


def test_an_unmapped_dataset_is_an_error():
	with pytest.raises(ValueError):
		cdaweb.dataset('fpi', 'des', 1)


# ---------------------------------------------------------------------------
# downloading
# ---------------------------------------------------------------------------

DAY = ['20190815235700', '20190816073853', '20190816074323', '20190816074533', '20190816075003', '20190816090000']


def server_for(files_by_ds, **kw):
	return FakeCdaweb({ds: {n: body(n) for n in ns} for ds, ns in files_by_ds.items()}, **kw)


def test_it_fetches_what_pyspedas_would_and_nothing_else(tmp_path, monkeypatch):
	server = server_for({'MMS1_SCM_BRST_L2_SCB': names(1, 'scb', DAY)})
	use(monkeypatch, server)
	status = cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 40), datetime(2019, 8, 16, 7, 50),
	                         str(tmp_path), log=lambda *a: None)
	assert status == cdaweb.OK
	# from the last file before 07:40 (07:38:53) through the end of the listing (07:49:59): not 07:50:03, not
	# 09:00, and not the previous day's 23:57 (listed from 00:00)
	assert on_disk(tmp_path) == {f'mms1/scm/brst/l2/scb/2019/08/16/{n}' for n in names(1, 'scb', DAY[1:4])}
	for n in names(1, 'scb', DAY[1:4]):
		with open(tmp_path / 'mms1/scm/brst/l2/scb/2019/08/16' / n, 'rb') as fh:
			assert fh.read() == body(n)


def test_a_file_whose_data_starts_after_its_name_stamp_is_still_listed(tmp_path, monkeypatch):
	"""CDAWeb lists by the data's StartTime, up to 463 s after the name's stamp; the SDC by the stamp. A call
	ending 2 s after a file's stamp (the census's [first + 1 s, last + 2 s]) must still bring that file."""
	ns = names(1, 'dce', ['20150902170704', '20150902181000'])
	server = server_for({'MMS1_EDP_BRST_L2_DCE': ns}, start_lag={ns[1]: 463})
	use(monkeypatch, server)
	cdaweb.download('edp', 'dce', (1,), datetime(2015, 9, 2, 17, 7, 5), datetime(2015, 9, 2, 18, 10, 2),
	                str(tmp_path), log=lambda *a: None)
	assert on_disk(tmp_path) == {f'mms1/edp/brst/l2/dce/2015/09/02/{n}' for n in ns}


def test_a_request_near_midnight_reaches_into_the_previous_evening(tmp_path, monkeypatch):
	server = server_for({'MMS1_SCM_BRST_L2_SCB': names(1, 'scb', DAY)})
	use(monkeypatch, server)
	cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 0, 0), datetime(2019, 8, 16, 7, 39),
	                str(tmp_path), log=lambda *a: None)
	assert 'mms1/scm/brst/l2/scb/2019/08/15/' + names(1, 'scb', DAY)[0] in on_disk(tmp_path)


def test_a_same_size_file_is_not_fetched_again_but_a_different_size_is(tmp_path, monkeypatch):
	ns = names(1, 'scb', DAY[1:3])
	server = server_for({'MMS1_SCM_BRST_L2_SCB': ns})
	use(monkeypatch, server)
	folder = tmp_path / 'mms1/scm/brst/l2/scb/2019/08/16'
	folder.mkdir(parents=True)
	(folder / ns[0]).write_bytes(b'x' * len(body(ns[0])))         # same size: kept as is (pyspedas' rule)
	(folder / ns[1]).write_bytes(b'short')                        # different size: fetched again
	cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 39), datetime(2019, 8, 16, 7, 44),
	                str(tmp_path), log=lambda *a: None)
	assert server.gets == [ns[1]]
	assert (folder / ns[1]).read_bytes() == body(ns[1])


def test_a_429_stops_new_files_keeps_finished_ones_and_writes_nothing_for_it(tmp_path, monkeypatch):
	ns = names(1, 'dce', DAY[1:5])
	server = server_for({'MMS1_EDP_BRST_L2_DCE': ns}, behaviour={ns[1]: Resp(429, headers={'Retry-After': '300'})})
	use(monkeypatch, server)
	status = cdaweb.download('edp', 'dce', (1,), datetime(2019, 8, 16, 7, 39), datetime(2019, 8, 16, 7, 51),
	                         str(tmp_path), concurrent=1, log=lambda *a: None)
	assert status == cdaweb.RATE_LIMITED
	assert server.gets == ns[:2]                                  # nothing started after the 429
	assert on_disk(tmp_path) == {f'mms1/edp/brst/l2/dce/2019/08/16/{ns[0]}'}


@pytest.mark.parametrize('page, outcome', [
	(b'<html><body>429 Too Many Requests</body></html>', cdaweb.RATE_LIMITED),
	(b'<html><body>Internal error</body></html>', cdaweb.FAILED),
])
def test_a_page_sent_as_200_never_lands_under_a_cdf_name(tmp_path, monkeypatch, no_sleep, page, outcome):
	ns = names(1, 'scb', DAY[1:2])
	server = server_for({'MMS1_SCM_BRST_L2_SCB': ns}, behaviour={ns[0]: lambda n: Resp(content=page)})
	use(monkeypatch, server)
	status = cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 39), datetime(2019, 8, 16, 7, 40),
	                         str(tmp_path), log=lambda *a: None)
	assert status == outcome
	assert on_disk(tmp_path) == set()                             # no .cdf, and no .part left either


def test_a_short_transfer_is_retried_then_failed_and_nothing_is_written(tmp_path, monkeypatch, no_sleep):
	ns = names(1, 'scb', DAY[1:2])
	server = server_for({'MMS1_SCM_BRST_L2_SCB': ns}, behaviour={ns[0]: lambda n: Resp(content=body(ns[0])[:1500])})
	use(monkeypatch, server)
	status = cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 39), datetime(2019, 8, 16, 7, 40),
	                         str(tmp_path), log=lambda *a: None)
	assert status == cdaweb.FAILED
	assert server.gets == ns * cdaweb.FILE_TRIES
	assert on_disk(tmp_path) == set()


def test_a_dropped_connection_is_retried(tmp_path, monkeypatch, no_sleep):
	ns = names(1, 'scb', DAY[1:2])
	good = body(ns[0])
	server = server_for({'MMS1_SCM_BRST_L2_SCB': ns},
	                    behaviour={ns[0]: lambda n: Resp(content=good, fail_after=1000) if n == 1 else Resp(content=good)})
	use(monkeypatch, server)
	status = cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 39), datetime(2019, 8, 16, 7, 40),
	                         str(tmp_path), log=lambda *a: None)
	assert status == cdaweb.OK and server.gets == ns * 2
	assert on_disk(tmp_path) == {f'mms1/scm/brst/l2/scb/2019/08/16/{ns[0]}'}


def test_a_listed_file_that_is_not_there_is_left_out(tmp_path, monkeypatch):
	ns = names(1, 'scb', DAY[1:3])
	server = server_for({'MMS1_SCM_BRST_L2_SCB': ns}, behaviour={ns[0]: Resp(404)})
	use(monkeypatch, server)
	log = []
	status = cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 39), datetime(2019, 8, 16, 7, 44),
	                         str(tmp_path), log=log.append)
	assert status == cdaweb.OK
	assert on_disk(tmp_path) == {f'mms1/scm/brst/l2/scb/2019/08/16/{ns[1]}'}
	assert any('404' in line for line in log)


@pytest.mark.parametrize('listing_status, outcome', [(429, cdaweb.RATE_LIMITED), (503, cdaweb.RATE_LIMITED),
                                                     (500, cdaweb.FAILED)])
def test_a_refused_listing(tmp_path, monkeypatch, listing_status, outcome):
	server = server_for({'MMS1_SCM_BRST_L2_SCB': names(1, 'scb', DAY)}, listing_status=listing_status)
	use(monkeypatch, server)
	assert cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 39), datetime(2019, 8, 16, 7, 44),
	                       str(tmp_path), log=lambda *a: None) == outcome
	assert server.gets == [] and on_disk(tmp_path) == set()


@pytest.mark.parametrize('concurrent', [8, 3])
def test_it_downloads_concurrent_files_at_a_time(tmp_path, monkeypatch, concurrent):
	stamps = [f'20190816{7 + i // 60:02d}{i % 60:02d}00' for i in range(20)]
	ns = names(1, 'scb', stamps)
	server = server_for({'MMS1_SCM_BRST_L2_SCB': ns}, delay=0.05)
	use(monkeypatch, server)
	status = cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 0), datetime(2019, 8, 16, 7, 30),
	                         str(tmp_path), concurrent=concurrent, log=lambda *a: None)
	assert status == cdaweb.OK and len(on_disk(tmp_path)) == 20
	assert server.max_in_flight == concurrent


def test_the_default_is_eight_at_a_time_from_the_settings_file(tmp_path, monkeypatch):
	stamps = [f'20190816{7 + i // 60:02d}{i % 60:02d}00' for i in range(12)]
	server = server_for({'MMS1_SCM_BRST_L2_SCB': names(1, 'scb', stamps)}, delay=0.05)
	use(monkeypatch, server)
	cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 0), datetime(2019, 8, 16, 7, 30), str(tmp_path),
	                log=lambda *a: None)
	assert server.max_in_flight == 8


def test_an_unmounted_drive_downloads_nothing(monkeypatch):
	import app_config
	monkeypatch.setattr(app_config.os, 'name', 'posix')
	monkeypatch.setattr(app_config.os.path, 'ismount', lambda p: False)
	with pytest.raises(OSError):
		cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 0), datetime(2019, 8, 16, 7, 30),
		                '/mnt/q/PlasmaCoop/pydata', log=lambda *a: None)
