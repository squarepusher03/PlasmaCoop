"""Rate-limit pages and the SDC pacer (Graham, 2026-10-01).

  "All files should be checked that they don't just download as rate-limit pages, and if
   they are, delete them and make the amount of time between requests longer"
  "If you hit a 429, wait 300s, then dip down by half every time a CDF comes through, with
   a minimum wait time of 30 seconds"
  The wait carries across restarts ({pickle_root}/sdc_pacer.json).

The SDC's 429 reply, as pyspedas saves it under the CDF's name (75 bytes, seen 1,040 times
on E:), is PAGE below. Everything runs on a fake clock in a temp folder; nothing calls the SDC.
"""
import json
import os
import sys
from datetime import datetime, timedelta

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault('MPLBACKEND', 'Agg')

import acquisition
import cdf_validation
import ffttest

PAGE = b'HTTP/1.1 429 Too Many Requests\r\nContent-Type: text/html\r\nRetry-After: 300\r\n'
QUIET = lambda *a, **k: None


@pytest.fixture(autouse=True)
def clock(monkeypatch):
	"""A fresh pacer on fake monotonic and wall clocks."""
	state = {'t': 1000.0, 'wall': 1_800_000_000.0, 'sleeps': []}

	def sleep(dt):
		state['sleeps'].append(dt)
		state['t'] += dt
		state['wall'] += dt

	monkeypatch.setattr(acquisition, '_now', lambda: state['t'])
	monkeypatch.setattr(acquisition, '_wall', lambda: state['wall'])
	monkeypatch.setattr(acquisition, '_sleep', sleep)
	monkeypatch.setattr(acquisition, '_PACER', acquisition._SdcPacer())
	monkeypatch.setattr(ffttest, '_CHECKED_CDFS', set())
	mode = cdf_validation.get_mode()
	yield state
	if mode is not None:                                   # acquire switches validation off
		from spacepy import pycdf
		pycdf.lib._library.CDFsetValidate(pycdf.const.VALIDATEFILEon if mode else pycdf.const.VALIDATEFILEoff)


def scb_path(data, stamp='20210223100000', probe=1):
	return os.path.join(data, f'mms{probe}', 'scm', 'brst', 'l2', 'scb', stamp[:4], stamp[4:6], stamp[6:8],
	                    f'mms{probe}_scm_brst_l2_scb_{stamp}_v1.0.0.cdf')


def write_page(path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'wb') as fh:
		fh.write(PAGE)
	return path


def test_the_saved_reply_is_recognised_as_a_rate_limit_page():
	assert acquisition._is_rate_limit_page(PAGE)


# ---------------------------------------------------------------------------
# the pacer remembers across runs
# ---------------------------------------------------------------------------

def test_a_new_run_keeps_the_300_s_an_earlier_run_was_asked_for(tmp_path, clock):
	first = acquisition._SdcPacer()
	first.attach(str(tmp_path), log=QUIET)
	first.called()
	first.settle(rate_limited=True, got_cdf=False, log=QUIET)
	saved = json.loads((tmp_path / acquisition.PACER_STATE_NAME).read_text())
	assert saved['gap_s'] == 300

	clock['t'] += 40          # 40 s later a new process starts (its monotonic clock differs)
	clock['wall'] += 40
	second = acquisition._SdcPacer()
	second.attach(str(tmp_path), log=QUIET)
	assert second.gap_s == 300
	assert second.wait(log=QUIET)
	assert sum(clock['sleeps']) == pytest.approx(260)      # the rest of the 300 s


def test_the_halved_gap_is_what_the_next_run_starts_from(tmp_path, clock):
	pacer = acquisition._SdcPacer()
	pacer.attach(str(tmp_path), log=QUIET)
	pacer.called()
	pacer.settle(rate_limited=True, got_cdf=False, log=QUIET)
	pacer.called()
	pacer.settle(rate_limited=False, got_cdf=True, log=QUIET)
	clock['wall'] += 1000                                  # long after: nothing left to wait
	again = acquisition._SdcPacer()
	again.attach(str(tmp_path), log=QUIET)
	assert again.gap_s == 150
	assert again.wait(log=QUIET) and clock['sleeps'] == []


@pytest.mark.parametrize('content', [None, b'', b'not json', b'{"gap_s": "x"}'])
def test_a_missing_or_unreadable_memory_means_the_normal_gap(tmp_path, content):
	if content is not None:
		(tmp_path / acquisition.PACER_STATE_NAME).write_bytes(content)
	pacer = acquisition._SdcPacer()
	pacer.attach(str(tmp_path), log=QUIET)
	assert pacer.gap_s == 30 and pacer.last_end is None


@pytest.mark.parametrize('source, pacer, state_name', [('sdc', '_PACER', acquisition.PACER_STATE_NAME),
                                                       ('cdaweb', '_CDAWEB_PACER', 'cdaweb_pacer.json')])
def test_acquire_attaches_the_memory_to_its_pickle_root(tmp_path, source, pacer, state_name):
	"""The source the settings file names brings its own pacer and memory file (CDAWeb's since 2026-10-06)."""
	import app_config
	app_config.update('download', {'source': source})
	data, cache = tmp_path / 'pydata', tmp_path / 'cache'
	os.makedirs(data)
	acquisition.acquire('B', (1,), datetime(2021, 2, 23, 10), datetime(2021, 2, 23, 10, 1),
	                    data_dir=str(data), pickle_root=str(cache), download=False, log=QUIET)
	assert getattr(acquisition, pacer).state_path == os.path.join(str(cache), state_name)


# ---------------------------------------------------------------------------
# every path deletes a rate-limit page, uncounted
# ---------------------------------------------------------------------------

def test_the_file_check_deletes_a_page_without_counting_it(tmp_path, capsys):
	path = write_page(scb_path(str(tmp_path / 'pydata')))
	log = tmp_path / 'corrupted.txt'
	got = ffttest.check_cdfs([(path, 'B', 1)], corrupted_log=str(log), pickle_root=str(tmp_path / 'cache'))
	assert got == [None] and not os.path.exists(path)
	assert cdf_validation.attempts(path, 'B', pickle_root=str(tmp_path / 'cache')) == 0
	assert log.read_text().strip() == path                 # still handed to the redownload loop
	assert 'rate-limit (429) page, not counted' in capsys.readouterr().out


@pytest.fixture
def sdc_source():
	"""The old-download-path tests are about the SDC: the settings file names it (CDAWeb is the default since
	2026-10-06, and conftest gives each test its own settings file)."""
	import app_config
	app_config.update('download', {'source': 'sdc'})


def test_the_old_download_path_deletes_a_page_and_waits_300_s(tmp_path, monkeypatch, sdc_source):
	data = str(tmp_path / 'pydata')
	target = scb_path(data)
	calls = []

	def sdc(inst, mode, probes, start, end, data_dir):
		calls.append((inst, mode, tuple(probes)))
		write_page(target)
		return acquisition.OK

	monkeypatch.setattr(acquisition, '_sdc_download', sdc)
	start = datetime(2021, 2, 23, 10)
	ffttest.attempt_download('scm', 'scb', 1, start, start + timedelta(minutes=1), data)
	assert calls == [('scm', 'scb', (1,))]
	assert not os.path.exists(target)                      # the page did not survive
	assert acquisition._PACER.gap_s == 300                 # and the next call waits the full 300 s


def test_the_old_download_path_halves_the_gap_when_a_cdf_arrives(tmp_path, monkeypatch, sdc_source):
	from test_direct_cdf import write_cdf
	data = str(tmp_path / 'pydata')
	start = datetime(2021, 2, 23, 10)

	def sdc(inst, mode, probes, t0, t1, data_dir):
		write_cdf(data_dir, 'B', '20210223100000', start, 8192)
		return acquisition.OK

	monkeypatch.setattr(acquisition, '_sdc_download', sdc)
	acquisition._PACER.gap_s = 300
	ffttest.attempt_download('scm', 'scb', [1], start, start + timedelta(minutes=1), data)
	assert acquisition._PACER.gap_s == 150


def test_the_old_download_path_makes_one_call_per_probe(tmp_path, monkeypatch, sdc_source):
	calls = []
	monkeypatch.setattr(acquisition, '_sdc_download',
	                    lambda inst, mode, probes, t0, t1, d: calls.append(tuple(probes)) or acquisition.OK)
	start = datetime(2021, 2, 23, 10)
	ffttest.attempt_download('fgm', '', [1, 2], start, start + timedelta(minutes=1), str(tmp_path))
	assert calls == [(1,), (2,)]
