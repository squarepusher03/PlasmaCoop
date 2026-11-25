"""Where acquisition downloads from (Graham, 2026-10-06: "Switch ALL downloads to CDAWeb at 8 at a time"): the
settings file's source, each downloader with its own pacer, and every download path taking the same one."""
import os
from datetime import datetime, timedelta

import pytest

import acquisition
import app_config
import ffttest

QUIET = lambda *a, **k: None


@pytest.fixture
def clock(monkeypatch):
	"""Fresh pacers on a fake clock; sleeps are recorded, not slept."""
	state = {'t': 1000.0, 'sleeps': []}

	def sleep(dt):
		state['sleeps'].append(dt)
		state['t'] += dt

	monkeypatch.setattr(acquisition, '_sleep', sleep)
	monkeypatch.setattr(acquisition, '_now', lambda: state['t'])
	monkeypatch.setattr(acquisition, '_wall', lambda: 1_800_000_000.0 + state['t'])
	monkeypatch.setattr(acquisition, '_PACER', acquisition._SdcPacer())
	monkeypatch.setattr(acquisition, '_CDAWEB_PACER', acquisition._SdcPacer(
		base_s=0, max_s=acquisition.SDC_GAP_MAX_S, state_name='cdaweb_pacer.json', label='CDAWeb'))
	return state


def test_cdaweb_is_the_default_source():
	assert acquisition._default_downloader() is acquisition._cdaweb_download


def test_the_settings_file_can_name_the_sdc_again():
	app_config.update('download', {'source': 'sdc'})
	assert acquisition._default_downloader() is acquisition._sdc_download


def test_each_downloader_brings_its_own_pacer(clock):
	assert acquisition._pacer_for(acquisition._cdaweb_download) is acquisition._CDAWEB_PACER
	assert acquisition._pacer_for(acquisition._sdc_download) is acquisition._PACER
	assert acquisition._pacer_for(lambda *a: acquisition.OK) is acquisition._PACER     # a test's fake: the SDC's
	assert acquisition._CDAWEB_PACER.base_s == 0 and acquisition._PACER.base_s == acquisition.SDC_CALL_GAP_S


def test_cdaweb_calls_follow_each_other_with_no_wait(tmp_path, clock):
	calls = []

	def fake(inst, mode, probes, t0, t1, data_dir, cancelled=None, log=None):
		calls.append((probes, cancelled is not None, log is not None))
		return acquisition.OK

	fake.pacer_name, fake.wants_context = '_CDAWEB_PACER', True
	start = datetime(2021, 2, 23, 10)
	acquisition.acquire('B', (1, 2, 3), start, start + timedelta(minutes=1), data_dir=str(tmp_path),
	                    pickle_root=str(tmp_path / 'cache'), downloader=fake, log=QUIET)
	assert calls == [((1,), True, True), ((2,), True, True), ((3,), True, True)]   # cancel and log passed in
	assert clock['sleeps'] == []                                                   # no gap between them


def test_the_sdc_still_waits_30_s_between_calls(tmp_path, clock):
	start = datetime(2021, 2, 23, 10)
	acquisition.acquire('B', (1, 2), start, start + timedelta(minutes=1), data_dir=str(tmp_path),
	                    pickle_root=str(tmp_path / 'cache'), downloader=lambda *a: acquisition.OK, log=QUIET)
	assert sum(clock['sleeps']) == pytest.approx(acquisition.SDC_CALL_GAP_S)


def test_a_cdaweb_429_waits_300_s_then_halves_back_to_no_gap(clock):
	pacer = acquisition._CDAWEB_PACER
	pacer.settle(True, False, log=QUIET)
	assert pacer.gap_s == 300
	gaps = []
	while pacer.gap_s:
		pacer.settle(False, True, log=QUIET)
		gaps.append(pacer.gap_s)
	assert gaps[:3] == [150, 75, 37.5] and gaps[-1] == 0 and len(gaps) < 12


def test_attempt_download_takes_the_settings_files_source(tmp_path, monkeypatch, clock):
	calls = []

	def fake(inst, mode, probes, t0, t1, data_dir, cancelled=None, log=None):
		calls.append((inst, mode, tuple(probes)))
		return acquisition.OK

	fake.pacer_name, fake.wants_context = '_CDAWEB_PACER', True
	monkeypatch.setattr(acquisition, '_cdaweb_download', fake)
	start = datetime(2021, 2, 23, 10)
	ffttest.attempt_download('edp', 'dce', [1, 2], start, start + timedelta(minutes=1), str(tmp_path))
	assert calls == [('edp', 'dce', (1,)), ('edp', 'dce', (2,))]


def test_the_cdaweb_downloader_hands_the_settings_concurrency_to_cdaweb(monkeypatch):
	seen = {}

	def download(inst, mode, probes, t0, t1, data_dir, concurrent=None, cancelled=None, log=None):
		seen.update(concurrent=concurrent)
		return acquisition.OK

	monkeypatch.setattr(acquisition.cdaweb, 'download', download)
	app_config.update('download', {'concurrent': 5})
	start = datetime(2021, 2, 23, 10)
	assert acquisition._cdaweb_download('scm', 'scb', (1,), start, start, '/d') == acquisition.OK
	assert seen == {'concurrent': 5}
