"""
The acquisition phase is wired into every entry point, not just main.py's.

Graham, 2026-09-27: acquire all the CDFs for a range, validate them, retry the bad
ones, and only then compute; nothing after it validates or deletes. 2026-09-29:
"After this, do the wiring." Each entry point is checked for two things: it
acquires before anything reads a CDF, and what reads CDFs afterwards never
downloads. acquire itself is stubbed -- tests/test_acquisition.py covers it --
so nothing here touches E: or the SDC.

Run (the `plasma` conda env):

    python -m pytest -q tests/test_acquisition_wiring.py
"""
from __future__ import annotations

import inspect
import os
import re
import sys
from datetime import datetime, timedelta

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')

import acquisition
import download_progress as dp
import event_export as ee
import plot_biggest_spikes as pbs
import spike_biggest_burst_fft as sbf

RNG = ee.RequestRange(datetime(2021, 2, 23, 10, 0), datetime(2021, 2, 23, 12, 0))
W = (datetime(2021, 2, 23, 11, 0), datetime(2021, 2, 23, 11, 5))


def _acquired(letter, probes, start, end, unresolved=()):
	got = acquisition.Acquired(letter, tuple(probes), start, end)
	got.unresolved = set(unresolved)
	return got


# ---------------------------------------------------------------- event UI

def test_check_coverage_validates_before_it_counts(monkeypatch):
	order = []

	def fake_acquire(letter, probes, start, end, **kw):
		order.append(('acquire', letter, tuple(probes), kw['download']))
		return _acquired(letter, probes, start, end,
		                 ['mms3_edp_brst_l2_dce_20210223100000_v1.0.0.cdf'] if letter == 'E' else ())

	monkeypatch.setattr(ee, 'acquire', fake_acquire)
	monkeypatch.setattr(ee, 'count_local_files', lambda l, p, s, e: order.append(('count', l, p)) or 1)
	cov = ee.check_coverage(RNG, ('B', 'E'), (1, 3))
	assert order[:2] == [('acquire', 'B', (1, 3), False), ('acquire', 'E', (1, 3), False)]
	assert [o[0] for o in order[2:]] == ['count'] * 4
	assert dict(cov) == {(1, 'B'): 1, (3, 'B'): 1, (1, 'E'): 1, (3, 'E'): 1}
	assert cov.refetch == {(3, 'E')}          # a corrupt file deleted: offered for download again


def test_download_with_progress_passes_cancel_through(tmp_path):
	seen = []
	stop = lambda: False
	dp.download_with_progress(RNG, [(1, 'E')], lambda *r: None, poll_s=0.01, data_dir=str(tmp_path),
	                          download=lambda rng, missing, cancelled=None: seen.append(cancelled),
	                          cancelled=stop)
	# a download without the keyword still works when nothing is passed
	dp.download_with_progress(RNG, [(1, 'E')], lambda *r: None, poll_s=0.01, data_dir=str(tmp_path),
	                          download=lambda rng, missing: seen.append('no keyword'))
	assert seen == [stop, 'no keyword']


# ---------------------------------------------------------------- plot_biggest_spikes

def test_the_spikes_driver_validates_what_is_present_and_acquires_what_is_missing(monkeypatch):
	calls = []

	def fake_acquire(letter, probes, start, end, **kw):
		calls.append((letter, tuple(probes), kw.get('download', True)))
		return _acquired(letter, probes, start, end,
		                 ['mms2_edp_brst_l2_dce_20210223110000_v1.0.0.cdf']
		                 if letter == 'E' and kw.get('download', True) is False else ())

	monkeypatch.setattr(pbs, 'acquire', fake_acquire)
	monkeypatch.setattr(pbs, 'probes', [1, 2])
	missing = pbs._validate_present(*W, [('fgm', '')])
	assert calls == [('B', (1, 2), False), ('E', (1, 2), False)]      # fgm is fetched, not re-checked
	assert missing == [('fgm', ''), ('edp', 'dce')]                   # the corrupt EDP file is fetched again

	calls.clear()
	pbs._download_request(*W, missing)
	assert calls == [('F', (1, 2), True), ('E', (1, 2), True)]


# ---------------------------------------------------------------- spike_biggest_burst_fft

def test_the_spike_script_acquires_up_front(monkeypatch):
	calls = []
	monkeypatch.setattr(sbf, 'acquire', lambda letter, probes, start, end, **kw:
	                    calls.append((letter, list(probes), start, end)))
	monkeypatch.setattr(sbf, 'FIELDS', ['B', 'E'])
	monkeypatch.setattr(sbf, 'PROBES', [1, 3])

	monkeypatch.setattr(sbf, 'FIXED_WINDOW', W)
	monkeypatch.setattr(sbf, 'CALC_FCE', True)
	sbf.acquire_inputs()
	assert calls == [('B', [1, 3], *W), ('E', [1, 3], *W), ('F', [1, 3], *W)]

	calls.clear()
	monkeypatch.setattr(sbf, 'FIXED_WINDOW', None)                    # the day the burst is picked from
	monkeypatch.setattr(sbf, 'CALC_FCE', False)
	sbf.acquire_inputs()
	day = (sbf.DAY, sbf.DAY + timedelta(days=1))
	assert calls == [('B', [1, 3], *day), ('E', [1, 3], *day)]

	src = inspect.getsource(sbf.main)
	assert src.index('acquire_inputs()') < src.index('find_biggest_burst(')


def test_the_spike_script_reads_without_downloading(monkeypatch):
	seen = {}

	def fake_range(probe, key, start, end, download=True):
		seen['discovery'] = download
		return []

	def fake_gen_fft(*a, **kw):
		seen['gen_fft'] = kw.get('download', True)
		raise FileNotFoundError

	monkeypatch.setattr(sbf, 'ensure_range_pickles', fake_range)
	monkeypatch.setattr(sbf, 'gen_fft', fake_gen_fft)
	with pytest.raises(FileNotFoundError):
		sbf.find_biggest_burst(1, 'Bz', sbf.DAY, sbf.field_spec('B'))
	assert sbf.stream_probe_matrix(1, 'Bz', *W, 10, 1e-6) is None
	assert seen == {'discovery': False, 'gen_fft': False}
	assert re.search(r'collect_fce\([^)]*download=False', inspect.getsource(sbf.main), re.S)


# ---------------------------------------------------------------- download.py

def test_download_py_acquires_day_by_day():
	"""download.py changes directory to E: when imported, so it is read, not run."""
	with open(os.path.join(ROOT, 'download.py')) as fh:
		src = fh.read()
	run = src[src.index("if __name__ == '__main__':"):]
	assert 'for day in days:\n\t\tacquire(key, probes, day, day + timedelta(days=1)' in run
	assert 'pool.map(_download_task, tasks)' not in run          # the six unpaced threads are gone
	assert run.index('acquire(key, probes') < run.index('check_cdfs(jobs')
