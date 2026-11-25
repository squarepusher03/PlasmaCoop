"""download_progress: progress of a running SDC download, read from the disk.

The downloader is replaced by a fake that writes CDF-named files into a temp
data_dir in pyspedas's layout, so nothing here touches the network."""
import os
import sys
import time
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import download_progress as dp
import event_export as ee


RNG = ee.RequestRange(datetime(2021, 2, 23, 10, 0), datetime(2021, 2, 23, 12, 0))   # 120 minutes


def _cdf(data_dir, letter, probe, stamp, size=1000):
	inst, mode = ee.INSTRUMENTS[letter]
	name = {'B': f'mms{probe}_scm_brst_l2_scb_{stamp}_v2.2.1.cdf',
	        'E': f'mms{probe}_edp_brst_l2_dce_{stamp}_v3.0.0.cdf',
	        'F': f'mms{probe}_fgm_brst_l2_{stamp}_v5.283.0.cdf'}[letter]
	parts = [str(data_dir), f'mms{probe}', inst, 'brst', 'l2'] + ([mode] if mode else [])
	d = os.path.join(*parts, stamp[:4], stamp[4:6], stamp[6:8])
	os.makedirs(d, exist_ok=True)
	path = os.path.join(d, name)
	with open(path, 'wb') as f:
		f.write(b'\xcd\xf3' + b'\0' * (size - 2))
	return path


def _fake_download(data_dir, landings, delay=0.03):
	"""A download_missing stand-in that lands (letter, probe, stamp) files one by one."""
	calls = []

	def download(rng, missing):
		calls.append((rng, tuple(missing)))
		for letter, probe, stamp in landings:
			time.sleep(delay)
			_cdf(data_dir, letter, probe, stamp)
	return download, calls


def _run(tmp_path, missing, landings, poll_s=0.01, delay=0.03):
	download, calls = _fake_download(tmp_path, landings, delay)
	reports = []
	final = dp.download_with_progress(RNG, missing, lambda *r: reports.append(r), poll_s=poll_s,
	                                  download=download, data_dir=str(tmp_path))
	return final, reports, calls


def test_poll_interval_is_three_to_five_seconds():
	assert 3 <= dp.POLL_S <= 5


def test_counts_only_files_that_land_and_ends_full(tmp_path):
	_cdf(tmp_path, 'E', 1, '20210223100500')                      # already on disk: not counted
	final, reports, calls = _run(tmp_path, [(1, 'E')],
	                             [('E', 1, '20210223103000'), ('E', 1, '20210223110000')])
	assert calls and calls[0][1] == ((1, 'E'),)
	assert final.done == final.total == 120
	assert 'Download finished: 2 file(s)' in final.text
	# Before the final report the newest landing is 11:00 -> 60 of 120 minutes.
	assert reports[-2][0] == 60 or max(r[0] for r in reports[:-1]) <= 60


def test_done_never_decreases_and_reports_keep_coming(tmp_path):
	_final, reports, _ = _run(tmp_path, [(1, 'B'), (2, 'B')],
	                          [('B', 1, '20210223101000'), ('B', 1, '20210223113000'),
	                           ('B', 2, '20210223100000'), ('B', 2, '20210223114500')], delay=0.05)
	dones = [r[0] for r in reports]
	assert dones == sorted(dones)
	assert len(reports) >= 5                                      # first, several polls, final
	assert all(r[1] == 240 for r in reports)


def test_first_report_is_immediate_and_at_zero(tmp_path):
	_final, reports, _ = _run(tmp_path, [(1, 'E')], [('E', 1, '20210223103000')])
	assert reports[0][0] == 0
	assert 'waiting for the first file' in reports[0][2]


def test_stamps_outside_the_range_are_clamped(tmp_path):
	watcher = dp.LandingWatcher(RNG, [(1, 'E')], data_dir=str(tmp_path))
	_cdf(tmp_path, 'E', 1, '20210223095000')                      # pyspedas fetches a little before start
	assert watcher.poll().done == 0
	_cdf(tmp_path, 'E', 1, '20210223130000')                      # past the end
	assert watcher.poll().done == 120


def test_a_later_pair_landing_completes_earlier_pairs(tmp_path):
	watcher = dp.LandingWatcher(RNG, [(2, 'B'), (1, 'B'), (1, 'E')], data_dir=str(tmp_path))
	assert watcher.pairs == [('B', 1), ('B', 2), ('E', 1)]
	_cdf(tmp_path, 'B', 1, '20210223103000')
	assert watcher.poll().done == 30
	_cdf(tmp_path, 'E', 1, '20210223101500')                      # E starts: both B probes are done
	p = watcher.poll()
	assert p.done == 120 + 120 + 15
	assert 'MMS1 E (EDP) through 2021-02-23 10:15 UTC' in p.text


def test_fgm_has_no_mode_directory(tmp_path):
	watcher = dp.LandingWatcher(RNG, [(3, 'F')], data_dir=str(tmp_path))
	path = _cdf(tmp_path, 'F', 3, '20210223110000')
	assert os.sep.join(['mms3', 'fgm', 'brst', 'l2', '2021']) in path
	assert watcher.poll().done == 60


def test_status_text_reports_files_mb_and_elapsed(tmp_path):
	clock = iter([0.0, 75.0]).__next__
	watcher = dp.LandingWatcher(RNG, [(1, 'E')], data_dir=str(tmp_path), clock=clock)
	_cdf(tmp_path, 'E', 1, '20210223103000', size=2_500_000)
	text = watcher.poll().text
	assert '1 file(s), 2.5 MB' in text
	assert '1:15 elapsed' in text


def test_only_day_directories_in_range_are_scanned(tmp_path):
	watcher = dp.LandingWatcher(RNG, [(1, 'B')], data_dir=str(tmp_path))
	days = [d.split(os.sep)[-3:] for d in watcher._dirs[('B', 1)]]
	assert days == [['2021', '02', '22'], ['2021', '02', '23']]
	_cdf(tmp_path, 'B', 1, '20210301103000')                      # another day entirely: never seen
	assert watcher.poll().done == 0


def test_a_failing_download_raises(tmp_path):
	def boom(rng, missing):
		raise RuntimeError('SDC unreachable')
	with pytest.raises(RuntimeError, match='SDC unreachable'):
		dp.download_with_progress(RNG, [(1, 'E')], lambda *r: None, poll_s=0.01,
		                          download=boom, data_dir=str(tmp_path))
