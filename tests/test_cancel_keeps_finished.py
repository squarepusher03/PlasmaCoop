"""What a cancel deletes (Graham, 2026-11-24: "if they cancel midway through a request is being run, the only files
that should be deleted are ones being currently downloaded and output artifacts currently being generated, the rest
should persist"): a download under way stops and its .part goes, finished downloads stay; the event or event image
being written is deleted (its .partial files), finished ones and an event's older copy stay. Synthetic, tmp_path.

    python -m pytest -q tests/test_cancel_keeps_finished.py
"""
import os
import sys
import threading
import time
from datetime import datetime

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import cdaweb
import event_export as ee
from test_cdaweb import DAY, Resp, body, names, on_disk, server_for, use
from test_export_outputs import EVENT, binned, files_of, read, settings  # noqa: F401  (binned is a fixture)


@pytest.fixture(autouse=True)
def empty_registries(monkeypatch):
	monkeypatch.setattr(cdaweb, '_downloading', set())
	monkeypatch.setattr(ee, '_writing', set())


# ------------------------------------------------------------------------------------------------ downloads

def test_a_cancel_stops_the_downloads_under_way_deletes_their_parts_and_keeps_the_finished(tmp_path, monkeypatch):
	ns = names(1, 'dce', DAY[1:5])
	slow = {n: (lambda n_: (lambda _k: Resp(content=body(n_, 60_000), delay=0.01)))(n) for n in ns[1:]}
	server = server_for({'MMS1_EDP_BRST_L2_DCE': ns}, behaviour=slow)
	for n in ns[1:]:                                             # listed at the slow bodies' size
		server.files['MMS1_EDP_BRST_L2_DCE'][n] = body(n, 60_000)
	use(monkeypatch, server)
	flag = threading.Event()
	threading.Timer(0.15, flag.set).start()
	t0 = time.monotonic()
	status = cdaweb.download('edp', 'dce', (1,), datetime(2019, 8, 16, 7, 39), datetime(2019, 8, 16, 7, 51),
	                         str(tmp_path), concurrent=2, cancelled=flag.is_set, log=lambda *a: None)
	took = time.monotonic() - t0
	assert status == cdaweb.FAILED
	assert on_disk(tmp_path) == {f'mms1/edp/brst/l2/dce/2019/08/16/{ns[0]}'}   # finished: kept; no .part anywhere
	assert (tmp_path / 'mms1/edp/brst/l2/dce/2019/08/16' / ns[0]).read_bytes() == body(ns[0])
	assert ns[3] not in server.gets                              # nothing new started after the cancel
	assert took < 0.6 * 0.6 + 0.5                                # stopped part way, well before 60 blocks of 0.01 s
	assert cdaweb._downloading == set()


def test_a_429_still_lets_the_files_under_way_finish(tmp_path, monkeypatch):
	ns = names(1, 'dce', DAY[1:4])
	server = server_for({'MMS1_EDP_BRST_L2_DCE': ns}, delay=0.01,
	                    behaviour={ns[1]: Resp(429, headers={'Retry-After': '300'})})
	use(monkeypatch, server)
	status = cdaweb.download('edp', 'dce', (1,), datetime(2019, 8, 16, 7, 39), datetime(2019, 8, 16, 7, 46),
	                         str(tmp_path), concurrent=2, log=lambda *a: None)
	assert status == cdaweb.RATE_LIMITED
	assert f'mms1/edp/brst/l2/dce/2019/08/16/{ns[0]}' in on_disk(tmp_path)      # the one under way finished
	assert not any(p.endswith('.part') for p in on_disk(tmp_path))


def test_force_quit_deletes_the_parts_still_listed(tmp_path):
	part = tmp_path / 'x.cdf.part'
	part.write_bytes(b'half')
	cdaweb._downloading.add(str(part))
	assert cdaweb.discard_partial_downloads() == []
	assert not part.exists()


def test_a_stale_part_is_swept_from_the_folder_a_download_uses(tmp_path, monkeypatch):
	ns = names(1, 'scb', DAY[1:2])
	use(monkeypatch, server_for({'MMS1_SCM_BRST_L2_SCB': ns}))
	folder = tmp_path / 'mms1/scm/brst/l2/scb/2019/08/16'
	folder.mkdir(parents=True)
	(folder / 'left_by_a_force_quit.cdf.part').write_bytes(b'half')
	cdaweb.download('scm', 'scb', (1,), datetime(2019, 8, 16, 7, 39), datetime(2019, 8, 16, 7, 40),
	                str(tmp_path), log=lambda *a: None)
	assert on_disk(tmp_path) == {f'mms1/scm/brst/l2/scb/2019/08/16/{ns[0]}'}


# ------------------------------------------------------------------------------------------------ outputs

def all_files(root):
	return sorted(os.path.relpath(os.path.join(d, n), root) for d, _s, ns in os.walk(root) for n in ns)


def test_an_uncancelled_event_is_written_as_before_with_no_partial_left(tmp_path, binned):
	s = settings(tmp_path)
	assert ee.export_event(s, EVENT, log=lambda *_: None, cancelled=lambda: False) == ee.WRITTEN
	assert files_of(s) == [ee.CSV_NAME, ee.PNG_NAME]
	direct = tmp_path / 'direct'
	direct.mkdir()
	ee.write_event_csv(str(direct / 'a.csv'), EVENT.start, binned[1], binned[0], 'B', binned[4], s.layout)
	assert read(s, ee.CSV_NAME) == (direct / 'a.csv').read_bytes()        # the .partial route changes no byte
	assert ee._writing == set()


@pytest.mark.parametrize('after', [1, 2])                    # cancel after the CSV, or after the graph too
def test_an_event_cancelled_while_written_is_deleted(tmp_path, binned, after):
	s = settings(tmp_path)
	calls = {'n': 0}

	def cancelled():
		calls['n'] += 1
		return calls['n'] >= after
	assert ee.export_event(s, EVENT, log=lambda *_: None, cancelled=cancelled) == ee.DISCARDED
	assert files_of(s) == []                                     # nothing of it, no .partial
	assert ee._writing == set()


def test_a_cancelled_rerun_leaves_the_events_old_files_as_they_were(tmp_path, binned):
	s = settings(tmp_path)
	assert ee.export_event(s, EVENT, log=lambda *_: None) == ee.WRITTEN
	folder = ee.event_dir(s.top_dir, EVENT)
	for name in (ee.CSV_NAME, ee.PNG_NAME):
		with open(os.path.join(folder, name), 'wb') as fh:
			fh.write(b'old ' + name.encode())
	assert ee.export_event(s, EVENT, log=lambda *_: None, cancelled=lambda: True) == ee.DISCARDED
	assert read(s, ee.CSV_NAME) == b'old ffts.csv' and read(s, ee.PNG_NAME) == b'old graph.png'
	assert files_of(s) == [ee.CSV_NAME, ee.PNG_NAME]


def test_without_cancelled_an_event_is_always_finished(tmp_path, binned):
	s = settings(tmp_path)
	assert ee.export_event(s, EVENT, log=lambda *_: None) == ee.WRITTEN   # the census's call


def test_a_failing_write_leaves_no_partial(tmp_path, binned, monkeypatch):
	s = settings(tmp_path)

	def boom(*a, **k):
		raise RuntimeError('draw failed')
	monkeypatch.setattr(ee, 'draw_event_png', boom)
	with pytest.raises(RuntimeError):
		ee.export_event(s, EVENT, log=lambda *_: None)
	assert files_of(s) == [] and ee._writing == set()


def test_force_quit_deletes_listed_partials_and_the_sweep_takes_stale_ones(tmp_path):
	listed = tmp_path / 'ffts.csv.partial'
	listed.write_bytes(b'half')
	ee._writing.add(str(listed))
	assert ee.discard_partial_outputs(wait_s=0.1) == []
	assert not listed.exists()
	stale = tmp_path / 'MMS1' / 'B' / 'graph.png.partial'
	stale.parent.mkdir(parents=True)
	stale.write_bytes(b'half')
	kept = tmp_path / 'MMS1' / 'B' / 'graph.png'
	kept.write_bytes(b'whole')
	assert ee.sweep_partials(str(tmp_path)) == 1
	assert not stale.exists() and kept.read_bytes() == b'whole'


def test_force_quit_waits_for_a_commit_under_way(tmp_path):
	ee._commit_lock.acquire()
	t0 = time.monotonic()
	threading.Timer(0.3, ee._commit_lock.release).start()
	ee.discard_partial_outputs(wait_s=5)
	assert 0.25 <= time.monotonic() - t0 < 3


def test_an_event_image_cancelled_while_drawn_is_deleted(tmp_path, monkeypatch):
	import event_images as ei

	class Plan:
		field, name, kind = 'B', 'img.png', 'long'
	drawn = []

	def fake_draw(path, *a, **k):
		with open(path, 'wb') as fh:
			fh.write(b'png')
		drawn.append(path)
		return path
	monkeypatch.setattr(ei, 'draw_image', fake_draw)
	s = settings(tmp_path)
	def cancelled():
		return bool(drawn)                           # the cancel comes while the first image is drawn
	done = ei.write_event_images(s, [Plan(), Plan()], (1,), cancelled=cancelled, log=lambda *_: None)
	assert done == []                                # the one drawn when the cancel came is not counted ...
	assert all_files(s.top_dir) == []                # ... nor kept
	assert drawn and drawn[0].endswith('img.png.partial')
	done = ei.write_event_images(s, [Plan()], (1,), log=lambda *_: None)
	assert [p for _pl, p in done] == [os.path.join(s.top_dir, ei.IMAGES_DIR, 'B', 'img.png')]
	assert all_files(s.top_dir) == [os.path.join(ei.IMAGES_DIR, 'B', 'img.png')]
