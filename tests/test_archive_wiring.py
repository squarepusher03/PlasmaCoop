"""The preprocessed archive wired into the pipeline (2026-11-03): once a CDF is archived and deleted, everything the
pipeline reads from it comes out the same -- discovery, bounds, good runs, sample times, FFTs (seams included),
validation, cache staleness -- and acquisition archives, verifies per request and leaves nothing behind. With the
archive off nothing changes and no database is made.

Synthetic CDFs at the real paths and names (test_direct_cdf.write_cdf), in the test's own folder.
"""
import os
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('spacepy.pycdf')

import acquisition
import cdaweb
import ffttest
import preproc_archive as pa
from ffttest import SAMPLE_RATE
from test_direct_cdf import INSTRUMENTS, T, write_cdf


@pytest.fixture(autouse=True)
def _fresh_archive_state():
	pa._checked_versions.clear()
	pa._checked.clear()
	ffttest._CHECKED_CDFS.clear()
	ffttest._scm_good_segments.cache_clear()
	ffttest._edp_good_segments.cache_clear()
	yield
	pa._checked_versions.clear()
	pa._checked.clear()
	ffttest._CHECKED_CDFS.clear()
	ffttest._scm_good_segments.cache_clear()
	ffttest._edp_good_segments.cache_clear()


def _pair(data, key, bad=()):
	"""One recording split into two files at 4 s, so windows borrow across the seam both ways."""
	curr = write_cdf(data, key, '20210223100000', T(0), 32800, bad=bad)
	nxt = write_cdf(data, key, '20210223100004', T(0), 32736, first=32800)
	return curr, nxt


def _ffts(key, curr, nxt):
	inst, mode = INSTRUMENTS[key]
	args = (key + 'z', inst, mode, SAMPLE_RATE, 0.25, int(SAMPLE_RATE * 0.5), 2.0)
	return (ffttest._compute_file_ffts(curr, nxt, *args, T(0), T(4)),
	        ffttest._compute_file_ffts(nxt, None, *args, T(4), None, prev_path=curr, prev_file_time=T(0)))


def _readers(key, path):
	segs = ffttest._scm_good_segments if key == 'B' else ffttest._edp_good_segments
	return (ffttest.cdf_time_bounds(path, key=key, probe=1), ffttest._good_sample_times(path, key, 1),
	        ffttest._recorded_sample_times(path, key, 1), segs.__wrapped__(path, 1),
	        ffttest._segments_from_cdf(path, key, SAMPLE_RATE, 1))


def _same(a, b):
	if isinstance(a, np.ndarray):
		return np.array_equal(a, b)
	if isinstance(a, tuple):
		return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
	return a == b


@pytest.mark.parametrize('key, bad', [('B', [slice(0, 3000)]), ('E', [slice(5000, 5001), slice(9000, 9400)])])
def test_an_archived_cdf_reads_exactly_as_the_cdf_did(tmp_path, monkeypatch, key, bad):
	data = str(tmp_path / 'pydata')
	curr, nxt = _pair(data, key, bad)
	inst, mode = INSTRUMENTS[key]
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'off')
	want_scan = ffttest._scan_cdf_files(data, inst, mode, 1)
	want_ffts = _ffts(key, curr, nxt)
	want = [_readers(key, p) for p in (curr, nxt)]
	want_mtime = [os.path.getmtime(p) for p in (curr, nxt)]
	assert len(want_ffts[0]) and len(want_ffts[1])

	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'on')                 # archive, and delete each CDF
	out = pa.archive_cdfs([curr, nxt], key, log=lambda *_: None)
	assert (out['archived'], out['deleted']) == (2, 2) and not os.path.exists(curr) and not os.path.exists(nxt)

	assert ffttest._scan_cdf_files(data, inst, mode, 1) == want_scan      # same stamps, same paths
	for got, exp in zip(_ffts(key, curr, nxt), want_ffts):
		pd.testing.assert_frame_equal(got, exp)                     # bit for bit, both seams
	for p, exp in zip((curr, nxt), want):
		assert _same(_readers(key, p), exp)
	assert [ffttest._source_mtime(p) for p in (curr, nxt)] == pytest.approx(want_mtime, abs=1e-6)
	assert ffttest.check_cdfs([(p, key, 1) for p in (curr, nxt)]) == [curr, nxt]
	assert ffttest.cdf_check(key, 1)([curr, nxt]) == [curr, nxt]
	assert ffttest.find_local_cdfs(data, inst, mode, T(1), T(6), probe=1) == want_scan


def test_archiving_stores_in_batches_and_deletes_only_behind_verified_rows(tmp_path, monkeypatch):
	"""Five CDFs, batches of two: three flushes (2, 2, 1). A row whose drive copy differs keeps its own CDF; the
	other four are archived and deleted."""
	import drive_read
	data = str(tmp_path / 'pydata')
	paths = [write_cdf(data, 'B', f'2021022310{m:02d}00', T(60 * m), 4096) for m in range(5)]
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'on')
	monkeypatch.setattr(pa, 'STORE_BATCH_ROWS', 2)
	sizes, real_store, real_rows = [], pa._store_rows, drive_read.rows
	bad = os.path.basename(paths[3])

	def store(db, rows):
		sizes.append(len(rows))
		return real_store(db, rows)

	def rows(db, root, rowids):
		out = real_rows(db, root, rowids)
		with closing(sqlite3.connect(db)) as conn:
			hit = conn.execute(f'SELECT rowid FROM {pa.table()} WHERE name = ?', (bad,)).fetchone()
		if hit and hit[0] in out:
			values = out[hit[0]]
			i = [type(v) for v in values].index(bytes)
			values[i] = values[i][:-1] + bytes([values[i][-1] ^ 1])
		return out

	monkeypatch.setattr(pa, '_store_rows', store)
	monkeypatch.setattr(drive_read, 'rows', rows)
	out = pa.archive_cdfs(paths, 'B', log=lambda *_: None, max_workers=2)
	assert sorted(sizes) == [1, 2, 2]
	assert (out['archived'], out['deleted'], out['failed']) == (4, 4, 1)
	assert [os.path.exists(p) for p in paths] == [False, False, False, True, False]
	assert [pa.current(p) is not None for p in paths] == [True, True, True, False, True]


def test_prewarm_from_the_archive_writes_the_pickles_the_cdfs_gave(tmp_path, monkeypatch):
	data = str(tmp_path / 'pydata')
	curr, nxt = _pair(data, 'B')
	spec = [('Bz', 1, 0.25, 3000, [(T(0), T(8), False)])]
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'off')
	ffttest.prewarm(data, str(tmp_path / 'cache_cdf'), spec, write_manifest=False, max_workers=2)
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'on')
	pa.archive_cdfs([curr, nxt], 'B', log=lambda *_: None)
	ffttest.prewarm(data, str(tmp_path / 'cache_arch'), spec, write_manifest=False, max_workers=2)
	for ft in (T(0), T(4)):
		a = pd.read_pickle(ffttest._fft_pickle_path(str(tmp_path / 'cache_cdf'), 'scm', 'scb', 0.25, ft, probe=1))
		b = pd.read_pickle(ffttest._fft_pickle_path(str(tmp_path / 'cache_arch'), 'scm', 'scb', 0.25, ft, probe=1))
		pd.testing.assert_frame_equal(a, b)


def test_archiving_a_cdf_does_not_make_its_fft_pickle_stale(tmp_path, monkeypatch):
	"""The row keeps the CDF's mtime, so a cache built from the CDF stays good once the CDF is gone."""
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	curr, nxt = _pair(data, 'B')
	spec = [('Bz', 1, 0.25, 3000, [(T(0), T(8), False)])]
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'off')
	ffttest.prewarm(data, cache, spec, write_manifest=False, max_workers=2)
	pkl = ffttest._fft_pickle_path(cache, 'scm', 'scb', 0.25, T(0), probe=1)
	before = os.path.getmtime(pkl)
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'on')
	pa.archive_cdfs([curr, nxt], 'B', log=lambda *_: None)
	ffttest._invalidate_if_stale(pkl, ffttest._read_version(pkl), source_paths=(curr, nxt))
	assert os.path.exists(pkl) and os.path.getmtime(pkl) == before


def test_acquisition_archives_verifies_and_leaves_no_cdf(tmp_path, monkeypatch):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	curr, nxt = _pair(data, 'E')
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'on')
	logs = []
	got = acquisition.acquire('E', [1], T(0), T(8), data_dir=data, pickle_root=cache, download=False, log=logs.append)
	assert got.files[1] == [curr, nxt]                              # listed at their paths, from the archive
	assert not os.path.exists(curr) and not os.path.exists(nxt)
	assert pa.current(curr) is not None and pa.current(nxt) is not None
	assert any('Archived 2 CDF(s)' in line for line in logs)
	# A second acquisition: the archived files pass validation. The per-request check is off by default (Graham,
	# 2026-11-03: "Turn per request verify off"), so nothing is read from the drive ...
	pa._checked.clear()
	logs.clear()
	again = acquisition.acquire('E', [1], T(0), T(8), data_dir=data, pickle_root=cache, download=False,
	                            log=logs.append)
	assert again.files[1] == [curr, nxt] and not again.unresolved
	assert not any('Request check' in line for line in logs)
	# ... and with it on, the request check finds them intact.
	import app_config
	app_config.update('archive', {'verify_each_request': True})
	logs.clear()
	again = acquisition.acquire('E', [1], T(0), T(8), data_dir=data, pickle_root=cache, download=False,
	                            log=logs.append)
	assert again.files[1] == [curr, nxt] and not again.unresolved
	assert any('Request check: 2 intact' in line for line in logs)


def test_a_corrupt_archived_row_is_dropped_and_reported_when_it_cannot_be_fetched(tmp_path, monkeypatch):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	curr, nxt = _pair(data, 'B')
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'on')
	import app_config
	app_config.update('archive', {'verify_each_request': True})       # the per-request check (off by default)
	acquisition.acquire('B', [1], T(0), T(8), data_dir=data, pickle_root=cache, download=False, log=lambda *_: None)
	db = pa.db_path(data)
	with closing(sqlite3.connect(db)) as conn:                       # damage one row on the drive too
		blob = conn.execute(f'SELECT signal_blob FROM {pa.table()} WHERE name = ?',
		                    (os.path.basename(nxt),)).fetchone()[0]
		conn.execute(f'UPDATE {pa.table()} SET signal_blob = ? WHERE name = ?',
		             (blob[:-1] + bytes([blob[-1] ^ 1]), os.path.basename(nxt)))
		conn.commit()
		conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
	pa._checked.clear()                                             # a later session
	got = acquisition.acquire('B', [1], T(0), T(8), data_dir=data, pickle_root=cache, download=False,
	                          log=lambda *_: None)
	assert pa.current(nxt) is None                                  # deleted from the archive
	assert os.path.basename(nxt) in got.unresolved                  # download=False: reported missing
	assert got.files[1] == [curr]                                   # the request never uses the bad row


def test_with_the_archive_off_acquisition_keeps_every_cdf_and_makes_no_database(tmp_path, monkeypatch):
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	curr, nxt = _pair(data, 'B')
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'off')
	got = acquisition.acquire('B', [1], T(0), T(8), data_dir=data, pickle_root=cache, download=False,
	                          log=lambda *_: None)
	assert got.files[1] == [curr, nxt] and os.path.exists(curr) and os.path.exists(nxt)
	assert not os.path.exists(pa.db_path(data))


def test_cdaweb_skips_a_file_the_archive_holds_at_its_listed_size(tmp_path, monkeypatch):
	data = str(tmp_path / 'pydata')
	curr, nxt = _pair(data, 'B')
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'on')
	pa.archive_cdfs([curr, nxt], 'B', log=lambda *_: None)
	held = cdaweb._archive_held(data, 'scm', 'scb', 1)
	assert set(held) == {os.path.basename(curr), os.path.basename(nxt)}
	listed = [cdaweb.Listed(cdaweb._stamp(n), n, 'x/' + n, size) for n, size in held.items()]
	listed.append(cdaweb.Listed(cdaweb._stamp(os.path.basename(nxt)), os.path.basename(nxt).replace('v1.0.0', 'v1.0.1'),
	                            'x/new', 123))
	fetched = []
	monkeypatch.setattr(cdaweb, 'session_factory', lambda: object())
	monkeypatch.setattr(cdaweb, 'list_files', lambda session, ds, first, last: sorted(listed))
	monkeypatch.setattr(cdaweb, '_one', lambda session, f, dest, stop, *_cancelled: (fetched.append(f.name), ('ok', 0))[1])
	cdaweb.download('scm', 'scb', (1,), T(0), T(8), data, concurrent=1, log=lambda *_: None)
	assert fetched == [os.path.basename(nxt).replace('v1.0.0', 'v1.0.1')]   # only the file the archive lacks
