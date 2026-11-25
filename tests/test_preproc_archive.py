"""preproc_archive and drive_read, on synthetic rows in the test's own data folder (never E:).

Covers the codec round trip, the drive reader against SQL, the one-time drive check before a row is verified, reads
that never check again (Graham, 2026-11-03), the optional launch pass (verify_archive) and its setting, the version
and schema guards, the covering index, the filters, and that nothing opens a database while the archive is off. preprocess() itself needs a real CDF: its
proof against the live readers is _wsl_scratch/preproc_reuse_proof.py.
"""
import os
import random
import sqlite3
import time
import zlib
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import drive_read
import ffttest
import preproc_archive as pa

T0 = int(pd.Timestamp('2016-02-01 00:00:00').value)


@pytest.fixture
def archive_on(monkeypatch):
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'keep')
	pa._checked_versions.clear()
	pa._checked.clear()
	yield
	pa._checked_versions.clear()
	pa._checked.clear()


def _b_path(data_dir, stamp='20160201000000', probe=1):
	inst, mode, _fs = ffttest.field_burst('B')
	name = f'mms{probe}_{inst}_brst_l2_{mode}_{stamp}_v2.2.0.cdf'
	t = ffttest._burst_cdf_name(name)[3]
	return os.path.join(str(data_dir), f'mms{probe}', inst, 'brst', 'l2', mode, f'{t:%Y}', f'{t:%m}', f'{t:%d}', name)


def _archived(n=60_000, start_ns=T0, seed=1):
	rng = np.random.default_rng(seed)
	recorded = start_ns + (np.arange(n, dtype=np.int64) * 122_072 // 1000) * 1000      # MMS times: whole microseconds
	good = rng.random(n) > 0.05
	signal = rng.standard_normal(int(good.sum())).astype(np.float32)
	return pa.Archived(recorded, good, signal)


def _row(path, pre, size=15_000_000, mtime_ns=1_700_000_000_000_000_000):
	return pa._row_of(path, pre, SimpleNamespace(st_size=size, st_mtime_ns=mtime_ns))


def _stored(data_dir, **kw):
	path = _b_path(data_dir, **kw)
	pre = _archived(start_ns=int(pd.Timestamp(ffttest._burst_cdf_name(os.path.basename(path))[3]).value))
	db = pa.db_path(str(data_dir))
	assert pa._store(db, _row(path, pre))
	return path, pre, db


# --- codec ------------------------------------------------------------------------------------------------------

def test_codec_round_trip_is_exact():
	pre = _archived()
	back = pa._decode(pa.encode_time(pre.recorded), pa.encode_good(pre.good), pa.encode_signal(pre.signal))
	assert np.array_equal(back.recorded, pre.recorded)
	assert np.array_equal(back.good, pre.good)
	assert np.array_equal(back.signal.view(np.uint32), pre.signal.view(np.uint32))
	assert pa.decode_time(pa.encode_time(np.empty(0, np.int64))).size == 0


@pytest.mark.parametrize('steps_us', [
	[122, 123] * 500,                                        # MMS: every step fits a byte, nothing escaped
	[122] * 10 + [254, 255, 256] + [123] * 10,               # the last byte value, the escape value, one over
	[122] * 5 + [4_000_000, 122, 2 ** 40, 0, 0, 122],        # gaps of seconds and days; repeated times
	[122, -5, 122, -(2 ** 33), 122],                         # times going back: stored in full as well
	[], [255], [0],                                          # one record; two records
])
def test_time_codec_v2_round_trips_every_kind_of_step(steps_us):
	"""v2 (Graham, 2026-11-03: "Try the step resize"): one byte a step, the rest kept in full."""
	t = (T0 // 1000 + np.concatenate([[0], np.cumsum(np.array(steps_us, np.int64))])) * 1000
	blob = pa.encode_time(t)
	assert np.array_equal(pa.decode_time(blob), t)
	assert pa.decode_time(blob).dtype == np.int64


def test_time_codec_v2_is_smaller_than_v1_and_a_damaged_escape_count_raises():
	t = _archived().recorded
	blob = pa.encode_time(t)
	v1 = np.int64(t[0] // 1000).tobytes() + zlib.compress(pa._shuffle(np.diff(t // 1000)), 6)
	assert len(blob) < len(v1)
	raw = bytearray(zlib.decompress(blob[8:]))
	raw[0] += 1                                              # one more escape than the steps mark
	with pytest.raises(ValueError):
		pa.decode_time(blob[:8] + zlib.compress(bytes(raw)))


def test_a_v1_time_blob_still_decodes_as_v1():
	"""Tables of older versions stay readable by name (query, read_interval)."""
	t = _archived().recorded
	v1 = np.int64(t[0] // 1000).tobytes() + zlib.compress(pa._shuffle(np.diff(t // 1000)), 6)
	assert np.array_equal(pa.decode_time(v1, version=1), t)
	assert np.array_equal(pa._decode(v1, pa.encode_good(np.ones(t.size, bool)), version=1).recorded, t)


def test_codec_refuses_what_it_cannot_store_exactly():
	with pytest.raises(ValueError):
		pa.encode_time(np.array([T0, T0 + 1], np.int64))                 # not whole microseconds
	with pytest.raises(ValueError):
		pa.encode_signal(np.zeros(4, np.float64))                       # Graham: not 64 bit


# --- drive_read against SQL -------------------------------------------------------------------------------------

@pytest.mark.parametrize('page_size,journal', [(512, 'DELETE'), (4096, 'TRUNCATE'), (65536, 'WAL')])
def test_drive_read_matches_sql_on_every_row(tmp_path, page_size, journal):
	"""Rows of every size (inline, overflowing, many overflow pages), a third deleted and a third replaced so later
	rows reuse freed pages, every row read both ways."""
	path = str(tmp_path / f'walk_{page_size}.sqlite')
	rng = random.Random(page_size)
	with closing(sqlite3.connect(path)) as conn:
		conn.execute(f'PRAGMA page_size = {page_size}')
		conn.execute(f'PRAGMA journal_mode = {journal}')
		conn.execute('CREATE TABLE t (name TEXT PRIMARY KEY, a INTEGER, c REAL, g BLOB)')
		for i in range(300):
			conn.execute('INSERT INTO t VALUES (?,?,?,?)', (f'row é {i}', rng.choice([0, -1, 1 << 40, None]),
			                                                rng.random(), rng.randbytes(rng.randrange(6 * page_size))))
		conn.commit()
		names = [r[0] for r in conn.execute('SELECT name FROM t')]
		rng.shuffle(names)
		conn.executemany('DELETE FROM t WHERE name = ?', [(n,) for n in names[:100]])
		for n in names[100:200]:
			conn.execute('INSERT OR REPLACE INTO t VALUES (?,?,?,?)', (n, 7, 0.5, rng.randbytes(rng.randrange(9000))))
		conn.commit()
		if journal == 'WAL':
			assert conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] == 0
		root = drive_read.root_page(conn, 't')
		rows = conn.execute('SELECT rowid, * FROM t').fetchall()
		assert len(rows) == 200
		for rowid, *want in rows:
			assert drive_read.row(path, root, rowid) == want
		with pytest.raises(KeyError):
			drive_read.row(path, root, 10 ** 9)
		many = drive_read.rows(path, root, [r[0] for r in rows] + [10 ** 9])     # one handle, the same rows
		assert {rowid: v for rowid, v in many.items() if rowid != 10 ** 9} == {r[0]: list(r[1:]) for r in rows}
		assert isinstance(many[10 ** 9], KeyError)


# --- writing: the drive check -----------------------------------------------------------------------------------

def test_store_verifies_against_the_drive_and_load_gives_the_arrays_back(tmp_path, archive_on):
	path, pre, db = _stored(tmp_path)
	with closing(sqlite3.connect(db)) as conn:
		assert conn.execute(f'SELECT verified FROM {pa.table()}').fetchall() == [(1,)]
	meta = pa.current(path)                                 # the CDF is not on disk: the row stands for it
	assert meta['name'] == os.path.basename(path) and meta['good_first_ns'] == int(pre.times[0])
	got = pa.load(path)
	assert np.array_equal(got.recorded, pre.recorded) and np.array_equal(got.good, pre.good)
	assert np.array_equal(got.signal.view(np.uint32), pre.signal.view(np.uint32))
	assert pa.load(path, signal=False).signal is None


def _one_byte_off(real, bad_rowids):
	"""drive_read.rows, with one byte of a blob flipped in the rows named."""
	def rows(db, root, rowids):
		out = real(db, root, rowids)
		for rowid in bad_rowids:
			values = out[rowid]
			i = [type(v) for v in values].index(bytes)
			b = bytearray(values[i])
			b[-1] ^= 1
			values[i] = bytes(b)
		return out
	return rows


def test_store_leaves_the_row_unverified_when_the_drive_copy_differs(tmp_path, archive_on, monkeypatch):
	real = drive_read.rows
	monkeypatch.setattr(drive_read, 'rows', _one_byte_off(real, [1]))
	path = _b_path(tmp_path)
	db = pa.db_path(str(tmp_path))
	assert not pa._store(db, _row(path, _archived()))
	monkeypatch.setattr(drive_read, 'rows', real)
	with closing(sqlite3.connect(db)) as conn:
		assert conn.execute(f'SELECT verified FROM {pa.table()}').fetchall() == [(0,)]
	assert pa.current(path) is None                         # unverified rows are never read
	assert not pa._checked


def test_a_batch_verifies_every_row_but_the_one_whose_drive_copy_differs(tmp_path, archive_on, monkeypatch):
	"""One commit, one checkpoint and one drive handle for the batch, but each row compared by itself: a mismatch
	leaves only its own row unverified (and so only its own CDF on disk)."""
	stamps = ['20160201000000', '20160201010000', '20160201020000']
	paths = [_b_path(tmp_path, stamp=s) for s in stamps]
	rows = [_row(p, _archived(seed=i, start_ns=int(pd.Timestamp(ffttest._burst_cdf_name(os.path.basename(p))[3]).value)))
	        for i, p in enumerate(paths)]
	db = pa.db_path(str(tmp_path))
	calls = []
	real = drive_read.rows

	def spy(db_, root, rowids):
		calls.append(list(rowids))
		return _one_byte_off(real, [2])(db_, root, rowids)    # rowid 2: the second row inserted

	monkeypatch.setattr(drive_read, 'rows', spy)
	assert pa._store_rows(db, rows) == [True, False, True]
	assert calls == [[1, 2, 3]]                             # every row of the batch through one handle
	monkeypatch.setattr(drive_read, 'rows', real)
	with closing(sqlite3.connect(db)) as conn:
		assert conn.execute(f'SELECT name, verified FROM {pa.table()} ORDER BY rowid').fetchall() == [
			(os.path.basename(p), v) for p, v in zip(paths, (1, 0, 1))]
	assert [pa.current(p) is not None for p in paths] == [True, False, True]
	assert sorted(k[2] for k in pa._checked) == [1, 3]       # the verified rows count as checked this session
	assert pa._store_rows(db, []) == []


# --- later checks: per request, the background scan, their precedence ----------------------------------------------

def _tamper(db, name, column='signal_blob'):
	"""Flip one bit of a stored row and checkpoint it, so the drive holds the damage too."""
	with closing(sqlite3.connect(db)) as conn:
		blob = conn.execute(f'SELECT {column} FROM {pa.table()} WHERE name = ?', (name,)).fetchone()[0]
		conn.execute(f'UPDATE {pa.table()} SET {column} = ? WHERE name = ?', (blob[:-1] + bytes([blob[-1] ^ 1]), name))
		conn.commit()
		assert conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] == 0


def test_reads_take_a_verified_row_as_it_is(tmp_path, archive_on, monkeypatch):
	"""Ordinary reads (load, current, scan) do no digest work and never touch the drive reader: the checks are
	verify_request and the background scan, which run before."""
	path, _pre, db = _stored(tmp_path)
	with closing(sqlite3.connect(db)) as conn:               # metadata damage: no digest check notices it on a read
		conn.execute(f'UPDATE {pa.table()} SET cdf_size = cdf_size + 1')
		conn.commit()

	def no_drive(*a, **k):
		raise AssertionError('a read went to the drive')

	monkeypatch.setattr(drive_read, 'row', no_drive)
	monkeypatch.setattr(pa, '_intact', no_drive)
	assert pa.load(path) is not None and pa.current(path) is not None
	inst, mode, _fs = ffttest.field_burst('B')
	assert len(pa.scan(str(tmp_path), inst, mode, 1)) == 1


def test_verify_archive_keeps_intact_rows_and_deletes_corrupt_ones(tmp_path, archive_on):
	path, _pre, db = _stored(tmp_path)
	other, _pre2, _db = _stored(tmp_path, stamp='20160201010000')
	_tamper(db, os.path.basename(path))
	logs = []
	out = pa.verify_archive(str(tmp_path), log=logs.append)
	assert (out['rows'], out['intact'], out['deleted'], out['unreadable'], out['cancelled']) == (2, 1, 1, 0, False)
	assert pa.current(path) is None                          # gone: downloaded again when its range is next requested
	assert pa.current(other) is not None
	assert any('corrupt on the drive' in line for line in logs)
	again = pa.verify_archive(str(tmp_path), log=logs.append)
	assert (again['intact'], again['skipped']) == (1, 0)       # a new lap reads every row again


@pytest.mark.parametrize('column', ['time_blob', 'good_blob', 'metadata'])
def test_verify_archive_catches_damage_in_any_part(tmp_path, archive_on, column):
	path, _pre, db = _stored(tmp_path)
	if column == 'metadata':
		with closing(sqlite3.connect(db)) as conn:
			conn.execute(f'UPDATE {pa.table()} SET cdf_size = cdf_size + 1')         # metadata: no blob touched
			conn.commit()
			conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
	else:
		_tamper(db, os.path.basename(path), column)
	assert pa.verify_archive(str(tmp_path), log=lambda *_: None)['deleted'] == 1


def test_verify_archive_rereads_once_before_deleting(tmp_path, archive_on, monkeypatch):
	path, _pre, _db = _stored(tmp_path)
	real, calls = drive_read.row, []

	def bad_first(db, root, rowid):
		calls.append(rowid)
		values = real(db, root, rowid)
		if len(calls) == 1:
			values[-1] = b'x' * len(values[-1])                    # one bad read: the next one is right
		return values

	monkeypatch.setattr(drive_read, 'row', bad_first)
	out = pa.verify_archive(str(tmp_path), log=lambda *_: None)
	assert (out['intact'], out['deleted'], len(calls)) == (1, 0, 2)


def test_verify_archive_keeps_a_row_it_cannot_read(tmp_path, archive_on, monkeypatch):
	path, _pre, _db = _stored(tmp_path)

	def unreadable(*a, **k):
		raise OSError(5, 'I/O error')

	real = drive_read.row
	monkeypatch.setattr(drive_read, 'row', unreadable)
	out = pa.verify_archive(str(tmp_path), log=lambda *_: None)
	assert (out['unreadable'], out['deleted']) == (1, 0)
	monkeypatch.setattr(drive_read, 'row', real)
	assert pa.current(path) is not None


def test_verify_archive_stops_when_cancelled(tmp_path, archive_on):
	_stored(tmp_path)
	out = pa.verify_archive(str(tmp_path), log=lambda *_: None, cancelled=lambda: True)
	assert (out['cancelled'], out['intact'], out['deleted']) == (True, 0, 0)


def test_verify_on_launch_is_on_unless_the_settings_turn_it_off(tmp_path, monkeypatch):
	import app_config
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'keep')
	assert pa.verify_on_launch()
	app_config.update('archive', {'verify_on_launch': False})
	assert not pa.verify_on_launch()
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'off')
	app_config.update('archive', {'verify_on_launch': True})
	assert not pa.verify_on_launch()                       # never while the archive is off


def test_verify_archive_without_a_database_does_nothing(tmp_path, archive_on):
	assert pa.verify_archive(str(tmp_path), log=lambda *_: None)['rows'] == 0
	assert not os.path.exists(pa.db_path(str(tmp_path)))


def test_verify_request_checks_exactly_the_rows_a_request_uses(tmp_path, archive_on):
	a, _p, db = _stored(tmp_path, stamp='20160201000000')
	b, _p, _db = _stored(tmp_path, stamp='20160201010000')
	c, _p, _db = _stored(tmp_path, stamp='20160201020000')
	_tamper(db, os.path.basename(b))
	pa._checked.clear()                                       # a later session: nothing checked yet
	not_archived = _b_path(tmp_path, '20160201030000')
	with pa.request():
		out = pa.verify_request([a, b, not_archived], log=lambda *_: None)
	assert (out['rows'], out['intact'], out['deleted']) == (2, 1, 1)
	assert out['deleted_names'] == [os.path.basename(b)]       # the caller acquires these again before using the range
	assert pa.current(b) is None and pa.current(a) is not None
	assert not any(k[3] == os.path.basename(c) for k in pa._checked)   # c was not part of the request
	again = pa.verify_request([a], log=lambda *_: None)
	assert (again['intact'], again['skipped']) == (0, 1)


def test_the_background_scan_waits_for_a_request_then_resumes(tmp_path, archive_on):
	"""Graham, 2026-11-03: "the per request pass should take precedence and the launch pass resumes after the
	request". With a request open, the scan checks nothing; the request checks its rows; when the request closes the
	scan carries on, and does not read again what the request checked."""
	import time
	paths = [_stored(tmp_path, stamp=f'201602010{h}0000')[0] for h in range(4)]
	pa._checked.clear()                                       # a later session: nothing checked yet
	logs = []
	with pa.request():
		scan = pa.BackgroundScan(str(tmp_path), log=logs.append).start()
		time.sleep(0.6)
		assert not pa._checked                                  # the scan checked nothing while the request is open
		out = pa.verify_request(paths[:2], log=logs.append)
		assert out['intact'] == 2
		assert scan.result is None                              # still waiting
	deadline = time.monotonic() + 30
	while scan.laps < 1 and time.monotonic() < deadline:
		time.sleep(0.05)
	r = scan.result
	scan.cancel()
	assert scan.join(10)
	assert r['paused'] >= 1 and r['intact'] + r['skipped'] == 4 and r['skipped'] >= 2 and r['deleted'] == 0


def test_a_background_scan_can_be_cancelled_while_it_waits(tmp_path, archive_on):
	_stored(tmp_path)
	with pa.request():
		scan = pa.BackgroundScan(str(tmp_path), log=lambda *_: None).start()
		scan.cancel()
		assert scan.join(10)
	assert scan.result['cancelled'] and scan.result['intact'] == 0


def test_verify_each_request_is_off_unless_the_settings_turn_it_on(tmp_path, monkeypatch):
	"""Graham, 2026-11-03: "Turn per request verify off" -- the option stays, off by default."""
	import app_config
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'keep')
	assert not pa.verify_each_request()
	app_config.update('archive', {'verify_each_request': True})
	assert pa.verify_each_request()
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'off')
	app_config.update('archive', {'verify_each_request': True})
	assert not pa.verify_each_request()
	assert pa.verify_request([_b_path(tmp_path)], log=lambda *_: None)['rows'] == 0




def test_the_scan_carries_on_where_the_last_one_stopped_and_wraps(tmp_path, archive_on, monkeypatch):
	"""Graham, 2026-11-03: "it can start where it left off, but it's going to need to start from the beginning when
	it gets to the end". A cancelled scan saves the row it reached; the next one (a new session) starts after it,
	runs to the end, wraps, and stops after going round once."""
	names = [os.path.basename(_stored(tmp_path, stamp=f'201602010{h}0000')[0]) for h in range(5)]
	real, read = drive_read.row, []

	def recording(db, root, rowid):
		values = real(db, root, rowid)
		read.append(values[0])
		return values

	monkeypatch.setattr(drive_read, 'row', recording)
	calls = []

	def stop_after_two():
		calls.append(1)
		return len(read) >= 2

	first = pa.verify_archive(str(tmp_path), log=lambda *_: None, cancelled=stop_after_two)
	assert first['cancelled'] and read == names[:2] and first['started_after'] == 0
	pa._checked.clear()                                       # a new launch
	read.clear()
	second = pa.verify_archive(str(tmp_path), log=lambda *_: None)
	assert read == names[2:] + names[:2]                       # after row 2, to the end, then wrapped
	assert (second['intact'], second['started_after'], second['cancelled']) == (5, 2, False)
	pa._checked.clear()
	read.clear()
	third = pa.verify_archive(str(tmp_path), log=lambda *_: None)
	assert third['started_after'] == 2 and read == names[2:] + names[:2]   # a full cycle ends where it began


def test_a_fresh_archive_is_scanned_from_the_beginning(tmp_path, archive_on):
	_stored(tmp_path, stamp='20160201000000')
	_stored(tmp_path, stamp='20160201010000')
	out = pa.verify_archive(str(tmp_path), log=lambda *_: None)
	assert (out['started_after'], out['intact']) == (0, 2)


def test_the_background_scan_goes_on_forever_until_cancelled(tmp_path, archive_on, monkeypatch):
	"""Graham, 2026-11-03: "The global scan should not stop once it's wrapped around, it should continue forever."
	Lap after lap, every row read again each lap, until cancel()."""
	import time
	names = [os.path.basename(_stored(tmp_path, stamp=f'201602010{h}0000')[0]) for h in range(2)]
	real, read = drive_read.row, []

	def recording(db, root, rowid):
		values = real(db, root, rowid)
		read.append(values[0])
		return values

	monkeypatch.setattr(drive_read, 'row', recording)
	scan = pa.BackgroundScan(str(tmp_path), log=lambda *_: None, min_lap_s=0).start()
	deadline = time.monotonic() + 30
	while scan.laps < 3 and time.monotonic() < deadline:
		time.sleep(0.02)
	scan.cancel()
	assert scan.join(10)
	assert scan.laps >= 3 and scan.error is None
	assert read[:6] == names * 3                               # each lap reads both rows again, in order


def test_a_short_lap_waits_out_the_minimum_before_the_next(tmp_path, archive_on):
	import time
	_stored(tmp_path)
	scan = pa.BackgroundScan(str(tmp_path), log=lambda *_: None, min_lap_s=60).start()
	deadline = time.monotonic() + 30
	while scan.laps < 1 and time.monotonic() < deadline:
		time.sleep(0.02)
	time.sleep(0.5)
	assert scan.laps == 1                                      # waiting out the 60 s, not reading again
	t0 = time.monotonic()
	scan.cancel()
	assert scan.join(5) and time.monotonic() - t0 < 2          # a cancel ends the wait at once


def test_a_row_verified_at_write_counts_as_checked_this_session(tmp_path, archive_on):
	"""_store reads the row back from the drive, so the request check right after archiving does not read it again."""
	path, _pre, _db = _stored(tmp_path)
	out = pa.verify_request([path], log=lambda *_: None)
	assert (out['rows'], out['skipped'], out['intact']) == (1, 1, 0)


# --- guards -----------------------------------------------------------------------------------------------------

def test_params_drift_without_a_bump_raises_on_write_and_read(tmp_path, archive_on, monkeypatch):
	path, _pre, db = _stored(tmp_path)
	pa._checked_versions.clear()
	monkeypatch.setattr(pa, 'params', lambda: {'changed': 'without a bump'})
	with pytest.raises(pa.ArchiveVersionError):
		pa.current(path)
	with pytest.raises(pa.ArchiveVersionError):
		pa._store(db, _row(path, _archived()))


def test_a_table_from_before_the_digests_is_refused(tmp_path, archive_on):
	db = pa.db_path(str(tmp_path))
	with closing(sqlite3.connect(db)) as conn:
		conn.execute(f'CREATE TABLE {pa.table()} (name TEXT PRIMARY KEY, cdf_size INTEGER NOT NULL, cdf_mtime_ns '
		             'INTEGER NOT NULL, good_first_ns INTEGER, good_last_ns INTEGER, verified INTEGER NOT NULL DEFAULT '
		             '0, time_blob BLOB NOT NULL, good_blob BLOB NOT NULL, signal_blob BLOB NOT NULL)')
		conn.commit()
	with pytest.raises(pa.ArchiveVersionError, match='digests'):
		pa.current(_b_path(tmp_path))


def test_nothing_opens_a_database_while_the_archive_is_off(tmp_path, monkeypatch):
	monkeypatch.setenv('PLASMACOOP_ARCHIVE', 'off')
	path = _b_path(tmp_path)
	inst, mode, _fs = ffttest.field_burst('B')
	assert pa.current(path) is None and pa.load(path) is None
	assert pa.scan(str(tmp_path), inst, mode, 1) == [] and pa.held(str(tmp_path), inst, mode, 1) == {}
	assert pa.archive_cdfs([path], ('B',))['archived'] == 0
	assert not os.path.exists(pa.db_path(str(tmp_path)))


def test_an_on_disk_cdf_with_other_bytes_is_not_stood_for(tmp_path, archive_on):
	path, _pre, _db = _stored(tmp_path)
	os.makedirs(os.path.dirname(path))
	with open(path, 'wb') as fh:
		fh.write(b'not the bytes the row was read from')
	assert pa.current(path) is None


# --- metadata queries: the covering index, the filters ----------------------------------------------------------

def test_metadata_queries_are_answered_from_the_covering_index(tmp_path, archive_on):
	_path, _pre, db = _stored(tmp_path)
	with closing(sqlite3.connect(db)) as conn:
		plan = ' '.join(r[-1] for r in conn.execute(
			f'EXPLAIN QUERY PLAN SELECT {", ".join(pa._META)} FROM {pa.table()} '
			f'WHERE verified = 1 AND (name GLOB ?)', ('mms1_scm_*',)))
	assert f'COVERING INDEX {pa.table()}_meta' in plan


def test_query_and_read_interval_filter_by_time(tmp_path, archive_on):
	a, pre_a, _db = _stored(tmp_path, stamp='20160201000000')
	b, pre_b, _db = _stored(tmp_path, stamp='20160201010000')
	names = [m['name'] for m in pa.query(str(tmp_path), probe=1, field='B')]
	assert names == [os.path.basename(a), os.path.basename(b)]
	lo, hi = pd.Timestamp(int(pre_b.times[100])), pd.Timestamp(int(pre_b.times[200]))
	assert [m['name'] for m in pa.query(str(tmp_path), probe=1, field='B', start=lo, end=hi)] == [os.path.basename(b)]
	t, s = pa.read_interval(str(tmp_path), 1, 'B', lo, hi)
	assert np.array_equal(t, pre_b.times[100:201])
	assert np.array_equal(s.view(np.uint32), pre_b.signal[100:201].view(np.uint32))
	assert pa.query(str(tmp_path), probe=2, field='B') == []


def test_forget_deletes_exactly_the_named_rows(tmp_path, archive_on):
	"""The census's delete step (2026-11-04): a chunk's archive rows go, every other row stays."""
	keep, _pre, _db = _stored(tmp_path, stamp='20160201000000')
	drop, _pre2, _db2 = _stored(tmp_path, stamp='20160201001000')
	assert pa.forget(str(tmp_path), [os.path.basename(drop), 'mms1_scm_brst_l2_scb_20990101000000_v1.0.0.cdf'],
	                 log=lambda *a: None) == 1
	assert pa.load(drop) is None and pa.load(keep) is not None
	assert pa.forget(str(tmp_path), [], log=lambda *a: None) == 0
	assert pa.forget(str(tmp_path / 'nowhere'), ['x.cdf'], log=lambda *a: None) == 0
