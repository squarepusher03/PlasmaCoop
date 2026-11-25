"""The preprocessed archive: every burst CDF's FFT input in one SQLite database, so the CDF itself can go.

STATUS (2026-11-03): reworked; drive check, digests, the per-request check and the background scan added; NOT wired
in. Nothing imports this module yet. tests/test_preproc_archive.py covers the codec, the version guard, the drive
check, verify_request, the background scan and their precedence.
preprocess() and
the codec are checked
against the live readers on 14 real CDFs (_wsl_scratch/preproc_reuse_proof.py); HANDOFF_preprocessed-archive.md has
what is left before it is used.

Graham, 2026-10-10 (full prompt: HANDOFF_preprocessed-archive_PROMPT.md): "I want to store the data in a compressed,
easily accessible, and filterable format ... replace it with the form of the data immediately prior to FFT
calculation ... the E & B field values should NOT be 64 bit, they should be 32 bit. The preprocessing sequence should
be versioned ... After data has been preprocessed and archived in the DB, delete the CDF(s)."

WHAT A ROW KEEPS, per B (SCM) or E (EDP) burst CDF: the two arrays the pipeline's own reader takes from that CDF
(ffttest._burst_records: every record's time, and which records are quality-good), and the FFT input made from them. Nothing derived from those arrays is stored. The good runs, the recorded runs, the
quality runs and the bounds are all computed from them by the code that computes them from a CDF today, so the
archive adds no second copy of any of that logic (Graham, 2026-10-27: "reuse the math for preprocessing as much as
possible").

  time_blob      every record's time, int64 ns -- what ffttest._recorded_sample_times reads
  good_blob      one bit per record, set on the quality-good records: the ones ffttest.read_burst_data keeps
  signal_blob    float32 |field| at each good record, sqrt(z**2 + y**2 + x**2) of read_burst_data's columns --
                 _compute_file_ffts' own expression, in its own order, so a window takes exactly what it took
  cdf_size, cdf_mtime_ns     the CDF's stat when it was read. The size is the downloader's same-name-same-size rule;
                 the mtime is what the cache staleness checks compare against once the CDF is gone
  good_first_ns, good_last_ns    the first and last good sample: what query() filters on
  verified       1 once the row has been read back FROM THE DRIVE and found identical. Only verified rows are read
  meta_sha256    sha256 of the five metadata values above (name to good_last_ns)
  blob_sha256    three sha256s, 32 bytes each: time_blob, good_blob, signal_blob. zlib's own check covers most of a
                 blob, but not the 8 raw bytes in front of the time blob: one flipped bit there shifts every time

DRIVE CHECK AND DIGESTS (Graham's OK, 2026-11-03). Bytes read from E: through the Windows cache have come back wrong
while the drive held the right ones, and a CDF deleted behind a row that only the cache held right is lost. So:
  - writing: rows are stored in batches (STORE_BATCH_ROWS / STORE_BATCH_BYTES, 2026-11-03). A batch is committed,
    checkpointed into the main file once, then each of its rows is read back from the drive (drive_read, past the
    cache in the venv) and compared byte for byte with the row in memory; verified is set on the rows that matched,
    in one transaction. A row that did not match stays unverified and its CDF stays; the rest of its batch is not
    affected. The writer's connection stays open throughout: an uncached open with nothing else holding the file
    drops the whole database from the Windows cache.
  - at write: a row is verified when a requested range not yet in the archive has been downloaded, preprocessed whole
    and stored, and only then is its CDF deleted.
  - later, two passes, both optional and both on by default (Graham, 2026-11-03: "An impartial verification of used
    data is unacceptable. I would rather have an option to verify each request"): verify_request checks every row a
    request is about to use from the drive, before it uses any; the background scan (BackgroundScan, started at
    launch) checks the whole archive, waiting while a request is open and resuming after it. A row checked once in a
    session is not read again in it. Ordinary reads (load, scan, query, ...) never check: they come after a pass.
    A row whose drive copy stays bad is deleted, so the file counts as not archived and is downloaded again. The
    digests exist for these passes: they let a row be checked by itself, years after its CDF is gone.
  - the covering index (preproc_v<N>_meta) holds every metadata column, so scan, held and query read about 0.4 KB
    per file instead of the 46 KB a row's leaf page carries of its blobs (measured 2026-10-30, 300 rows).
  Measured costs on E: (venv, 20 real rows): checkpoint 75 ms and drive read 15 ms median per row; sha256 of a row's
  blobs about 1 ms. Not covered: pycdf's CDF reads, the drive's own 64 MB cache, bad RAM in this process.

The quality test is not repeated here: ffttest._burst_records is the one place the variables are read and the
quality test is applied, for read_burst_data and for this alike. So the quality test, the epoch conversion and the
variables read live in ffttest alone; changing any of them changes what this stores, and needs a PREPROC_VERSION bump.

ONE OPEN, NO SECOND VALIDATION (Graham, 2026-10-29). preprocess() opens the CDF once and takes the times, the quality
and the field from that one open, holding them in memory until the row is built. Acquisition has already validated
the file, so the open skips the library's validation (Graham: "only validate it once").

SIZE AND SPEED, measured 2026-10-10 on 14 real CDFs (_wsl_scratch/preproc_codec_study.py) and again 2026-10-27 and
2026-10-29 on the same files with this layout (_wsl_scratch/preproc_reuse_proof.py): see the handoff for the table. A good sample
costs 19-20 bytes in an SCM CDF and 36-38 in an EDP one, and about 2.4-3.2 here; almost all of it is the float32
signal. The times and good bits together are a small fraction of a byte per sample.

VERSIONS (Graham, 2026-10-10: "for each time we change how we preprocess the data, the version should increase and a
table should be made for the new version, leaving all the previously calculated data behind and only preprocessing
what has been requested and not preprocessed in the new version"). What counts as such a change (Graham, 2026-10-29):
"versions should only change when schema changes or the design of how the code works changes (e.g. entirely
replacing how acquisition or preprocessing works, adapting for windows or osx)". Fixes, refactors and speed-ups keep
the version. PREPROC_VERSION names the table, preproc_v<N>. Rows of older tables are left where they are and never
read by the pipeline (query and read_interval can name one). A file the current table lacks is fetched again if its
CDF is gone (cdaweb.download is to skip only files the CURRENT table holds) and preprocessed into it. The versions
table keeps each version's params(); if they drift from what the version recorded, every write AND every read of the
current table raises instead of mixing two preprocessings under one table. The params cover the codec (the blobs'
format, so part of the schema) and EDP_MIN_QUALITY, and are compared as text: edit them only along with a bump.

WHERE: <cdf_dir>/mms_preprocessed.sqlite, the folder the CDFs it replaces were downloaded to. A CDF path names its
archive by itself (_split), so a test's data folder gets its own.

HOW THE PIPELINE IS TO SEE IT: an archived CDF keeps its path. ffttest._scan_cdf_files lists it at the path it had,
and each reader asks the archive before it opens a file, so discovery, seams, manifests and FFT pickles work
unchanged whether the CDF is on disk, archived, or both. A row stands for a path only while an on-disk CDF of that
name, if there is one, still has the size and mtime the row was read from.

The archive is off unless the settings file turns it on (app_config archive.enabled); PLASMACOOP_ARCHIVE=off|on|keep
overrides it for one process tree (the census is to set off, so it keeps every CDF it checks). Off, nothing here
opens a database and the pipeline is exactly what it was.

Only stdlib + numpy at import time: every FFT worker will import this through ffttest.
"""
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import zlib
from contextlib import contextmanager
from typing import NamedTuple, Optional

import numpy as np

import app_config
import drive_read

ARCHIVE_NAME = 'mms_preprocessed.sqlite'

# Bump only for a schema change (the tables, what a row holds, the codec) or a redesign of how acquisition or
# preprocessing works (replaced wholesale, adapted for Windows or macOS) -- Graham, 2026-10-29. Fixes, refactors and
# speed-ups keep it. A bump starts a new table, and is to move the FFT pickles and good-run sidecars computed from it
# into directories of their own (ffttest._fft_pickle_path, _segments_sidecar_path: '_pre<N>' suffix when N > 1), so
# nothing computed from one preprocessing is read as another's -- for a bump that changes the values a row gives
# back. That is not built yet, and v2 does not need it: v2 changes only how the times are stored, and every reader
# gets the same arrays from a v2 row as from a v1 row of the same CDF.
#   v1  2026-10-10 .. 11-03: times as int64 steps.
#   v2  2026-11-03 (Graham: "Try the step resize"): times as one byte per step. Decoding a file's times took 25 ms
#       in v1, nearly all of it zlib re-inflating 7.5 MB of int64 steps that are almost all zero bytes; 4 ms this
#       way (_wsl_scratch/archive_decode_bench_venv.py). No real data was archived as v1.
PREPROC_VERSION = 2

# time: the first time in whole microseconds, then zlib 6 of: the escape count (uint32), each escaped step in full
# (int64), and one byte per step, TIME_ESCAPE marking a step that does not fit (a gap, a negative step, or 255 us and
# over). MMS steps are 122 or 123 us. good: one bit per record (numpy packbits), zlib 6. signal: float32,
# byte-shuffled, zlib 1. v1's time codec (every step as int64, byte-shuffled, zlib 6) is kept to read v1 tables.
CODEC = 'time:us-first-u8-steps-255-escapes-int64-z6;good:packbits-z6;signal:f32-shuf-z1'
TIME_ESCAPE = 255

ARCHIVED_FIELDS = ('B', 'E')

CANCELLED = 'cancelled'          # a pool job's error text when the run was cancelled under it


def params():
	"""What PREPROC_VERSION means, as the code stands: recorded with the version's table, and compared on every
	write (ArchiveVersionError). The strings name what was done for a reader of the database; the numbers in them
	are the code's own constants."""
	import ffttest
	return {
		'read': 'ffttest._burst_records, one open, not validated again (acquisition validated it)',
		'B_good': "SCM qf == 'GGG' (ffttest._qf_good_mask)",
		'E_good': f'EDP quality >= {ffttest.EDP_MIN_QUALITY}',
		'signal': 'float32 sqrt(z**2 + y**2 + x**2) at each good record',
		'codec': CODEC,
	}


# A burst CDF in the download layout: <data_dir>/mms<p>/<inst>/brst/l2/[<mode>/]YYYY/MM/DD/<name>.cdf
_LAYOUT = re.compile(r'^(?P<root>.+?)[\\/]+mms\d[\\/]+[a-z]+[\\/]+brst[\\/]+l2[\\/]+(?:[a-z]+[\\/]+)?'
                     r'\d{4}[\\/]+\d{2}[\\/]+\d{2}[\\/]+(?P<name>[^\\/]+\.cdf)$')


class ArchiveVersionError(RuntimeError):
	"""The code's preprocessing no longer matches what its PREPROC_VERSION recorded."""


class Archived(NamedTuple):
	"""One archived CDF, as its readers would have read it."""
	recorded: np.ndarray            # int64 ns of every record: _recorded_sample_times
	good: np.ndarray                # bool per record: the records read_burst_data keeps
	signal: Optional[np.ndarray]    # float32 |field| per good record; None when load() was not asked for it

	@property
	def times(self):
		"""int64 ns of the good records: read_burst_data's time column, _good_sample_times."""
		return self.recorded[self.good]


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def _setting():
	"""(enabled, delete_cdfs): the environment's PLASMACOOP_ARCHIVE over the settings file."""
	env = (os.environ.get('PLASMACOOP_ARCHIVE') or '').strip().lower()
	if env == 'off':
		return False, False
	if env == 'on':
		return True, True
	if env == 'keep':
		return True, False
	cfg = app_config.load().get('archive') or {}
	return bool(cfg.get('enabled', False)), bool(cfg.get('delete_cdfs', True))


def enabled():
	return _setting()[0]


def delete_enabled():
	enabled_, delete = _setting()
	return enabled_ and delete


# ---------------------------------------------------------------------------
# where
# ---------------------------------------------------------------------------

def db_path(data_dir):
	return os.path.join(data_dir, ARCHIVE_NAME)


def table(version=None):
	return f'preproc_v{PREPROC_VERSION if version is None else int(version)}'


def _split(path):
	"""(data_dir, name) for a burst CDF path in the download layout, or None."""
	m = _LAYOUT.match(path)
	return (m.group('root'), m.group('name')) if m else None


def virtual_path(data_dir, inst, mode, probe, name):
	"""Where ffttest._scan_cdf_files' glob would list this CDF if it were on disk -- built the way glob builds it
	(the pattern's literal head, then one os.path.join per directory level), so an archived file and an on-disk one
	sort, compare and dedupe exactly as two on-disk files do. UNVERIFIED on Windows: see the handoff."""
	import ffttest
	t = ffttest._burst_cdf_name(name)[3]
	return os.path.join(os.path.join(data_dir, f'mms{probe}/{inst}/brst/l2/{mode}'),
	                    f'{t:%Y}', f'{t:%m}', f'{t:%d}', name)


def _name_glob(probe=None, inst=None, mode=None):
	"""A GLOB over burst CDF names: mms<p>_<inst>_brst_l2_<mode>_<stamp>_v<version>.cdf. Its literal head lets
	SQLite answer it from the primary key's index."""
	p = '?' if probe is None else str(int(probe))
	return f'mms{p}_{inst or "*"}_brst_l2_{mode + "_" if mode else ""}*'


# ---------------------------------------------------------------------------
# the database
# ---------------------------------------------------------------------------

_SCHEMA = '''
CREATE TABLE IF NOT EXISTS {t} (
	name          TEXT    PRIMARY KEY,
	cdf_size      INTEGER NOT NULL,
	cdf_mtime_ns  INTEGER NOT NULL,
	good_first_ns INTEGER,
	good_last_ns  INTEGER,
	verified      INTEGER NOT NULL DEFAULT 0,
	time_blob     BLOB    NOT NULL,
	good_blob     BLOB    NOT NULL,
	signal_blob   BLOB    NOT NULL,
	meta_sha256   BLOB    NOT NULL,
	blob_sha256   BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS {t}_meta ON {t}
	(name, verified, good_first_ns, good_last_ns, cdf_size, cdf_mtime_ns);
CREATE TABLE IF NOT EXISTS versions (
	version INTEGER PRIMARY KEY,
	params  TEXT    NOT NULL
);
'''

_META = ('name', 'cdf_size', 'cdf_mtime_ns', 'good_first_ns', 'good_last_ns')
_BLOBS = ('time_blob', 'good_blob', 'signal_blob')
_COLUMNS = _META + ('verified',) + _BLOBS + ('meta_sha256', 'blob_sha256')

_checked_versions = set()        # (db, version) whose params this process has compared
_write_lock = threading.Lock()


def _connect(db, create=False):
	"""A connection, or None when the database is not there and create is False (readers never create it). Use it
	through _db, which closes it."""
	if not create and not os.path.exists(db):
		return None
	new = False
	if create:
		os.makedirs(os.path.dirname(db) or '.', exist_ok=True)
		new = not os.path.exists(db)
	conn = sqlite3.connect(db, timeout=120.0)
	try:
		conn.execute('PRAGMA busy_timeout = 120000')
		if new:
			conn.execute('PRAGMA page_size = 65536')    # big blobs: fewer overflow pages; only takes before the first table
		try:
			conn.execute('PRAGMA journal_mode = WAL')   # readers (FFT workers) read while acquisition writes
		except sqlite3.Error:
			pass
		conn.execute('PRAGMA synchronous = FULL')       # a row is durable before its CDF is deleted
		if create:
			conn.executescript(_SCHEMA.format(t=table()))
			_check_version(conn, db, write=True)
	except BaseException:
		conn.close()
		raise
	return conn


@contextmanager
def _db(db, create=False):
	"""_connect's connection for the block, closed when the block ends, however it ends. None when the database is
	not there and create is False."""
	conn = _connect(db, create)
	try:
		yield conn
	finally:
		if conn is not None:
			conn.close()


def _check_version(conn, db, write=False):
	"""Raise ArchiveVersionError when the code's params() are not what PREPROC_VERSION recorded in this database --
	on a read of the current table as on a write, so no row preprocessed one way is served as another. A write records
	the params when the version is new; a read of a version nothing recorded has no rows to serve, and passes."""
	key = (db, PREPROC_VERSION)
	if key in _checked_versions:
		return
	if _has_table(conn):
		missing = [c for c in _COLUMNS if c not in {r[1] for r in conn.execute(f'PRAGMA table_info({table()})')}]
		if missing:
			raise ArchiveVersionError(
				f'{db}: table {table()} lacks {missing}: it was written before the digests were added (2026-11-03). '
				f'Only test databases were; move this one away.')
	now = json.dumps(params(), sort_keys=True)
	row = conn.execute('SELECT params FROM versions WHERE version = ?', (PREPROC_VERSION,)).fetchone()
	if row is None:
		if not write:
			return
		conn.execute('INSERT INTO versions (version, params) VALUES (?, ?)', (PREPROC_VERSION, now))
		conn.commit()
	elif row[0] != now:
		raise ArchiveVersionError(
			f'{db}: preprocessing version {PREPROC_VERSION} was recorded as {row[0]}, but the code now does {now}. '
			f'Bump preproc_archive.PREPROC_VERSION so the new preprocessing gets a table of its own.')
	_checked_versions.add(key)


def _has_table(conn, version=None):
	return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
	                    (table(version),)).fetchone() is not None


def _readable(conn, db, version=None):
	"""True when this database has the table to read; the current version's table only after the version guard
	(_check_version) passes. An older version's table is read as it is: it is never the pipeline's."""
	if conn is None or not _has_table(conn, version):
		return False
	if version is None or int(version) == PREPROC_VERSION:
		_check_version(conn, db)
	return True


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------

def _shuffle(a):
	b = np.ascontiguousarray(a).view(np.uint8).reshape(-1, a.dtype.itemsize)
	return np.ascontiguousarray(b.T).tobytes()


def _unshuffle(raw, dtype):
	w = np.dtype(dtype).itemsize
	n, rest = divmod(len(raw), w)
	if rest:
		raise ValueError(f'{len(raw)} bytes are not whole {np.dtype(dtype).name} values')
	return np.frombuffer(raw, np.uint8).reshape(w, n).T.copy().view(dtype).reshape(n)


def encode_time(t):
	"""int64 ns in whole microseconds -> the first microsecond, then the steps one byte each, deflated (CODEC). A step
	that does not fit a byte below TIME_ESCAPE is written as TIME_ESCAPE and kept in full in front of the steps."""
	t = np.asarray(t, dtype=np.int64)
	if t.size == 0:
		return b''
	if np.any(t % 1000):
		raise ValueError('sample times are expected in whole microseconds')
	us = t // 1000
	steps = np.diff(us)
	big = (steps < 0) | (steps >= TIME_ESCAPE)
	escaped = steps[big].astype('<i8')
	small = np.where(big, TIME_ESCAPE, steps).astype(np.uint8)
	payload = np.uint32(escaped.size).astype('<u4').tobytes() + escaped.tobytes() + small.tobytes()
	return np.int64(us[0]).astype('<i8').tobytes() + zlib.compress(payload, 6)


def decode_time(blob, version=None):
	"""int64 ns of every record from a time blob of this preprocessing version (default the current one)."""
	if version is not None and int(version) == 1:
		return _decode_time_v1(blob)
	if not blob:
		return np.empty(0, np.int64)
	first = int(np.frombuffer(blob[:8], '<i8')[0])
	raw = zlib.decompress(blob[8:])
	n_esc = int(np.frombuffer(raw[:4], '<u4')[0])
	body = 4 + 8 * n_esc
	if len(raw) < body:
		raise ValueError(f'the time blob says {n_esc} escaped steps but holds {len(raw)} bytes')
	small = np.frombuffer(raw[body:], np.uint8)
	out = np.empty(small.size + 1, np.int64)
	out[0] = first
	np.cumsum(small, dtype=np.int64, out=out[1:])
	at = np.flatnonzero(small == TIME_ESCAPE)
	if at.size != n_esc:
		raise ValueError(f'the time blob marks {at.size} escaped steps, and holds {n_esc}')
	if n_esc:                                     # each escaped step counted as TIME_ESCAPE so far: add the rest
		fix = np.zeros(small.size, np.int64)
		fix[at] = np.frombuffer(raw[4:body], '<i8') - TIME_ESCAPE
		out[1:] += np.cumsum(fix)
	out[1:] += first
	out *= 1000
	return out


def _decode_time_v1(blob):
	"""v1's time blob: the first microsecond, then every step as int64, byte-shuffled and deflated."""
	if not blob:
		return np.empty(0, np.int64)
	first = int(np.frombuffer(blob[:8], np.int64)[0])
	steps = _unshuffle(zlib.decompress(blob[8:]), np.int64)
	out = np.empty(steps.size + 1, np.int64)
	out[0] = first
	np.cumsum(steps, out=out[1:])
	out[1:] += first
	return out * 1000


def encode_good(good):
	return zlib.compress(np.packbits(np.asarray(good, dtype=bool)).tobytes(), 6)


def decode_good(blob, n):
	bits = np.frombuffer(zlib.decompress(blob), np.uint8)
	if bits.size != (n + 7) // 8:
		raise ValueError(f'{bits.size} bytes of good bits for {n} records')
	return np.unpackbits(bits, count=n).astype(bool)


def encode_signal(s):
	s = np.asarray(s)
	if s.dtype != np.float32:
		raise ValueError(f'the signal is stored as float32, not {s.dtype}')
	return zlib.compress(_shuffle(s), 1)


def decode_signal(blob, n):
	s = _unshuffle(zlib.decompress(blob), np.float32)
	if s.size != n:
		raise ValueError(f'{s.size} signal values for {n} good records')
	return s


def signal_of(x, y, z):
	"""The FFT input at each sample: |field| as _compute_file_ffts builds it -- squares added z, y, x -- stored
	float32 (Graham: "should NOT be 64 bit"). MMS SCM and EDP burst vectors are float32 already (every file
	checked), so for them the cast is a no-op and the values are bit-for-bit what every window computed."""
	return np.sqrt(z**2 + y**2 + x**2).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# digests
# ---------------------------------------------------------------------------

def _value_bytes(v):
	"""One metadata value, typed and length-prefixed so no two different rows hash alike."""
	if v is None:
		return b'\x00'
	if isinstance(v, int):
		return b'\x01' + int(v).to_bytes(8, 'big', signed=True)
	if isinstance(v, str):
		v = v.encode('utf-8')
		return b'\x02' + len(v).to_bytes(8, 'big') + v
	v = bytes(v)
	return b'\x03' + len(v).to_bytes(8, 'big') + v


def _meta_digest(d):
	h = hashlib.sha256()
	for c in _META:
		h.update(_value_bytes(d[c]))
	return h.digest()


def _blob_digests(d):
	return b''.join(hashlib.sha256(d[c]).digest() for c in _BLOBS)


def _intact(d):
	"""True when a whole row's digests hold for what it holds: its metadata and each of its blobs."""
	sums = d.get('blob_sha256')
	if d.get('meta_sha256') != _meta_digest(d) or not isinstance(sums, bytes) or len(sums) != 32 * len(_BLOBS):
		return False
	return all(isinstance(d.get(c), bytes) and sums[32 * i:32 * (i + 1)] == hashlib.sha256(d[c]).digest()
	           for i, c in enumerate(_BLOBS))


def _drive_copy(conn, db, version, rowid):
	"""{column: value} of the row as the drive holds the main file, or None when it cannot be found or parsed there."""
	t = table(version)
	try:
		cols = [r[1] for r in conn.execute(f'PRAGMA table_info({t})')]
		return dict(zip(cols, drive_read.row(db, drive_read.root_page(conn, t), rowid)))
	except (KeyError, ValueError, OSError, sqlite3.Error):
		return None


def _select(conn, version, where, args, blobs=()):
	"""{column: value} dicts of the verified rows matching where: the metadata, plus the named blobs. Taken as they
	are: the checks are verify_request (before a request uses rows) and the background scan, never a read."""
	sel = _META + tuple(blobs)
	return [dict(zip(sel, r)) for r in conn.execute(
		f'SELECT {", ".join(sel)} FROM {table(version)} WHERE verified = 1 AND ({where})', args).fetchall()]


def _decode(time_blob, good_blob, signal_blob=None, version=None):
	"""Archived from a row's blobs; version: the table's (default the current one), for its time codec."""
	recorded = decode_time(time_blob, version)
	good = decode_good(good_blob, recorded.size)
	signal = None if signal_blob is None else decode_signal(signal_blob, int(np.count_nonzero(good)))
	return Archived(recorded, good, signal)


# ---------------------------------------------------------------------------
# preprocessing one CDF
# ---------------------------------------------------------------------------

def preprocess(cdf_path, key, probe=None):
	"""Archived(recorded, good, signal) of one B or E CDF, from ONE open of it held until everything is read:
	ffttest._burst_records gives every record's time (what _recorded_sample_times reads), the quality-good mask and
	the field, and the good records' field is the FFT input exactly as read_burst_data and _compute_file_ffts take it.
	The open skips the CDF library's validation: acquisition validated the file once already (Graham, 2026-10-29).
	Raises when the CDF cannot be read."""
	import cdf_validation
	import ffttest
	from spacepy import pycdf
	f = key[0]
	if f not in ARCHIVED_FIELDS:
		raise ValueError(f'only B and E burst files are archived, not {key!r}')
	if probe is None:
		probe = ffttest._burst_cdf_name(cdf_path)[0]
	with cdf_validation.validation(False):     # the mode applies at open; the lock is held for the open alone
		cdf = pycdf.CDF(cdf_path)
	with cdf:
		time, var, good = ffttest._burst_records(cdf, cdf_path, f, probe)
	recorded = np.asarray(time.as_unit('ns').asi8, dtype=np.int64)
	good = np.asarray(good, dtype=bool)
	v = var[good]
	return Archived(recorded, good, signal_of(v[:, 0], v[:, 1], v[:, 2]))


def _row_of(cdf_path, pre, st):
	times = pre.times
	row = {
		'name': os.path.basename(cdf_path), 'cdf_size': st.st_size, 'cdf_mtime_ns': st.st_mtime_ns,
		'good_first_ns': int(times[0]) if times.size else None,
		'good_last_ns': int(times[-1]) if times.size else None,
		'verified': 0,
		'time_blob': encode_time(pre.recorded), 'good_blob': encode_good(pre.good),
		'signal_blob': encode_signal(pre.signal),
	}
	row['meta_sha256'] = _meta_digest(row)
	row['blob_sha256'] = _blob_digests(row)
	return row


_stop = None     # in an archiving pool worker: the run's stop flag (a multiprocessing Event), set on cancel


def _init_worker(stop):
	global _stop
	_stop = stop


def _stopping():
	return _stop is not None and _stop.is_set()


def _preprocess_job(job):
	"""Pool worker: preprocess + encode one CDF, and prove the codec gives the arrays back exactly. Returns
	(cdf_path, row or None, error text or None). The run's stop flag is checked before the read and again when it
	ends, so a cancelled run's jobs return CANCELLED instead of carrying on."""
	cdf_path, key, probe = job
	if _stopping():
		return cdf_path, None, CANCELLED
	try:
		before = os.stat(cdf_path)
		pre = preprocess(cdf_path, key, probe)
		if _stopping():
			return cdf_path, None, CANCELLED
		after = os.stat(cdf_path)
		if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
			return cdf_path, None, 'the CDF changed while it was read'
		row = _row_of(cdf_path, pre, after)
		back = _decode(row['time_blob'], row['good_blob'], row['signal_blob'])
		if not (np.array_equal(back.recorded, pre.recorded) and np.array_equal(back.good, pre.good)
		        and np.array_equal(back.signal.view(np.uint32), pre.signal.view(np.uint32))):
			return cdf_path, None, 'the encoded row does not decode to the arrays it was made from'
		return cdf_path, row, None
	except Exception as exc:  # noqa: BLE001 -- the caller logs it and keeps the CDF
		return cdf_path, None, f'{type(exc).__name__}: {exc}'


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

# archive_cdfs stores rows in batches (Graham, 2026-11-03: "Go for that batched verification"): a batch is flushed
# once it holds this many rows or this many bytes of blobs, and at the end. One store per row cost ~0.38 s on E:
# (three syncs, a checkpoint and a fresh connection each); a batch pays the syncs and the checkpoint once. Measured on
# E: (venv, MMS1 2016-02-01, 26 B + 26 E CDFs; _wsl_scratch/archive_batch_timing_venv.py): the store took 20.5 s one
# row at a time and 9.3 s batched; batches of 8 overlap the preprocessing pool better than 32 (archive_cdfs 13.5 s
# for both fields against 16.2 s; one row at a time, 23.7 s).
STORE_BATCH_ROWS = 8
STORE_BATCH_BYTES = 128 * 1024 * 1024
_SQL_VARS = 500                  # names or rowids per IN (...): under SQLite's bound-parameter limit


def _drive_copies(conn, db, version, rowids):
	"""{rowid: {column: value}} of these rows as the drive holds the main file, read through one uncached handle. A
	row that cannot be found or parsed there is left out; so is every row when the file cannot be opened."""
	t = table(version)
	try:
		cols = [r[1] for r in conn.execute(f'PRAGMA table_info({t})')]
		got = drive_read.rows(db, drive_read.root_page(conn, t), rowids)
	except (KeyError, ValueError, OSError, sqlite3.Error):
		return {}
	return {r: dict(zip(cols, v)) for r, v in got.items() if not isinstance(v, Exception)}


def _store_rows(db, rows):
	"""Insert (or replace) these rows in one transaction, checkpoint them into the database's main file once, read
	each back FROM THE DRIVE (past the Windows cache, in the venv; one handle for the batch, every row read by itself)
	and mark verified, in one more transaction, each row whose every blob, number and digest came back exactly as the
	row in memory holds it. All on one connection, kept open: see drive_read's caller rules. Returns one bool per row,
	True when verified. A row whose drive copy differs or cannot be read stays unverified, so its CDF stays and the
	next acquisition tries again; the rest of the batch is not affected. A reader that keeps the checkpoint from
	finishing leaves the whole batch unverified."""
	if not rows:
		return []
	t = table()
	with _write_lock, _db(db, create=True) as conn:
		for row in rows:
			cols = list(row)
			conn.execute(f'INSERT OR REPLACE INTO {t} ({", ".join(cols)}) VALUES ({", ".join("?" * len(cols))})',
			             [row[c] for c in cols])
		conn.commit()
		busy = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]   # (0, -1, -1) when not in WAL mode
		if busy:
			return [False] * len(rows)
		names = [row['name'] for row in rows]
		ids = {}
		for i in range(0, len(names), _SQL_VARS):
			chunk = names[i:i + _SQL_VARS]
			ids.update(conn.execute(f'SELECT name, rowid FROM {t} WHERE name IN ({", ".join("?" * len(chunk))})',
			                        chunk).fetchall())
		back = _drive_copies(conn, db, None, sorted(set(ids.values())))
		ok = []
		for row in rows:
			d = back.get(ids.get(row['name']))
			ok.append(d is not None and all(d.get(c) == row[c] for c in row if c != 'verified'))
		good = [ids[row['name']] for row, k in zip(rows, ok) if k]
		for i in range(0, len(good), _SQL_VARS):
			chunk = good[i:i + _SQL_VARS]
			conn.execute(f'UPDATE {t} SET verified = 1 WHERE rowid IN ({", ".join("?" * len(chunk))})', chunk)
		conn.commit()
		now = time.monotonic()
		for row, k in zip(rows, ok):
			if k:
				_checked[(db, t, ids[row['name']], row['name'])] = now   # just read from the drive: checked
		return ok


def _store(db, row):
	"""One row through _store_rows: True when it verified."""
	return _store_rows(db, [row])[0]


def archive_cdfs(paths, key, delete=None, log=print, cancelled=None, max_workers=None):
	"""Archive these validated B or E CDFs, each into the archive of the folder it was downloaded to, then (delete,
	default from the settings) delete each one whose row verified. A CDF the current table already holds verified is
	not read again; it is only deleted. Every failure keeps the CDF and is logged, never raised: the file is then
	read from disk as before, and the next acquisition tries again. Nothing happens while the archive is off.
	Returns {'archived', 'deleted', 'failed', 'bytes_cdf', 'bytes_stored'}."""
	cancelled = cancelled or (lambda: False)
	out = {'archived': 0, 'deleted': 0, 'failed': 0, 'bytes_cdf': 0, 'bytes_stored': 0}
	if key[0] not in ARCHIVED_FIELDS or not enabled():
		return out
	delete = delete_enabled() if delete is None else delete
	todo, ready = [], []
	for path in dict.fromkeys(paths):
		if not os.path.exists(path) or _split(path) is None:
			continue
		(ready if current(path) is not None else todo).append(path)
	if not todo and not (delete and ready):
		return out
	if todo:
		log(f'Archiving {len(todo)} {key[0]} CDF(s) (preprocessing version {PREPROC_VERSION}) ...')
	jobs = [(p, key[0], None) for p in todo]

	def results():
		"""(path, row, err) per job as each finishes. A cancel is seen within half a second: the queued jobs are
		dropped, and the stop flag makes the running ones return when their read ends."""
		if len(jobs) <= 2:
			for job in jobs:
				if cancelled():
					return
				yield _preprocess_job(job)
			return
		import multiprocessing
		from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
		import ffttest
		workers = max_workers or min(ffttest.MAX_POOL_WORKERS, len(jobs))
		ctx = multiprocessing.get_context()
		stop = ctx.Event()
		with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
		                         initargs=(stop,)) as pool:
			pending = {pool.submit(_preprocess_job, job) for job in jobs}
			try:
				while pending:
					done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
					for fut in done:
						yield fut.result()
					if cancelled():
						return
			finally:
				# a cancel, or the loop below stopped reading: nothing else starts, nothing running carries on
				stop.set()
				for fut in pending:
					fut.cancel()

	pending = []                     # (path, row) preprocessed and not yet stored: flushed as one batch per archive

	def flush():
		by_db = {}
		for path, row in pending:
			by_db.setdefault(db_path(_split(path)[0]), []).append((path, row))
		pending.clear()
		for db, items in by_db.items():
			err = None
			try:
				oks = _store_rows(db, [row for _p, row in items])
			except ArchiveVersionError:
				raise
			except (sqlite3.Error, OSError) as exc:
				oks, err = [False] * len(items), f'{type(exc).__name__}: {exc}'
			for (path, row), ok in zip(items, oks):
				if not ok:
					out['failed'] += 1
					log(f'  {os.path.basename(path)}: the archive row did not read back as written '
					    f'({err or "mismatch"}); the CDF stays.')
					continue
				out['archived'] += 1
				out['bytes_cdf'] += row['cdf_size']
				out['bytes_stored'] += sum(len(row[c]) for c in _BLOBS)
				ready.append(path)

	for path, row, err in results():
		if cancelled():
			break
		if err == CANCELLED:
			continue
		if row is None:
			out['failed'] += 1
			log(f'  {os.path.basename(path)}: not archived ({err}); the CDF stays.')
			continue
		pending.append((path, row))
		if (len(pending) >= STORE_BATCH_ROWS
		        or sum(len(r[c]) for _p, r in pending for c in _BLOBS) >= STORE_BATCH_BYTES):
			flush()
	if not cancelled():
		flush()                      # on a cancel the unstored rows are dropped: their CDFs stay, as said below
	if cancelled():
		log('Archiving cancelled; the CDFs not yet archived stay on disk, and so do the ones archived: the next '
		    'acquisition deletes those without reading them again.')
	if out['archived']:
		log(f'Archived {out["archived"]} CDF(s): {out["bytes_cdf"] / 1e6:.0f} MB of CDF kept as '
		    f'{out["bytes_stored"] / 1e6:.1f} MB ({out["bytes_cdf"] / max(out["bytes_stored"], 1):.1f}x smaller).')
	if delete:
		for path in ready:
			if cancelled():
				break
			if current(path) is None:              # deleted only behind a verified, current row
				continue
			try:
				os.remove(path)
			except FileNotFoundError:
				pass
			except OSError as exc:
				log(f'  {os.path.basename(path)}: archived, but the CDF could not be deleted ({exc}); it stays.')
				continue
			out['deleted'] += 1
		if out['deleted']:
			log(f'Deleted {out["deleted"]} archived CDF(s).')
	return out


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def forget(data_dir, names, log=print):
	"""Delete these CDF names' rows from the current table of data_dir's archive; returns how many went. For the
	whole-mission census, which deletes each chunk's data before the next (2026-11-04): with the archive on it is the
	rows, not the CDFs, that hold the chunk (about 2.9 TB of archive for the whole mission, more than E: holds). The
	file does not shrink; the next chunk's rows reuse the freed pages."""
	names = sorted(set(names))
	db = db_path(data_dir)
	if not names or not os.path.exists(db):
		return 0
	gone = 0
	with _db(db) as conn:
		if not _has_table(conn):
			return 0
		t = table()
		with _write_lock:
			for i in range(0, len(names), 500):
				part = names[i:i + 500]
				cur = conn.execute(f'DELETE FROM {t} WHERE name IN ({",".join("?" * len(part))})', part)
				gone += cur.rowcount
			conn.commit()
	log(f'Archive: {gone} row(s) deleted of {len(names)} name(s) asked.')
	return gone


def _lookup(path, cols=()):
	"""(metadata dict, values of cols) of the verified current-version row standing for this CDF path, or None:
	archive off, path not in the download layout, not archived, or an on-disk CDF of that name whose size or mtime
	is not the one it was archived from (new bytes under the old name are preprocessed again)."""
	if not enabled():
		return None
	parts = _split(path)
	if parts is None:
		return None
	db = db_path(parts[0])
	with _db(db) as conn:
		if not _readable(conn, db):
			return None
		rows = _select(conn, None, 'name = ?', (parts[1],), cols)
	if not rows:
		return None
	got = rows[0]
	meta = {c: got[c] for c in _META}
	values = tuple(got[c] for c in cols)
	try:
		st = os.stat(path)
	except OSError:
		return meta, values
	if (st.st_size, st.st_mtime_ns) != (meta['cdf_size'], meta['cdf_mtime_ns']):
		return None
	return meta, values


def current(path):
	"""The metadata (a dict) of the verified current-version row standing for this CDF path, or None -- see _lookup."""
	hit = _lookup(path)
	return None if hit is None else hit[0]


def archived_only(path):
	"""current(path) when the CDF itself is not on disk, else None."""
	return None if os.path.exists(path) else current(path)


def load(path, signal=True):
	"""Archived(recorded, good, signal) of the current row standing for this CDF path, or None. signal=False skips
	the float32 values, for the readers that want only times."""
	hit = _lookup(path, _BLOBS if signal else _BLOBS[:2])
	return None if hit is None else _decode(*hit[1])


def scan(data_dir, inst, mode, probe):
	"""[(file_time, virtual path)] of every verified current-version row for this probe/instrument/mode."""
	import ffttest
	if not enabled():
		return []
	db = db_path(data_dir)
	with _db(db) as conn:
		if not _readable(conn, db):
			return []
		names = [m['name'] for m in _select(conn, None, 'name GLOB ?', (_name_glob(probe, inst, mode),))]
	return [(ffttest._burst_cdf_name(n)[3], virtual_path(data_dir, inst, mode, probe, n)) for n in names]


def held(data_dir, inst, mode, probe):
	"""{name: cdf_size} of every verified current-version row -- what a download need not fetch again."""
	if not enabled():
		return {}
	db = db_path(data_dir)
	with _db(db) as conn:
		if not _readable(conn, db):
			return {}
		return {m['name']: m['cdf_size']
		        for m in _select(conn, None, 'name GLOB ?', (_name_glob(probe, inst, mode),))}


# ---------------------------------------------------------------------------
# later verification: per request, and the background scan
# ---------------------------------------------------------------------------
#
# Graham, 2026-11-03: "An impartial verification of used data is unacceptable. I would rather have an option to
# verify each request. You may add the global scan in the background ... the per request pass should take precedence
# and the launch pass resumes after the request."
#
#   verify_request   every archived row a request is about to use, checked from the drive before any of it is used.
#                    On (archive.verify_each_request; default false since Graham, 2026-11-03: "Turn per request
#                    verify off"), no request uses a row this session has not
#                    checked, so what a request uses is verified whole, never in part.
#   background scan  every row of the archive, in a thread started at launch (archive.verify_on_launch, default
#                    true). While any request is open (request()), it waits between rows, and carries on from the
#                    row it stopped at when the last request closes. Across launches it carries on from the row the
#                    last scan reached; at the end it wraps to the beginning and goes on, lap after lap, until the
#                    app closes or cancels it.
#
# Both read a row from the drive (past the Windows cache in the venv) and compare it with the digests stored when it
# was written. The scan never stops on its own (Graham, 2026-11-03: "The global scan should not stop once it's
# wrapped around, it should continue forever"): each lap reads every row again, except rows a request checked during
# that lap. A request does not re-read a row already checked this session, by a request or by the scan: a verified
# row is never rewritten in place (a new download of the file is a new row, with a new rowid).

_SCAN_POSITION = ('CREATE TABLE IF NOT EXISTS scan_position '
                  '(tbl TEXT PRIMARY KEY, last_rowid INTEGER NOT NULL, updated_ns INTEGER NOT NULL)')
_gate = threading.Condition()
_open_requests = 0
_checked = {}                    # (db, table, rowid, name) -> time.monotonic() it was last checked intact, this session


def verify_on_launch():
	"""Whether the app is to start the background scan at launch: archive.verify_on_launch, on unless set false, and
	never while the archive is off."""
	return enabled() and bool((app_config.load().get('archive') or {}).get('verify_on_launch', True))


def verify_each_request():
	"""Whether every request verifies the rows it uses first: archive.verify_each_request, off unless set true
	(Graham, 2026-11-03: "Turn per request verify off"), and
	never while the archive is off."""
	return enabled() and bool((app_config.load().get('archive') or {}).get('verify_each_request', False))


@contextmanager
def request():
	"""Hold for the whole of one request, from its verification to its last read: the background scan waits between
	rows until no request is open, then resumes where it stopped. Requests may nest or overlap."""
	global _open_requests
	with _gate:
		_open_requests += 1
	try:
		yield
	finally:
		with _gate:
			_open_requests -= 1
			_gate.notify_all()


def _wait_for_requests(cancelled):
	"""Block while a request is open. True when it had to wait (the scan was paused)."""
	waited = False
	with _gate:
		while _open_requests and not cancelled():
			waited = True
			_gate.wait(0.5)
	return waited


def _check_row(conn, db, t, root, cols, rowid, name, attempts):
	"""('intact' | 'bad' | 'unreadable', MB, why) for one row read from the drive, read again on a failure."""
	verdict, why = 'bad', ''
	for _ in range(attempts):
		try:
			d = dict(zip(cols, drive_read.row(db, root, rowid)))
		except OSError as exc:
			verdict, why = 'unreadable', f'{type(exc).__name__}: {exc}'
			continue
		except (KeyError, ValueError) as exc:
			verdict, why = 'bad', f'{type(exc).__name__}: {exc}'
			continue
		if d.get('name') == name and _intact(d):
			_checked[(db, t, rowid, name)] = time.monotonic()
			return 'intact', sum(len(d[c]) for c in _BLOBS) / 1e6, ''
		verdict, why = 'bad', 'its digests do not match what the drive holds'
	return verdict, 0.0, why


def _scan_position(conn, t):
	"""The rowid the background scan of table t last reached, 0 when it never ran. Kept in the archive itself
	(scan_position), so it survives restarts."""
	conn.execute(_SCAN_POSITION)
	row = conn.execute('SELECT last_rowid FROM scan_position WHERE tbl = ?', (t,)).fetchone()
	return row[0] if row else 0


def _save_scan_position(conn, t, rowid):
	with _write_lock:
		conn.execute('INSERT OR REPLACE INTO scan_position (tbl, last_rowid, updated_ns) VALUES (?, ?, ?)',
		             (t, rowid, time.time_ns()))
		conn.commit()


def _done(key, since):
	"""Checked intact this session -- at or after `since` when it is given (the scan's lap start)."""
	at = _checked.get(key)
	return at is not None and (since is None or at >= since)


def _verify_rows(db, version, where, args, log, cancelled, attempts, pause, label, resume=False, since=None):
	"""Check the verified rows matching where, in rowid order; delete each that stays bad. pause: wait between rows
	while a request is open (the background scan). resume (the background scan, Graham 2026-11-03: "it can start
	where it left off, but it's going to need to start from the beginning when it gets to the end"): start after the
	row the last scan reached, go to the end, then from the beginning up to that row -- one lap -- saving the row
	reached every 30 s, on a cancel and at the end. since: a row checked at or after it is skipped (None: checked at
	all this session). Returns the counts, with the deleted names."""
	out = {'rows': 0, 'intact': 0, 'skipped': 0, 'deleted': 0, 'unreadable': 0, 'MB': 0.0, 'seconds': 0.0,
	       'cancelled': False, 'paused': 0, 'deleted_names': [], 'started_after': 0}
	t0 = last = time.perf_counter()
	with _db(db) as conn:
		if not _readable(conn, db, version):
			return out
		t = table(version)
		rows = conn.execute(f'SELECT rowid, name FROM {t} WHERE verified = 1 AND ({where}) ORDER BY rowid',
		                    args).fetchall()
		if resume:
			start = out['started_after'] = _scan_position(conn, t)
			rows = [r for r in rows if r[0] > start] + [r for r in rows if r[0] <= start]
		out['rows'] = len(rows)
		fresh = sum(not _done((db, t, r, n), since) for r, n in rows)
		if not fresh:
			out['skipped'] = len(rows)
			if resume and rows:
				_save_scan_position(conn, t, rows[-1][0])
			return out
		log(f'{label}: verifying {fresh} archived file(s) from the drive'
		    + (f' ({len(rows) - fresh} already checked)' if fresh < len(rows) else '')
		    + (f', starting after row {out["started_after"]}' if resume and out['started_after'] else '') + ' ...')
		root = drive_read.root_page(conn, t)
		cols = [r[1] for r in conn.execute(f'PRAGMA table_info({t})')]
		reached = None
		for done, (rowid, name) in enumerate(rows):
			if pause and _wait_for_requests(cancelled):
				out['paused'] += 1
			if cancelled():
				out['cancelled'] = True
				log(f'{label}: cancelled after {done} of {len(rows)} file(s); the rest stay as they are'
				    + ('; the next scan carries on from here.' if resume else '.'))
				break
			if _done((db, t, rowid, name), since):              # checked already, maybe by a request just now
				out['skipped'] += 1
				reached = rowid
				continue
			verdict, mb, why = _check_row(conn, db, t, root, cols, rowid, name, attempts)
			if verdict == 'intact':
				out['intact'] += 1
				out['MB'] += mb
			elif verdict == 'unreadable':
				out['unreadable'] += 1
				log(f'  {name}: could not be read from the drive ({why}); kept, checked again next time.')
			else:
				with _write_lock:
					conn.execute(f'DELETE FROM {t} WHERE rowid = ?', (rowid,))
					conn.commit()
				out['deleted'] += 1
				out['deleted_names'].append(name)
				log(f'  {name}: corrupt on the drive ({why}); deleted from the archive, so it is downloaded again.')
			reached = rowid
			now = time.perf_counter()
			if now - last >= 30:
				last = now
				if resume:
					_save_scan_position(conn, t, reached)
				log(f'  {label}: {done + 1} of {len(rows)} checked ({out["MB"]:.0f} MB, {now - t0:.0f} s)')
		if resume and reached is not None:
			_save_scan_position(conn, t, reached)
	out['seconds'] = round(time.perf_counter() - t0, 1)
	out['MB'] = round(out['MB'], 1)
	log(f'{label}: {out["intact"]} intact, {out["deleted"]} deleted as corrupt, {out["unreadable"]} unreadable, '
	    f'{out["skipped"]} already checked; {out["seconds"]:.0f} s.')
	return out


def verify_request(paths, log=print, cancelled=None, attempts=2):
	"""Check, from the drive, every archived row standing for these CDF paths (the files a request is about to use),
	before the request uses any of them. Paths with no verified current row are left to acquisition as today. A row
	that stays bad is deleted: its name is in 'deleted_names', and the caller acquires those files again (download,
	preprocess, store, verify) before the request goes on, so the request never uses a row this session has not
	checked. Rows checked earlier this session (by a request or the background scan) are not read again. Call it
	inside request(), so the background scan waits. Nothing happens while the archive is off."""
	cancelled = cancelled or (lambda: False)
	total = {'rows': 0, 'intact': 0, 'skipped': 0, 'deleted': 0, 'unreadable': 0, 'MB': 0.0, 'seconds': 0.0,
	         'cancelled': False, 'paused': 0, 'deleted_names': []}
	if not enabled():
		return total
	by_db = {}
	for p in dict.fromkeys(paths):
		parts = _split(p)
		if parts is not None:
			by_db.setdefault(db_path(parts[0]), []).append(parts[1])
	for db, names in by_db.items():
		for i in range(0, len(names), 500):                    # SQLite's bound-parameter limit
			chunk = names[i:i + 500]
			got = _verify_rows(db, None, f'name IN ({", ".join("?" * len(chunk))})', chunk, log, cancelled,
			                   attempts, pause=False, label='Request check')
			for k in ('rows', 'intact', 'skipped', 'deleted', 'unreadable', 'MB', 'seconds'):
				total[k] += got[k]
			total['deleted_names'] += got['deleted_names']
			if got['cancelled']:
				total['cancelled'] = True
				return total
	return total


def verify_archive(data_dir, log=print, cancelled=None, version=None, attempts=2, pause=True):
	"""Check every verified row of the archive in data_dir from the drive: the background scan's work, run in the
	caller's thread. pause (default): wait between rows while a request is open, then go on from where it stopped.
	One lap: it starts after the row the last scan reached (scan_position, kept in the archive, so across launches),
	runs to the end, then wraps to the beginning and stops when it has been round once. BackgroundScan runs lap after
	lap. Rows checked by a request during this lap are skipped; rows checked before it are read again.
	A row that stays bad is deleted, so the file counts as not archived and is downloaded again when its range is
	next requested; a row the drive cannot be read for is kept. About 26 MB/s on E: (venv, 2026-11-03).
	Returns {'rows', 'intact', 'skipped', 'deleted', 'unreadable', 'MB', 'seconds', 'cancelled', 'paused',
	'deleted_names', 'started_after'}."""
	return _verify_rows(db_path(data_dir), version, '1', (), log, cancelled or (lambda: False), attempts, pause,
	                    label='Archive scan', resume=True, since=time.monotonic())


class BackgroundScan:
	"""The background scan: verify_archive lap after lap in a daemon thread, forever, until cancel() (Graham,
	2026-11-03: "it should continue forever"). Started by the app at launch when verify_on_launch() says so. Each lap
	re-lists the rows, so files archived meanwhile join the next lap. A lap that takes less than min_lap_s is followed
	by a wait for the rest of it, so a small or empty archive is not read in a tight loop. result holds the last lap's
	counts, laps how many have finished."""

	def __init__(self, data_dir, log=print, min_lap_s=60.0):
		self._stop = threading.Event()
		self.result = None
		self.laps = 0
		self.error = None
		self._thread = threading.Thread(target=self._run, args=(data_dir, log, min_lap_s), name='archive-scan',
		                                daemon=True)

	def _run(self, data_dir, log, min_lap_s):
		try:
			while not self._stop.is_set():
				t0 = time.monotonic()
				self.result = verify_archive(data_dir, log=log, cancelled=self._stop.is_set)
				if self.result['cancelled']:
					break
				self.laps += 1
				self._stop.wait(max(0.0, min_lap_s - (time.monotonic() - t0)))
		except Exception as exc:  # noqa: BLE001 -- a failed scan is logged; the app goes on
			self.error = exc
			log(f'Archive scan stopped: {type(exc).__name__}: {exc}')

	def start(self):
		self._thread.start()
		return self

	def cancel(self):
		self._stop.set()
		with _gate:
			_gate.notify_all()

	def join(self, timeout=None):
		self._thread.join(timeout)
		return not self._thread.is_alive()


# ---------------------------------------------------------------------------
# filtering, for people (notebooks, ad-hoc scripts)
# ---------------------------------------------------------------------------

def query(data_dir, probe=None, field=None, mode=None, start=None, end=None, version=None):
	"""Metadata rows (dicts) of verified files, filtered by probe, field ('B'/'E'), burst mode and the good data
	overlapping [start, end] (datetimes, UTC), in filename-stamp order. With a field, mode defaults to the one that
	field has selected (ffttest.field_burst: B_BURST for B), so B's scb and schb files -- which cover the same times
	at different rates -- are never mixed. version picks an older table."""
	db = db_path(data_dir)
	with _db(db) as conn:
		return _query(conn, db, probe, field, mode, start, end, version)


def _query(conn, db, probe, field, mode, start, end, version):
	import ffttest
	if not _readable(conn, db, version):
		return []
	inst = None
	if field is not None:
		inst, selected, _fs = ffttest.field_burst(field[0])
		mode = mode or selected
	where, args = ['name GLOB ?'], [_name_glob(probe, inst, mode)]
	if end is not None:
		where.append('good_first_ns <= ?')
		args.append(_ns(end))
	if start is not None:
		where.append('good_last_ns >= ?')
		args.append(_ns(start))
	rows = _select(conn, version, ' AND '.join(where), args)
	return sorted(rows, key=lambda m: (ffttest._burst_cdf_name(m['name'])[3], m['name']))


def read_interval(data_dir, probe, field, start, end, version=None, mode=None):
	"""(t int64 ns, s float32) of every good sample of one probe, field and burst mode (query's default: the field's
	selected mode) inside [start, end], across files. One connection, closed when the reads are done."""
	lo, hi = _ns(start), _ns(end)
	db = db_path(data_dir)
	ts, ss = [np.empty(0, np.int64)], [np.empty(0, np.float32)]
	with _db(db) as conn:
		for meta in _query(conn, db, probe, field, mode, start, end, version):
			got = _select(conn, version, 'name = ?', (meta['name'],), _BLOBS)
			if not got:
				continue                                         # replaced or deleted since the query
			a = _decode(*(got[0][c] for c in _BLOBS), version=version)
			t = a.times
			m = (t >= lo) & (t <= hi)
			ts.append(t[m])
			ss.append(a.signal[m])
	t, s = np.concatenate(ts), np.concatenate(ss)
	order = np.argsort(t, kind='stable')
	return t[order], s[order]


def _ns(dt):
	import pandas as pd
	return int(pd.Timestamp(dt).as_unit('ns').value)
