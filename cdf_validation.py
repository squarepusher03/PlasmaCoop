"""The validation database: what the pipeline already knows about every CDF.

One row per CDF, in one sqlite file in the cache tree, carrying every fact that
would otherwise have to be rediscovered by opening the file again:

  status    'pass' or 'fail' -- the RESULT of validating it, not just successes.
            Graham, 2026-09-25: "you should be storing it there whether it passed
            or not since that's the whole point; you should be checking that a
            file has been validated by filename in the database before opening
            the file and getting hit with the 60ms validation."
  attempts  how many times in a row this file has failed to come down intact.
            Reset to zero the moment it passes -- it describes a CURRENT
            corruption, not a lifetime tally.
  sidecar   whether this file's segment sidecar has been computed.

Every lookup answers from the database and a stat. Nothing here ever OPENS a CDF:
that is the whole point, because the open with validation on IS the 60 ms.

Why it exists: opening a CDF with the library's file validation on costs ~50-90
ms, against ~0.4 ms with it off -- nearly all of a check is the library walking
the file. ffttest._CHECKED_CDFS remembers the answer only in memory, so every new
process and every pool worker starts empty and pays it all again. A repeat Find
events was 3-5 s slower than the pickles it replaced for exactly that reason.

The key is the filename plus the field letter. The field is in it because
_check_cdf reads different variables per field (B: scm acb + qf; E: edp epoch +
dce + quality; F: fgm b), so "this file is good" is only meaningful for the field
it was read as.

status and sidecar additionally carry the (mtime_ns, size) the file had when they
were recorded, and only count while both still match. That stat costs microseconds
and it is what catches a truncated download landing under a name that already
passed -- the one hole a name-only key would leave open. attempts deliberately
does NOT check it: it counts failures across refetches, so a counter that reset on
every new mtime would never reach the limit, and it has to keep answering after
the file is deleted.

The 5-strikes rule, Graham 2026-09-25 and 2026-09-27: a file that fails
validation is deleted on the spot -- no corrupt file outlives its detection, and
the deletion is what makes the redownload loop fetch it again -- and the failure
is counted here against its name. Once a name has failed `GIVE_UP_AFTER` times
the redownload loop stops asking for it, and its name goes in the interval
manifest's corrupted section so the record of what is missing outlives the file.
A copy some other download brings back is validated like any file before
anything reads it: bad, it is deleted and counted; intact, the record clears.

BASENAME, never the absolute path: the same file is 'E:/PlasmaCoop/...' under the
Windows venv and '/mnt/e/PlasmaCoop/...' under WSL, and one database serves both.
MMS burst names carry probe, instrument, mode, date, time and data version, so a
basename is unique across the whole tree, and v2.2.0 / v2.2.1 are separate rows,
as they must be.

Readers NEVER create the database. A pool of workers each creating and locking a
fresh sqlite file is strictly worse than the checks they were avoiding, and a
reader has nothing to record anyway.

Every failure degrades to "not known", never an exception. The cost of a wrong
"no" is one validated open -- exactly what the code did before this module. The
cost of a raise is a broken pipeline.

sqlite3 is the standard library, so nothing new is installed. It is the format
rather than a text file because at mission scale (~1.1 M CDFs) a text registry
would be read whole by every worker and rewritten on every check, where this is
one indexed lookup per open and readers can read while the parent writes.
"""

import os
import sqlite3
import threading
from contextlib import contextmanager

# Name of the database inside the cache root. One file, at the root, not one per
# directory: it is a record of what has been done, not a derived per-CDF
# artifact (a segment sidecar is the latter and lives beside its CDF's place in
# the tree).
REGISTRY_NAME = 'cdf_validation.sqlite'

# Failed validations of one name before the file is given up on and deleted.
GIVE_UP_AFTER = 5

PASS = 'pass'
FAIL = 'fail'

_SCHEMA = '''
CREATE TABLE IF NOT EXISTS touched (
	name     TEXT    NOT NULL,
	field    TEXT    NOT NULL,
	mtime_ns INTEGER NOT NULL DEFAULT 0,
	size     INTEGER NOT NULL DEFAULT 0,
	status   TEXT,
	sidecar  INTEGER NOT NULL DEFAULT 0,
	attempts INTEGER NOT NULL DEFAULT 0,
	PRIMARY KEY (name, field)
)
'''

# The validation itself is the 60 ms tax; this table is where its answer is kept
# so it is paid once per file version ever, not once per process (Graham,
# 2026-09-25: "validity is stored in the database for that file after it's been hit
# by the 60ms tax, and that database should persist over runs so it'll get better
# with time").
#
# Three different scopes in one row, and the CASEs below are what keep them apart:
#
#   status   scoped to the version. New bytes under the same name -> unknown
#            again, never stale-good. This is the truncated-refetch guard.
#   sidecar  scoped to the version, and also cleared on fail -> pass, since a
#            file that now passes is not the file that did not.
#   attempts NOT scoped to the version, and reset only by a pass. It counts
#            CONSECUTIVE failed redownloads of a corrupted file, not a lifetime
#            tally, and it must survive both a refetch and a deletion.
_UPSERT = '''
INSERT INTO touched (name, field, mtime_ns, size, status, sidecar, attempts)
VALUES (:name, :field, :mtime_ns, :size, :status, :sidecar, :bump)
ON CONFLICT(name, field) DO UPDATE SET
	status   = CASE WHEN :status IS NOT NULL                              THEN :status
	                WHEN touched.mtime_ns = :mtime_ns
	                 AND touched.size     = :size                         THEN touched.status
	                ELSE NULL END,
	sidecar  = CASE WHEN :sidecar = 1                                     THEN 1
	                WHEN touched.mtime_ns <> :mtime_ns
	                  OR touched.size     <> :size                        THEN 0
	                WHEN :status = 'pass' AND touched.status = 'fail'     THEN 0
	                ELSE touched.sidecar END,
	attempts = CASE WHEN :status = 'pass'                                 THEN 0
	                ELSE touched.attempts + :bump END,
	mtime_ns = :mtime_ns,
	size     = :size
'''


def registry_path(pickle_root=None):
	"""{pickle_root}/cdf_validation.sqlite.

	pickle_root=None falls back to data_paths.PICKLE_ROOT, which is the whole
	point of the default: the readers that need this most (_cdf_to_df,
	cdf_time_bounds, _scm_good_segments, _edp_good_segments, fgm_runs) are handed
	no pickle_root by their callers, so they cannot pass one. Imported lazily so
	a test can drive every function with an explicit path and never touch E:.

	READS may take this default. WRITES may not -- see _write. A caller that
	forgets to pass a root would otherwise silently record into the real data
	drive, which a test run did before that guard existed.
	"""
	if pickle_root is None:
		from data_paths import PICKLE_ROOT
		pickle_root = PICKLE_ROOT
	return os.path.join(pickle_root, REGISTRY_NAME)


def _file_version(cdf_path):
	"""(basename, mtime_ns, size). Zeroes when the file cannot be stat'ed, so a
	failure can still be recorded against a name whose file is already gone.

	Readers must NOT use the zero fallback to compare versions: (0, 0) matches
	itself, so a row written for an absent file would read back as current. They
	call _present_version instead."""
	name = os.path.basename(cdf_path)
	try:
		st = os.stat(cdf_path)
	except OSError:
		return name, 0, 0
	return name, st.st_mtime_ns, st.st_size


def _present_version(cdf_path):
	"""(mtime_ns, size) of a file that is actually there, or None.

	None is the reader's answer for "there is nothing to be valid": a file that
	is missing cannot be passed and cannot have a current sidecar, whatever the
	database happens to say about its name.
	"""
	try:
		st = os.stat(cdf_path)
	except OSError:
		return None
	return st.st_mtime_ns, st.st_size


def _connect(path, create):
	"""A connection to the database, or None.

	create=False returns None when the file is not there, so a reader never
	brings one into existence. busy_timeout makes a concurrent writer something
	to wait out rather than an error to handle; WAL is what lets a worker read
	while the parent writes, and is skipped quietly on a filesystem that will not
	grant it (measured 2026-09-24: granted on NTFS natively, on drvfs from WSL,
	and on ext4).
	"""
	if not create and not os.path.exists(path):
		return None
	try:
		if create:
			parent = os.path.dirname(path)
			if parent:
				os.makedirs(parent, exist_ok=True)
		conn = sqlite3.connect(path, timeout=5.0)
		conn.execute('PRAGMA busy_timeout = 5000')
		try:
			conn.execute('PRAGMA journal_mode = WAL')
		except sqlite3.Error:
			pass
		if create:
			conn.execute(_SCHEMA)
			conn.commit()
		return conn
	except (sqlite3.Error, OSError):
		return None


def _row(cdf_path, field, pickle_root, registry):
	"""(status, sidecar, attempts) for this basename+field, or None.

	Answers for a file that is no longer on disk, which the redownload counter
	needs, so it does NOT compare the version -- callers that want a
	version-scoped fact do that themselves. Opens nothing but the database --
	never the CDF, which is the entire point (Graham: don't check whether a file
	is validated by opening it).
	"""
	name = os.path.basename(cdf_path)
	conn = _connect(registry or registry_path(pickle_root), create=False)
	if conn is None:
		return None
	try:
		return conn.execute(
			'SELECT status, sidecar, attempts, mtime_ns, size'
			' FROM touched WHERE name = ? AND field = ?',
			(name, field[0]),
		).fetchone()
	except sqlite3.Error:
		return None
	finally:
		conn.close()


def _write(jobs, status, sidecar, bump, pickle_root, registry):
	"""Apply one fact to each (cdf_path, field), in a single transaction.

	One transaction for a batch is one fsync rather than one per file, which is
	the difference that matters when a day is 62 CDFs and a mission is a million.
	Returns how many rows were written; 0 on any failure, because the only cost
	of a lost write is that the next run redoes the work. A file that is already
	gone still gets its row -- that is why the counter survives deletion.
	"""
	# No implicit root on the WRITE path. registry_path()'s fallback exists for
	# readers that are handed no pickle_root; letting a write take it means any
	# caller that forgets one records into data_paths.PICKLE_ROOT -- the real data
	# drive. A test suite did exactly that. Recording nothing is the same
	# degradation every other failure here takes: the next run redoes the work.
	if registry is None and pickle_root is None:
		return 0
	rows = []
	for cdf_path, field in jobs:
		name, mtime_ns, size = _file_version(cdf_path)
		rows.append({'name': name, 'field': field[0], 'mtime_ns': mtime_ns, 'size': size,
		             'status': status, 'sidecar': 1 if sidecar else 0,
		             'bump': 1 if bump else 0})
	if not rows:
		return 0
	conn = _connect(registry or registry_path(pickle_root), create=True)
	if conn is None:
		return 0
	try:
		conn.executemany(_UPSERT, rows)
		conn.commit()
		return len(rows)
	except sqlite3.Error:
		return 0
	finally:
		conn.close()


# ---------------------------------------------------------------------------
# status -- has this exact file been validated, and what was the answer?
# ---------------------------------------------------------------------------

def status(cdf_path, field, pickle_root=None, registry=None):
	"""'pass', 'fail', or None when this exact file has never been validated.

	None means "validate it, and pay the 60 ms". It is also the answer when the
	file's (mtime_ns, size) no longer match what was recorded -- different bytes
	under the same name, which is the truncated-refetch case -- and on every
	failure path, so a reader never has to handle an error, only a cache miss.
	"""
	version = _present_version(cdf_path)
	if version is None:
		return None                      # no file, nothing to be valid
	row = _row(cdf_path, field, pickle_root, registry)
	if row is None or (row[3], row[4]) != version:
		return None                      # never seen, or different bytes same name
	return row[0] if row[0] in (PASS, FAIL) else None


def passed(cdf_path, field, pickle_root=None, registry=None):
	"""True when this exact version validated clean, so the next open of it can
	skip the library's file check. This is the question every reader asks."""
	return status(cdf_path, field, pickle_root, registry) == PASS


def failed(cdf_path, field, pickle_root=None, registry=None):
	"""True when this exact version is known bad, so it need not be opened at
	all -- it belongs in the redownload loop, or past give_up()."""
	return status(cdf_path, field, pickle_root, registry) == FAIL


def record_pass(cdf_path, field, pickle_root=None, registry=None):
	"""Record that this file validated clean, after it has actually been opened
	with validation on and passed.

	Zeroes the failed-redownload counter -- it counts a CURRENT corruption, not
	history -- and clears the sidecar flag if the file had been failing, since a
	file that now passes is different bytes from the one that did not."""
	return _write([(cdf_path, field)], PASS, False, False, pickle_root, registry) == 1


def record_passes(jobs, pickle_root=None, registry=None):
	"""record_pass() over an iterable of (cdf_path, field), one transaction."""
	return _write(jobs, PASS, False, False, pickle_root, registry)


# record() was this module's original name for record_pass(); kept so nothing
# that already calls it has to change (see the no-unilateral-removal rule).
record = record_pass
record_many = record_passes


def record_failure(cdf_path, field, pickle_root=None, registry=None):
	"""Record that this file failed validation and count the attempt.

	Returns the name's new total attempt count, or None if it could not be
	written. The count is what the redownload loop consults through give_up().
	"""
	if _write([(cdf_path, field)], FAIL, False, True, pickle_root, registry) != 1:
		return None
	return attempts(cdf_path, field, pickle_root, registry)


def attempts(cdf_path, field, pickle_root=None, registry=None):
	"""How many times in a row this NAME has failed to come down intact, whatever
	its bytes look like each time. 0 when unknown, and 0 again as soon as it
	passes. Survives the file being deleted,
	which it must -- at the limit the file is gone and the count is all that is
	left of it."""
	row = _row(cdf_path, field, pickle_root, registry)
	return 0 if row is None else int(row[2])


def give_up(cdf_path, field, pickle_root=None, registry=None, limit=GIVE_UP_AFTER):
	"""True once this name has failed validation `limit` times.

	The redownload loop's stopping condition: below the limit, a bad file is left
	for the loop to refetch; at it, the file is deleted and its name recorded in
	the interval manifest's corrupted section, so what is missing outlives it.
	"""
	return attempts(cdf_path, field, pickle_root, registry) >= limit


def corrupted_names(field=None, pickle_root=None, registry=None, limit=GIVE_UP_AFTER):
	"""Sorted basenames that have been given up on, for the manifest's corrupted
	section. field=None covers every field."""
	conn = _connect(registry or registry_path(pickle_root), create=False)
	if conn is None:
		return []
	try:
		if field is None:
			rows = conn.execute(
				'SELECT DISTINCT name FROM touched WHERE attempts >= ? ORDER BY name',
				(limit,),
			).fetchall()
		else:
			rows = conn.execute(
				'SELECT DISTINCT name FROM touched'
				' WHERE attempts >= ? AND field = ? ORDER BY name',
				(limit, field[0]),
			).fetchall()
	except sqlite3.Error:
		return []
	finally:
		conn.close()
	return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# sidecar -- is this the first touch?
# ---------------------------------------------------------------------------

def sidecar_done(cdf_path, field, pickle_root=None, registry=None):
	"""True when this exact file's segment sidecar has been computed.

	False means compute it -- the file has never been touched, or its bytes have
	changed since, in which case the old sidecar described data that is gone. Graham's rule, 2026-09-25:
	recalculate a re-downloaded CDF's sidecar, calculate each CDF's sidecar on
	first touch, and never on a touch after.

	Why ask here rather than stat the sidecar: at mission scale (~1.1 M CDFs) the
	existence check plus the staleness compare _segments_via_sidecar does today is
	two filesystem round trips per file, against one indexed lookup in a database
	the process has open anyway.
	"""
	version = _present_version(cdf_path)
	if version is None:
		return False
	row = _row(cdf_path, field, pickle_root, registry)
	return row is not None and (row[3], row[4]) == version and bool(row[1])


def record_sidecar(cdf_path, field, pickle_root=None, registry=None):
	"""Record that this version's segment sidecar now exists."""
	return _write([(cdf_path, field)], None, True, False, pickle_root, registry) == 1


def record_sidecars(jobs, pickle_root=None, registry=None):
	"""record_sidecar() over an iterable of (cdf_path, field), one transaction."""
	return _write(jobs, None, True, False, pickle_root, registry)


# ---------------------------------------------------------------------------
# the library's validation mode
# ---------------------------------------------------------------------------

# The mode is global to a process and applies at OPEN time, and threads share it
# (short_interval.plot_from_fft_data runs a ThreadPoolExecutor), so
# "set mode -> open -> restore" has to be one critical section or one thread's
# unvalidated open turns another's validated open off.
_MODE_LOCK = threading.RLock()


@contextmanager
def validation(on):
	"""Hold the library's file-validation mode at `on` for the block, then
	restore whatever it was.

	CDF() has no validate= keyword, so the mode is the only lever; the constants
	were characterised on 2026-09-15 (CDFgetValidate reads 1 on and 0 off;
	VALIDATEFILEon is itself -1). Yields the mode in force on entry, as an int,
	or None when the library could not be reached -- in which case the block
	still runs, just without the guarantee, because failing to toggle a speed
	optimisation must never stop a read.

	NOTE: this only governs opens made INSIDE the block. The mode is
	process-global, so any open elsewhere while a thread holds it off will also
	be unvalidated. Every reader open has to go through here, not most of them.
	"""
	try:
		from spacepy import pycdf
	except Exception:
		yield None
		return
	with _MODE_LOCK:
		try:
			previous = int(pycdf.lib._library.CDFgetValidate())
		except Exception:
			yield None
			return
		want = pycdf.const.VALIDATEFILEon if on else pycdf.const.VALIDATEFILEoff
		try:
			pycdf.lib._library.CDFsetValidate(want)
		except Exception:
			yield None
			return
		try:
			yield previous
		finally:
			try:
				restore = pycdf.const.VALIDATEFILEon if previous else pycdf.const.VALIDATEFILEoff
				pycdf.lib._library.CDFsetValidate(restore)
			except Exception:
				pass


def get_mode():
	"""1 when the library validates files on open, 0 when it does not, None when
	the library cannot be reached. Reads the mode; changes nothing."""
	try:
		from spacepy import pycdf
		return int(pycdf.lib._library.CDFgetValidate())
	except Exception:
		return None
