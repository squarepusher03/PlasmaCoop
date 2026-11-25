"""The acquisition phase: every CDF a request will open is downloaded and validated
before any computation starts, so nothing after it validates, deletes, or meets a
corrupt file.

Graham, 2026-09-27: "In the beginning there should be a phase where you try to
acquire all the CDFs for a range of time, then validate all of the received files,
and if some are corrupted, they alone should be retried 4 times before deleting
them and doing the manifest note. Every point from then on should never come in
contact with a corrupted file because they don't even exist on the system."

acquire() is that phase for one field (B, E or F) over one range:

  1. download   one paced call per requested probe, over the whole range -- only
                the probes asked for, never a batch of them (CDAWeb by default; below)
  2. validate   every file the download delivered, and every file on disk that the
                compute phase can open for the range (_candidates)
  3. retry      the files that failed, alone, until each passes or has failed
                GIVE_UP_AFTER (5) downloads in total
  4. give up    a name out of attempts stays deleted; its name is what the manifest's
                corrupted section lists

The rules, as decided (DESIGN_acquisition-phase.md §6):

  * A copy that fails validation is deleted on the spot, on every attempt, and the
    attempt is counted -- check_cdfs already does exactly this, so it is reused
    rather than rewritten. The count lives in cdf_validation keyed on the name, so
    it survives the deletions. Graham: "if an attempt to download a CDF ends up with
    a corrupted file, delete the file, regardless of how many tries have been
    attempted; you still make the same amount of attempts however."
  * A failed copy that cannot be deleted (something still holds it open) is not
    fetched again -- a download could not replace it either. It goes on a list, and
    the whole list is deleted as acquire()'s last step, once every other file has
    been downloaded and validated. Graham, 2026-09-30: "if it can't delete it, make
    a list or job pool to save for until after the acquisition and validation is
    otherwise complete, then lastly delete them all."
  * An SDC rate-limit reply (HTTP 429) is not an attempt. pyspedas writes the 429
    page under the CDF's name without checking the status, so such a file is
    recognised by its bytes (_is_rate_limit_page), deleted, and NOT counted. A 429
    on the file listing writes nothing at all; the download subprocess reports it
    instead (_download_and_report).
  * SDC calls are paced (_SdcPacer): at least SDC_CALL_GAP_S between them. A 429
    sets the gap to SDC_GAP_MAX_S, the wait the SDC's own 429 page asks for
    ("Retry-After: 300"), and every call that brings a CDF halves it again, never
    below SDC_CALL_GAP_S. Graham, 2026-10-01: "If you hit a 429, wait 300s, then
    dip down by half every time a CDF comes through, with a minimum wait time of 30
    seconds". The gap and when the last call ended are kept in
    {pickle_root}/sdc_pacer.json, so a new run honours a wait an earlier one was
    asked for. (It was 20 s, +15 s per 429 up to 90 s, from 2026-09-28.)
  * A given-up name that a later acquisition downloads again is validated like any
    file: an intact copy clears its record, a corrupt one is deleted at once with
    no further retries (its count is already past the limit).
  * The CDF library validates structure on every open (~60 ms each); acquisition's
    own checks keep that on, and once acquire() returns it is switched off for the
    rest of the process (skip_validation_on_open). Graham: "You shouldn't have to
    validate every single time you open a file, just in the beginning."

Where files come from (Graham, 2026-10-06: "Switch ALL downloads to CDAWeb at 8 at a
time"): the settings file (app_config) names the source. CDAWeb is the default:
cdaweb.download fetches the files pyspedas would have fetched from the SDC, 8 at a time,
paced by its own _CDAWEB_PACER (no gap between calls; 300 s after a 429, then halved per
call that brings a CDF). 'sdc' keeps the pyspedas download below, paced by _PACER. Each
downloader names its pacer (_pacer_for), so a test's fake downloader keeps the SDC
pacer. Every file still goes through the same validation, whichever source sent it.

Nothing here opens a CDF except through check_cdfs. The rate-limit sniff reads a
file's first bytes as plain bytes, never through the CDF library.
"""

import json
import logging
import multiprocessing
import os
import re
import sys
import time
from dataclasses import dataclass, field as _default
from datetime import datetime, timedelta

import app_config
import cdaweb
import cdf_validation
import ffttest
import preproc_archive

# Field letter -> (instrument, mode): the directory layout _scan_cdf_files reads and
# the datatype pyspedas downloads. B's mode follows its burst selection (scb or schb,
# instrument_spec.B_BURST); E is always dce. FGM has no mode subdirectory.
FIELDS = {'B': ffttest.field_burst('B')[:2], 'E': ffttest.field_burst('E')[:2], 'F': ('fgm', '')}

# Seconds between the end of one SDC call and the start of the next (Graham,
# 2026-10-01): 300 after any 429, then half as long after every call that brings a
# CDF -- 150, 75, 37.5 -- and never less than 30. The SDC's 429 page reads, verbatim,
# "HTTP/1.1 429 Too Many Requests / Content-Type: text/html / Retry-After: 300", so
# 300 is the wait it asks for. The 2026-09-28 rule (20 s, +15 s per 429, capped at
# 90 s) retried at 35-90 s, well inside that window.
SDC_CALL_GAP_S = 30
SDC_GAP_MAX_S = 300

# The pacer's memory, next to the validation database: the gap, and when the last
# SDC call ended, so a new process does not call the SDC early.
PACER_STATE_NAME = 'sdc_pacer.json'

# A refetch asks pyspedas for [stamp, stamp + this]. pyspedas turns the range's end
# into the SDC listing's end_date at whole-second precision after taking 0.1 s off
# it, so a bare 1 s would ask for files up to `stamp` itself and lean on the listing
# counting its end as inclusive. Two seconds keeps the file's own second inside the
# listing either way; no burst file is shorter than that, so no neighbour is pulled
# in by it.
_REFETCH_TAIL = timedelta(seconds=2)

# The first bytes of a CDF: 0xCDF3 for version 3 files (every MMS file), 0xCDF2 for
# version 2.6-2.7.
_CDF_MAGIC = (b'\xcd\xf3', b'\xcd\xf2')

# Download outcomes.
OK = 'ok'
RATE_LIMITED = 'rate_limited'   # the SDC answered 429 somewhere in the call
FAILED = 'failed'               # refused otherwise, no connection, timed out, or crashed
CANCELLED = 'cancelled'

# Validation verdicts.
PASS = 'pass'
FAIL = 'fail'                   # deleted and counted (check_cdfs)
PAGE = 'page'                   # a 429 page: deleted, not counted
MISSING = 'missing'             # gone between the scan and the check
LOCKED = 'locked'               # failed (FAIL or PAGE) but the delete failed: deleted last


@dataclass
class Acquired:
	"""What acquire() leaves behind for one field over one range.

	files       probe -> every file the compute phase may open for the range, each
	            one validated. Nothing else is on disk for it.
	given_up    names out of attempts (GIVE_UP_AFTER failed downloads): deleted, and
	            what the manifest's corrupted section should list for this request.
	unresolved  names deleted as corrupt that could not be fetched again this run
	            (downloads off, the SDC unreachable, cancelled, or a copy whose delete
	            had to wait for the end). Their counts stand; the next acquisition that
	            downloads fetches them. A copy that cannot be deleted even at the end
	            is listed here too, and left out of `files`.
	downloads   SDC calls made; rate_limited: how many of them met a 429.
	gap_s       the gap between SDC calls when acquire() returned."""
	field: str
	probes: tuple
	start: datetime
	end: datetime
	files: dict = _default(default_factory=dict)
	given_up: set = _default(default_factory=set)
	unresolved: set = _default(default_factory=set)
	downloads: int = 0
	rate_limited: int = 0
	gap_s: float = SDC_CALL_GAP_S
	cancelled: bool = False


# ---------------------------------------------------------------------------
# pacing
# ---------------------------------------------------------------------------

# Indirection so a test can run the pacer on a fake clock. _wall is the clock the
# pacer's memory is written in: monotonic time means nothing to another process.
_sleep = time.sleep
_now = time.monotonic
_wall = time.time


class _SdcPacer:
	"""The gap between SDC calls, process-wide, and remembered across runs.

	wait() and called() bracket every call and are made under ffttest._download_lock,
	the lock every download in the program already takes, so the gap is measured
	between consecutive calls whichever thread makes them. settle() then sets the
	gap for the next call from how this one went.

	attach(pickle_root) gives it a memory: {pickle_root}/sdc_pacer.json, holding the
	gap and the wall-clock time the last call ended. It picks up what an earlier run
	left there and writes after every call, so reopening the UI or rerunning main.py
	within the 300 s an earlier 429 asked for still waits. Every failure to read or
	write that file degrades to "no memory", never an exception: pacing must not stop
	a download."""

	def __init__(self, base_s=SDC_CALL_GAP_S, max_s=SDC_GAP_MAX_S, state_name=PACER_STATE_NAME, label='SDC'):
		self.base_s = base_s
		self.max_s = max_s
		self.gap_s = base_s
		self.last_end = None
		self.state_path = None
		self.state_name = state_name
		self.label = label

	def attach(self, pickle_root, log=print):
		"""Remember the gap in {pickle_root}/sdc_pacer.json, starting from what is there."""
		path = os.path.join(pickle_root, self.state_name)
		if path == self.state_path:
			return
		self.state_path = path
		try:
			with open(path) as fh:
				saved = json.load(fh)
			gap = min(max(float(saved['gap_s']), self.base_s), self.max_s)
			since = max(_wall() - float(saved['last_end_wall']), 0.0)
		except (OSError, ValueError, KeyError, TypeError):
			return
		# Keep whichever last call is more recent: this process's, or the saved one.
		if self.last_end is None or since < _now() - self.last_end:
			self.gap_s, self.last_end = gap, _now() - since
			if since < gap:
				log(f'An earlier run left a {gap:g} s gap between {self.label} calls; '
				    f'{gap - since:.0f} s of it is still to wait.')

	def _save(self):
		if self.state_path is None or self.last_end is None:
			return
		state = {'gap_s': self.gap_s, 'last_end_wall': _wall() - (_now() - self.last_end)}
		tmp = self.state_path + '.tmp'
		try:
			with open(tmp, 'w') as fh:
				json.dump(state, fh)
			os.replace(tmp, self.state_path)
		except OSError:
			pass

	def wait(self, cancelled=lambda: False, log=print):
		"""Sleep out whatever is left of the gap since the last call ended. False if
		cancelled while waiting."""
		if self.last_end is None:
			return True
		remaining = self.last_end + self.gap_s - _now()
		if remaining > 0.5:
			log(f'Waiting {remaining:.0f} s before the next {self.label} call (gap {self.gap_s} s) ...')
		while remaining > 0:
			if cancelled():
				return False
			_sleep(min(remaining, 1.0))
			remaining = self.last_end + self.gap_s - _now()
		return True

	def called(self):
		self.last_end = _now()
		self._save()

	def settle(self, rate_limited, got_cdf, log=print):
		"""The gap for the next call, from the one just made.

		  * met a 429        -> the full wait the SDC asks for (max_s, 300 s)
		  * brought a CDF    -> half the gap, never below base_s (30 s)
		  * otherwise        -> unchanged

		A call that met a 429 AND brought a CDF (one probe served, the next refused)
		waits the full 300 s: the refusal is the SDC's latest word."""
		if rate_limited:
			self.gap_s = self.max_s
			log(f'The {self.label} answered 429 (rate limit): waiting {self.gap_s:g} s before the next {self.label} call.')
		elif got_cdf and self.gap_s > self.base_s:
			self.gap_s = max(self.gap_s / 2, self.base_s)
			if self.gap_s - self.base_s < 1:
				self.gap_s = self.base_s             # a zero base would otherwise halve forever
			log(f'A CDF arrived: the gap between {self.label} calls is now {self.gap_s:g} s.')
		self._save()


_PACER = _SdcPacer()
# CDAWeb's: no gap between calls -- 400 requests, 8 at a time, met no limit of any kind (2026-10-06) -- but a 429
# still waits 300 s and then halves, as the SDC's does.
_CDAWEB_PACER = _SdcPacer(base_s=0, max_s=SDC_GAP_MAX_S, state_name='cdaweb_pacer.json', label='CDAWeb')


# ---------------------------------------------------------------------------
# the download subprocess
# ---------------------------------------------------------------------------

# _download_and_report's exit code. Bits, so a call that met both still says 429.
# 1 is left alone: it is what an uncaught exception exits with.
_EXIT_RATE_LIMITED = 2
_EXIT_REFUSED = 4


class _SdcWatch(logging.Handler):
	"""Hears pyspedas refuse or fail to reach the SDC.

	pyspedas reports a non-200 file listing only as a log line -- "Request to MMS SDC
	returned HTTP status code %d" -- and then skips that listing, so a 429 there
	leaves nothing on disk to find. A lost connection is "No internet connection!".
	Both are logged through the root logger, which is where this listens."""

	_STATUS = re.compile(r'returned HTTP status code (\d+)')

	def __init__(self):
		super().__init__(level=logging.WARNING)
		self.rate_limited = False
		self.refused = False

	def emit(self, record):
		try:
			message = record.getMessage()
		except Exception:
			return
		m = self._STATUS.search(message)
		if m:
			if m.group(1) == '429':
				self.rate_limited = True
			else:
				self.refused = True
		elif 'No internet connection' in message:
			self.refused = True


def _download_and_report(inst, mode, probe_list, trange, data_dir):
	"""Subprocess target: ffttest._do_download, with the outcome in the exit code.

	The download itself is _do_download unchanged. Around it:

	  * pyspedas is imported first, because its import configures the root logger
	    (basicConfig, then a level), and the watcher has to go on after that;
	  * download_only is set, so pyspedas returns once the files are on disk instead
	    of loading the whole range into tplot for _do_download's del_data() to throw
	    away. Which files are downloaded does not change;
	  * the root level is brought down to WARNING if something set it higher (a
	    forked child inherits download.py's CRITICAL), so the watcher hears refusals.

	Exit 0 when clean; the _EXIT_ bits otherwise. A missing pyspedas is a refusal: no
	download can have happened."""
	try:
		import pyspedas  # noqa: F401 -- for its logging setup; _do_download imports what it uses
	except ImportError:
		sys.exit(_EXIT_REFUSED)
	try:
		try:
			from pyspedas.mms_config import CONFIG as mms_cfg  # same order as _do_download
		except ImportError:
			from pyspedas.projects.mms.mms_config import CONFIG as mms_cfg
		mms_cfg['download_only'] = True
	except Exception:
		pass
	root = logging.getLogger()
	if root.level > logging.WARNING:
		for handler in root.handlers:
			handler.setLevel(max(handler.level, root.level))   # keep what they printed as it was
		root.setLevel(logging.WARNING)
	watch = _SdcWatch()
	root.addHandler(watch)
	try:
		ffttest._do_download(inst, mode, probe_list, trange, data_dir)
	finally:
		root.removeHandler(watch)
	sys.exit((_EXIT_RATE_LIMITED if watch.rate_limited else 0) | (_EXIT_REFUSED if watch.refused else 0))


def _status_from_exitcode(code):
	if code == 0:
		return OK
	if code in (_EXIT_RATE_LIMITED, _EXIT_RATE_LIMITED | _EXIT_REFUSED):
		return RATE_LIMITED
	return FAILED       # a refusal, a crash (1), a kill (negative), or anything unexpected


def _sdc_download(inst, mode, probes, start, end, data_dir):
	"""One SDC download of [start, end] for `probes`, in a subprocess bounded by
	ffttest.DOWNLOAD_TIMEOUT_S, as attempt_download runs it. Returns OK, RATE_LIMITED
	or FAILED.

	The caller holds ffttest._download_lock (see _fetch), which is why this does not
	take it: the pacing wait and the call are one critical section."""
	probe_list = [str(p) for p in probes]
	trange = [start.strftime('%Y-%m-%d/%H:%M:%S'), end.strftime('%Y-%m-%d/%H:%M:%S')]
	proc = multiprocessing.Process(target=_download_and_report, args=(inst, mode, probe_list, trange, data_dir))
	proc.start()
	proc.join(timeout=ffttest.DOWNLOAD_TIMEOUT_S)
	if proc.is_alive():
		print(f'Download for {inst} {trange} exceeded {ffttest.DOWNLOAD_TIMEOUT_S}s -- killing it.')
		proc.terminate()
		proc.join()
		return FAILED
	return _status_from_exitcode(proc.exitcode)


def _cdaweb_download(inst, mode, probes, start, end, data_dir, cancelled=None, log=print):
	"""One CDAWeb download of [start, end] for `probes`: the files pyspedas would have fetched from the SDC,
	app_config.download_concurrent() (8) at a time. Returns OK, RATE_LIMITED or FAILED, like _sdc_download.
	Called under ffttest._download_lock by _fetch, which passes cancelled and log (wants_context)."""
	return cdaweb.download(inst, mode, probes, start, end, data_dir, concurrent=app_config.download_concurrent(),
	                       cancelled=cancelled, log=log)


_cdaweb_download.pacer_name = '_CDAWEB_PACER'    # resolved at call time, so a test's fresh pacer is the one used
_cdaweb_download.wants_context = True


def _default_downloader():
	"""The downloader the settings file names: CDAWeb (the default) or the SDC. Looked up at call time, so
	patching acquisition._cdaweb_download or _sdc_download reaches every caller."""
	return _cdaweb_download if app_config.download_source() == 'cdaweb' else _sdc_download


def _pacer_for(downloader):
	"""The pacer a downloader names (pacer_name); the SDC's, _PACER, for one that names none -- _sdc_download
	and a test's fake."""
	return globals()[getattr(downloader, 'pacer_name', '_PACER')]


# ---------------------------------------------------------------------------
# which files
# ---------------------------------------------------------------------------

def _stamp(path):
	"""The filename timestamp, parsed as _scan_cdf_files parses it, or None."""
	m = re.search(r'_(\d{14})_', path)
	return datetime.strptime(m.group(1), '%Y%m%d%H%M%S') if m else None


def _candidates(data_dir, inst, mode, probe, start, end):
	"""Every CDF on disk that the compute phase can open for [start, end]: the
	anchor, every file from it through `end`, the anchor's predecessor and the last
	file's successor -- every version of each. From filename stamps only; nothing is
	opened.

	The window is the one the live discovery rules produce, using their own helpers:
	the anchor is _anchor_index's rightmost stamp <= start, or _fall_forward's first
	stamp after it; find_local_cdfs may fall forward from a stale anchor, which
	stays inside the window; pair_with_next_cdf reads the first file's predecessor
	and the last file's successor for seam samples; event_export's count_local_files
	and fgm_runs read the last file stamped before `start`. All of them are in here.

	Every version, because discovery keeps the later one of a stamp and falls back to
	the earlier when the later is gone."""
	files = ffttest._scan_cdf_files(data_dir, inst, mode, probe)
	stamps = sorted({ftime for ftime, _path in files})
	if not stamps:
		return []
	keyed = [(s, None) for s in stamps]
	first = ffttest._anchor_index(keyed, start)
	if first is None:
		first = ffttest._fall_forward(keyed, start)
	if first is None or stamps[first] > end:
		return []                              # nothing stamped in the range
	last = ffttest._anchor_index(keyed, end)
	wanted = set(stamps[max(first - 1, 0):min(last + 1, len(stamps) - 1) + 1])
	return [path for ftime, path in files if ftime in wanted]


def _snapshot(data_dir, inst, mode, probes, t0, t1):
	"""probe -> {path: (mtime_ns, size)} for the files a download of [t0, t1] can
	write: pyspedas lists from the start of t0's day (ten minutes earlier near
	midnight) and hands back one file before t0, so a day either side covers it."""
	lo, hi = t0 - timedelta(days=1), t1 + timedelta(days=1)
	out = {}
	for probe in probes:
		seen = {}
		for ftime, path in ffttest._scan_cdf_files(data_dir, inst, mode, probe):
			if lo <= ftime <= hi:
				try:
					st = os.stat(path)
				except OSError:
					continue
				seen[path] = (st.st_mtime_ns, st.st_size)
		out[probe] = seen
	return out


def _merge(into, arrived):
	for probe, paths in arrived.items():
		into.setdefault(probe, set()).update(paths)


# ---------------------------------------------------------------------------
# validating one file
# ---------------------------------------------------------------------------

def _head(path, n=4096):
	try:
		with open(path, 'rb') as fh:
			return fh.read(n)
	except OSError:
		return None


def _is_rate_limit_page(head):
	"""True for the SDC's 429 reply saved under a CDF name: not a CDF, readable text,
	and it says 429.

	The body has only ever been seen as "a ~75-byte HTML page" -- its exact text was
	never recorded -- so this asks for the two things any 429 page carries, the code
	or its reason phrase, rather than one exact body. Any other error page (503,
	404, ...) is not a rate limit and counts as a failed download."""
	if head is None or head.startswith(_CDF_MAGIC):
		return False
	sample = head[:512]
	printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in sample)
	if not sample or printable < 0.95 * len(sample):
		return False
	text = head.decode('latin-1').lower()
	return '429' in text or 'too many requests' in text


def _remove(path, why, log):
	"""Delete path. True when it is gone (deleted now or already absent), False when
	the delete failed (a locked file)."""
	name = os.path.basename(path)
	try:
		size = os.path.getsize(path)
		os.remove(path)
		log(f'Deleting {name} -- {size} bytes, {why}')
	except FileNotFoundError:
		pass
	except PermissionError:
		log(f'Cannot delete locked file: {name} ({why})')
		return False
	return True


def _validate(path, field, probe, pickle_root, passed, log):
	"""PASS, FAIL, PAGE, MISSING or LOCKED for one file. After FAIL, PAGE or MISSING
	the file is gone; LOCKED is a FAIL or PAGE whose delete failed, so the file is
	still on disk and waits for acquire()'s final deletion.

	A version already known good -- in this run, or in the database from any run --
	costs a stat and a lookup. A 429 page is deleted uncounted. Everything else goes
	through check_cdfs with the library's validation held on, which passes it and
	records it, or deletes and counts it. A file the preprocessed archive holds after
	its CDF was deleted passes: it was verified when it was archived."""
	if ffttest._archived_only(path, field):
		return PASS
	try:
		st = os.stat(path)
	except OSError:
		return MISSING
	version = (path, st.st_mtime_ns, st.st_size)
	if version in passed:
		return PASS
	if cdf_validation.passed(path, field, pickle_root=pickle_root):
		passed.add(version)
		return PASS
	head = _head(path)
	if _is_rate_limit_page(head):
		if not _remove(path, 'an SDC rate-limit (429) page, not counted', log):
			return LOCKED
		return PAGE
	if head is not None and not head.startswith(_CDF_MAGIC):
		# Not a CDF at all -- keep what the SDC actually sent, for the record.
		log(f'  {os.path.basename(path)} is not a CDF; it starts {head[:80]!r}')
	with cdf_validation.validation(True):
		ok = ffttest.check_cdfs([(path, field, probe)], pickle_root=pickle_root)[0]
	if ok is None:
		# check_cdfs has counted it and deleted it -- unless the delete failed, which
		# it only logs (_delete_bad_cdf), leaving the file where it was.
		return LOCKED if os.path.exists(path) else FAIL
	passed.add(version)
	return PASS


def _superseded(path, passed):
	"""True when a later version of the same file has passed: discovery keeps the
	later version of a stamp, so this one would never be read, and refetching it
	only spends SDC calls."""
	folder, name, stamp = os.path.dirname(path), os.path.basename(path), _stamp(path)
	return any(os.path.dirname(p) == folder and _stamp(p) == stamp and os.path.basename(p) > name
	           for p, _mtime, _size in passed)


# ---------------------------------------------------------------------------
# downloading
# ---------------------------------------------------------------------------

def _label(probes):
	return ', '.join(f'MMS{p}' for p in probes)


def _fetch(downloader, inst, mode, probes, t0, t1, data_dir, result, cancelled, log):
	"""One paced download of [t0, t1], repeated after the next gap for as long as the
	SDC refuses its file listing (429). Returns (status, arrived): arrived is probe ->
	the paths that appeared or changed, across every repeat, so each is validated.

	What the call delivered sets the next gap (_SdcPacer.settle): a 429 page among the
	arrivals is a 429 as much as a refused listing is, and a real CDF among them means
	the SDC is serving again."""
	arrived = {}
	pacer = _pacer_for(downloader)
	while True:
		before = _snapshot(data_dir, inst, mode, probes, t0, t1)
		with ffttest._download_lock:
			if not pacer.wait(cancelled, log):
				return CANCELLED, arrived
			log(f'Downloading {inst.upper()} for {_label(probes)} over '
			    f'{t0:%Y-%m-%d %H:%M:%S} - {t1:%Y-%m-%d %H:%M:%S} ...')
			try:
				if getattr(downloader, 'wants_context', False):
					status = downloader(inst, mode, probes, t0, t1, data_dir, cancelled=cancelled, log=log)
				else:
					status = downloader(inst, mode, probes, t0, t1, data_dir)
			finally:
				pacer.called()
		result.downloads += 1
		after = _snapshot(data_dir, inst, mode, probes, t0, t1)
		new = {p: {path for path, version in after[p].items() if before[p].get(path) != version} for p in probes}
		_merge(arrived, new)
		heads = [_head(path) for paths in new.values() for path in paths]
		limited = status == RATE_LIMITED or any(_is_rate_limit_page(h) for h in heads)
		if limited:
			result.rate_limited += 1
		pacer.settle(limited, any(h is not None and h.startswith(_CDF_MAGIC) for h in heads), log)
		if status != RATE_LIMITED:
			return status, arrived


def _refetch(downloader, inst, mode, waiting, lo, hi, data_dir, result, cancelled, log):
	"""Download the failed files again, alone, one probe per call. A probe's files
	stamped inside the range the first download covered share one call over their
	span -- everything else in that span is already on disk at the listed size, so
	pyspedas skips it. A file outside it (a seam neighbour, or a stray the download
	delivered) gets a call of its own, so no range around it is swept in."""
	inside, outside = {}, set()
	for path, probe in waiting.items():
		stamp = _stamp(path)
		if stamp is None:
			continue
		if lo <= stamp <= hi:
			inside.setdefault(probe, []).append(stamp)
		else:
			outside.add((probe, stamp))
	calls = [((probe,), min(stamps), max(stamps) + _REFETCH_TAIL) for probe, stamps in sorted(inside.items())]
	calls.extend(((probe,), stamp, stamp + _REFETCH_TAIL) for probe, stamp in sorted(outside))
	arrived = {}
	for probes, t0, t1 in calls:
		status, got = _fetch(downloader, inst, mode, probes, t0, t1, data_dir, result, cancelled, log)
		_merge(arrived, got)
		if status != OK:
			return status, arrived
	return OK, arrived


def _stop_downloading(status, result, log):
	if status == CANCELLED:
		result.cancelled = True
		log('Acquisition cancelled; no further downloads.')
	elif status == FAILED:
		log('The SDC download failed (refused, unreachable or timed out); '
		    'going on with what is on disk, no further downloads this run.')


# ---------------------------------------------------------------------------
# the phase
# ---------------------------------------------------------------------------

def _field_letter(field):
	letter = 'F' if field.upper() in ('F', 'FGM') else field[:1].upper()
	if letter not in FIELDS:
		raise ValueError(f'unknown field {field!r}; expected B, E or F')
	return letter


def acquire(field, probes, start, end, *, data_dir, pickle_root, download=True,
            downloader=None, cancelled=None, log=print):
	"""Download, validate and retry every CDF the compute phase can open for one
	field over [start, end] -- the module docstring has the rules. Returns Acquired.

	field       'B', 'E' or 'F' ('FGM' and signal keys like 'Bz' are accepted).
	probes      the probes the user selected. Only these are ever requested, and
	            each in its own SDC call (see below).
	download    False validates what is on disk and fetches nothing: a corrupt file
	            is still deleted and counted, and listed in `unresolved`.
	downloader  (inst, mode, probes, t0, t1, data_dir) -> OK / RATE_LIMITED / FAILED;
	            by default the settings file's source (_default_downloader: CDAWeb
	            unless it says 'sdc'). Tests give a fake one.
	cancelled   () -> bool, checked between steps and while waiting on the gap.

	pickle_root is required: it is where the validation database and its attempt
	counts live, and a count that cannot be written would let a corrupt file be
	retried forever."""
	letter = _field_letter(field)
	inst, mode = FIELDS[letter]
	probes = tuple(sorted({int(p) for p in probes}))
	downloader = downloader or _default_downloader()
	pacer = _pacer_for(downloader)
	cancelled = cancelled or (lambda: False)
	if pickle_root is None:
		raise ValueError('acquire needs pickle_root: the attempt counts live in its validation database')
	pacer.attach(pickle_root, log)           # the gap an earlier run was asked to keep
	result = Acquired(letter, probes, start, end)
	lo = start - ffttest.GAP_FILL_LOOKBACK      # the look-back discovery's anchor repair used

	received = {p: set() for p in probes}
	can_fetch = download
	if download:
		# One call per probe, never a batch (Graham, 2026-09-28: "acquire should not
		# call using all the probes at once; only request the probes the user has
		# selected"). A batched pyspedas call fires every probe's file listing back
		# to back with nothing between them -- the burst that drew the 429 of
		# 2026-09-13 -- and a 429 on any one of them repeats the whole call. Apart,
		# the pacing gap sits between every listing, and a refusal repeats only the
		# probe that was refused.
		for probe in probes:
			status, arrived = _fetch(downloader, inst, mode, (probe,), lo, end, data_dir, result, cancelled, log)
			_merge(received, arrived)
			if status != OK:
				_stop_downloading(status, result, log)
				can_fetch = False
				break

	passed = set()          # (path, mtime_ns, size) known good this run
	waiting = {}            # path -> probe: a copy failed and is gone; waiting on a good one
	counted = set()         # waiting paths whose latest failure counted (a 429 page does not)
	undeleted = {}          # path -> probe: failed, and its delete failed too; deleted last
	while True:
		for probe in probes:
			new = received.get(probe, set())
			received[probe] = set()
			for path in sorted(set(_candidates(data_dir, inst, mode, probe, start, end)) | new):
				if path in undeleted:
					if path not in new:
						continue    # already judged; only its deletion is left, at the end
					del undeleted[path]  # a download replaced it after all: judge the new copy
				verdict = _validate(path, letter, probe, pickle_root, passed, log)
				if verdict == LOCKED:
					# Not fetched again: a download cannot replace a file that cannot be
					# deleted. It waits for the final deletion after this loop.
					undeleted[path] = probe
					waiting.pop(path, None)
					counted.discard(path)
				elif verdict == PASS:
					waiting.pop(path, None)
					counted.discard(path)
				elif verdict == PAGE:
					# Deleted, not counted, refetched. The gap is _fetch's business: it
					# already went up if this page came from one of this run's
					# downloads, and a page some earlier run left behind is no 429 now.
					waiting[path] = probe
					counted.discard(path)
				elif verdict == FAIL:
					waiting[path] = probe
					counted.add(path)

		for path in list(waiting):
			name = os.path.basename(path)
			if _superseded(path, passed):
				log(f'  {name}: a later version passed; not fetching this one again')
			elif path in counted and cdf_validation.give_up(path, letter, pickle_root=pickle_root):
				log(f'  {name}: failed {cdf_validation.GIVE_UP_AFTER} downloads; given up')
				result.given_up.add(name)
			else:
				continue
			del waiting[path]
			counted.discard(path)

		if not waiting:
			break
		if not can_fetch or cancelled():
			result.cancelled = result.cancelled or cancelled()
			result.unresolved.update(os.path.basename(p) for p in waiting)
			break

		status, arrived = _refetch(downloader, inst, mode, waiting, lo, end, data_dir, result, cancelled, log)
		_merge(received, arrived)
		if status != OK:
			_stop_downloading(status, result, log)
			can_fetch = False
			continue        # still validate whatever that download delivered
		got = {path for paths in arrived.values() for path in paths}
		for path in list(waiting):
			if not os.path.exists(path):
				# The download went through and this file did not come with it: that
				# download of it failed too, and counts.
				n = cdf_validation.record_failure(path, letter, pickle_root=pickle_root)
				counted.add(path)
				log(f'  {os.path.basename(path)} did not come back on the refetch '
				    f'(attempt {n}/{cdf_validation.GIVE_UP_AFTER})')
			elif path not in got:
				# Still here and unchanged: the delete must have failed (a locked file),
				# so no download can replace it. Retrying would only spin; it waits for
				# the final deletion instead.
				log(f'  {os.path.basename(path)} is still on disk unchanged (locked?); deleting it last')
				undeleted[path] = waiting.pop(path)
				counted.discard(path)

	# Last of all, the copies whose delete failed when they were judged (Graham,
	# 2026-09-30: "make a list or job pool to save for until after the acquisition and
	# validation is otherwise complete, then lastly delete them all. Nothing should be
	# accessing those files at this point"). They are sorted as the loop sorts a
	# waiting file: a superseded copy is only deleted, a name out of attempts is given
	# up, and anything else is unresolved -- the next acquisition fetches it.
	locked = set()
	for path in sorted(undeleted):
		name = os.path.basename(path)
		if not _remove(path, 'failed validation; its delete was left until acquisition finished', log):
			log(f'  {name} still cannot be deleted; it is corrupt, so it is left out of the '
			    f'validated files. Close whatever holds it and acquire again.')
			locked.add(path)
		if _superseded(path, passed):
			continue
		if cdf_validation.give_up(path, letter, pickle_root=pickle_root):
			result.given_up.add(name)
		else:
			result.unresolved.add(name)

	# The preprocessed archive, when it is on (preproc_archive): every validated B or E CDF is archived (and,
	# unless the settings keep them, deleted), then the request's archived rows are checked from the drive
	# before anything uses them, and any found corrupt are fetched and archived again.
	if letter in ('B', 'E') and preproc_archive.enabled() and not cancelled():
		with preproc_archive.request():
			_archive_and_verify(letter, inst, mode, probes, start, end, lo, data_dir, pickle_root, downloader,
			                    can_fetch, result, locked, cancelled, log)

	result.files = {p: [f for f in _candidates(data_dir, inst, mode, p, start, end) if f not in locked]
	                for p in probes}
	result.gap_s = pacer.gap_s
	skip_validation_on_open()
	n_files = sum(len(v) for v in result.files.values())
	log(f'Acquired {letter} for {_label(probes)}: {n_files} file(s) validated, '
	    f'{len(result.given_up)} given up, {len(result.unresolved)} unresolved, '
	    f'{result.downloads} SDC call(s), gap {result.gap_s} s.')
	return result


def _archive_and_verify(letter, inst, mode, probes, start, end, lo, data_dir, pickle_root, downloader, can_fetch,
                        result, locked, cancelled, log):
	"""Acquisition's last step while the archive is on, after every CDF is validated:

	  1. archive   every validated CDF of the request not yet in the archive is preprocessed whole, stored,
	               verified against the drive, and deleted when the settings say so (preproc_archive.archive_cdfs)
	  2. verify    every archived row the request will use is checked from the drive before anything uses it
	               (preproc_archive.verify_request; archive.verify_each_request -- Graham, 2026-11-03: "An impartial
	               verification of used data is unacceptable"; default off since, Graham 2026-11-03: "Turn per
	               request verify off")
	  3. refetch   a row found corrupt has been deleted; its file is downloaded again, validated and archived
	               (1-2 once more). A file that still cannot be had is unresolved, so the request knows it is
	               missing.

	Called inside preproc_archive.request(), so the background scan waits."""
	waiting = {}
	for round_ in range(2):
		paths = {p: [f for f in _candidates(data_dir, inst, mode, p, start, end) if f not in locked] for p in probes}
		every = [f for fs in paths.values() for f in fs]
		preproc_archive.archive_cdfs([f for f in every if os.path.exists(f)], letter, log=log, cancelled=cancelled)
		if cancelled():
			return
		gone = [os.path.basename(f) for f in waiting if not os.path.exists(f) and ffttest._archived_row(f, letter) is None]
		if gone:
			log(f'  {len(gone)} corrupt archived file(s) could not be fetched again: {", ".join(sorted(gone))}')
			result.unresolved.update(gone)
		if not preproc_archive.verify_each_request():
			return
		out = preproc_archive.verify_request(every, log=log, cancelled=cancelled)
		bad = set(out['deleted_names'])
		if not bad or cancelled():
			return
		by_name = {os.path.basename(f): (f, p) for p, fs in paths.items() for f in fs}
		waiting = {by_name[n][0]: by_name[n][1] for n in bad if n in by_name and not os.path.exists(by_name[n][0])}
		if round_ or not can_fetch or not waiting:
			missing = sorted(os.path.basename(f) for f in waiting)
			if missing:
				log(f'  {len(missing)} corrupt archived file(s) are not fetched again this run: {", ".join(missing)}')
				result.unresolved.update(missing)
			return
		log(f'Fetching {len(waiting)} corrupt archived file(s) again ...')
		status, _arrived = _refetch(downloader, inst, mode, waiting, lo, end, data_dir, result, cancelled, log)
		passed = set()
		for path, probe in waiting.items():
			_validate(path, letter, probe, pickle_root, passed, log)
		if status != OK:
			_stop_downloading(status, result, log)
			can_fetch = False


def skip_validation_on_open():
	"""Switch the CDF library's validate-on-open off for this process; the mode it
	reads back (0), or None when the library cannot be reached.

	Only after acquisition: every file the compute phase opens has just been
	validated, so the ~60 ms structural walk per open buys nothing (Graham,
	2026-09-27, Q5). acquire()'s own checks hold the mode on through
	cdf_validation.validation(True), which takes the same lock as this and restores
	'off' afterwards, so a download thread acquiring the next request while this
	one computes still validates what it receives.

	The mode is per process. Pool workers start with the library default (on) and
	keep paying the 60 ms until they are started with it off."""
	try:
		from spacepy import pycdf
	except Exception:
		return None
	with cdf_validation._MODE_LOCK:
		try:
			pycdf.lib._library.CDFsetValidate(pycdf.const.VALIDATEFILEoff)
		except Exception:
			return None
	return cdf_validation.get_mode()
