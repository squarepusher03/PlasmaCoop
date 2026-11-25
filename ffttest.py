from datetime import datetime, timedelta

import os
import glob
import re
import shutil
import threading
import multiprocessing
import atexit
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial, lru_cache
from multiprocessing import Manager

import cdf_validation          # the per-CDF validation database (pass/fail, refetch count, sidecar)

_corrupted_log_lock = threading.Lock()

# General-purpose serialization for attempt_download -- harmless insurance
# for callers outside a queue-based pipeline (fce.py, short_interval.py,
# spike_biggest_burst_fft.py all trigger downloads directly through
# find_cdf_files, independent of each other). Scripts that route every
# download through a single dedicated worker thread (see
# plot_biggest_spikes.py) never contend on this at all, by construction.
_download_lock = threading.Lock()

# Hard wall-clock ceiling on a single download attempt. pyspedas/requests set
# their own per-file connect/read timeouts, but a stall somewhere upstream of
# that (proxy/VPN swallowing packets, a hung DNS lookup, etc.) isn't bounded
# by those -- one such stall previously blocked the whole pipeline for hours
# with _download_lock held the entire time. Running the actual download in a
# subprocess means a stalled attempt can be killed outright on timeout,
# rather than just abandoned-but-still-running-and-still-holding-the-lock.
DOWNLOAD_TIMEOUT_S = 900

# When repairing a gap at the `start` boundary, look slightly further back
# than `start` itself -- a real burst can begin a little before `start` and
# extend across it, and an exact-boundary query would miss that overlap
# entirely. Only widens the DOWNLOAD request; anchor selection and the
# manifest still clip to the real requested `start` (see cdf_data_intervals'
# clip_start), so this never reports data earlier than what was asked for.
GAP_FILL_LOOKBACK = timedelta(minutes=1)

# Serialization versions -- bump either constant whenever the corresponding
# generating logic changes in a way that makes existing cached pickles wrong
# (a new filter, a new/changed column, a fixed unit bug, etc). ensure_data_pickle
# and _compute_and_save_fft delete-and-regenerate any pickle whose version
# sidecar doesn't match, automatically, the next time anything touches it -- no
# manual cache-hunting required.
#
# Version is the SECOND of two staleness signals, not the only one: source
# mtime is checked first (see _invalidate_if_stale). Version answers "which
# code built this"; mtime answers "was the input the same input". A bump is
# therefore only needed when the CODE changed -- changed DATA is caught on its
# own, without touching these numbers. This exists because of two real incidents in
# one session: the SCM quality-flag filter (_cdf_to_df) and the zero-power-row
# removal (_compute_file_ffts) both silently affected years of already-cached
# pickles, and both were found only by hand. A pickle with no version sidecar
# at all (i.e. everything cached before this mechanism existed) is treated as
# a mismatch too, so introducing this at version 1 retroactively invalidates
# the entire pre-existing cache exactly once.
#
# Keyed BY INSTRUMENT rather than one number for everything: a change that only
# affects how one instrument is converted should only invalidate that
# instrument's cache. Adding the EDP quality filter to _cdf_to_df (see its
# docstring) changed edp content only -- scm/fgm pickles are byte-identical
# before and after, and re-converting every SCM burst file (plus recomputing
# every SCM FFT on top of it) to ship an EDP-only fix would be hours of pure
# waste. Bump only the instrument whose generating logic actually moved.
DATA_PICKLE_VERSION = {   # raw per-file CDF->DataFrame conversion (_cdf_to_df)
	'scm': 2,
	'edp': 4,             # 2->3: quality==3 filter added, matching SCM's qf filter
	                      # 3->4: relaxed to quality >= EDP_MIN_QUALITY (2). MUST
	                      # bump: the v3 pickles are EMPTY for 298/300 files on
	                      # the requested dates, and neither staleness signal in
	                      # _invalidate_if_stale can see a changed filter -- the
	                      # CDF mtime is untouched and existence+version was the
	                      # only other test. Without this bump the new threshold
	                      # would silently never take effect. The regenerated
	                      # data pickles are then newer than every FFT pickle
	                      # derived from them, so the E-side FFT cache cascades
	                      # automatically and FFT_PICKLE_VERSION needs no bump.
	'fgm': 2,
}
FFT_PICKLE_VERSION = {    # per-file FFT computation (_compute_file_ffts)
	'scm': 4,             # 2->3: TWO content changes at once. (a) the pickles now
	                      # hold the FULL spectrum to Nyquist -- lpf_lim is no
	                      # longer baked in, it is applied where the rows are
	                      # binned. (b) SCB_FS (now SAMPLE_RATE) corrected 8196.7 -> 8192, which
	                      # moves n_expected 4098 -> 4096 and so changes how many
	                      # samples a window borrows across a file boundary.
	                      # MUST bump: neither staleness signal can see either
	                      # one. The CDF mtime is untouched, and a PARAMETER
	                      # change is invisible to a version that did not move --
	                      # the same trap the lpf_lim cache-key fix documents in
	                      # _fft_pickle_path.
	                      #
	                      # 3->4: THREE more content changes, same reasoning.
	                      # (a) a window that cannot be filled with real samples
	                      # is no longer computed on partial data -- the old
	                      # n >= n_expected // 2 branch is gone, so every row now
	                      # stands for a full 2*pad window wherever it sits.
	                      # (b) the LAST file's successor is taken from the CDFs
	                      # on disk instead of from the request, so a file's FFT
	                      # no longer depends on which request computed it
	                      # (pair_with_next_cdf). (c) "filled" is measured from
	                      # the file's own timestamps (n_full), not assumed from
	                      # the nominal rate: the clock runs ~14 ppm slow, so a
	                      # gap-free 0.5 s window really holds 4095 OR 4096
	                      # samples by phase, and testing against the nominal
	                      # 4096 silently dropped 6.1% of gap-free windows.
	'edp': 5,             # 2->3: computed from quality-filtered data pickles now
	                      # 3->4: same full-spectrum + SCB_FS (now SAMPLE_RATE) change as scm
	                      # 4->5: same three window-fill changes as scm 3->4
	'fgm': 3,             # 2->3: same three window-fill changes as scm 3->4
}

# Lowest EDP quality tier accepted as real data. Per the EDP Data Products Guide
# v1.3 Table 4: 3 good / 2 OK, use with some caution / 1 bad, use with caution /
# 0 very bad or no data.
#
# 2 (not 3) because Table 5 of that guide maps every bit to the tier it forces,
# and only TWO conditions produce a 2: bit 6 "ASPOC is emitting non-zero
# current" and bit 8 "asymmetric probe configuration (e.g. MMS4 after
# 2016-06-12)". Neither corrupts a sample -- ASPOC is the ion emitter that
# controls spacecraft charging, so it biases the DC spacecraft-potential
# baseline a double probe measures against, not AC wave content at hundreds of
# Hz. Every tier-1 cause IS a genuine defect (bad bias settings, saturation from
# low density, probe in the ADP boom shadow, bias sweeps, ongoing maneuver), and
# tier 0 is disabled or saturated probes, so 1 and 0 stay excluded.
#
# This is not a hypothetical: on 2021-02-23 and 2021-03-02 -- both dates in
# plot_biggest_spikes.REQUESTS -- 298 of 300 EDP burst files contain ZERO
# tier-3 samples, because ASPOC ran the whole time (bitmask == 64 on every
# sample of the 07:46:53 burst, 3,614,822 of them). Under the old == 3 test
# every one of those files produced an empty data pickle, an empty FFT pickle,
# and no E panel at all -- indistinguishable from "no burst data here".
#
# SCM needs no equivalent knob: its qf has only 'GGG' and 'ZZZ' on this data,
# and the 'ZZZ' samples are literal all-zero readings from the L1B->L2 filter's
# zero-padded convolution edge (verified: 32768 rows, every component exactly
# 0.0), so qf == 'GGG' already means "not malformed".
EDP_MIN_QUALITY = 2

# Bumped 1->2: a large share of the cache (957/1186 files, going back to 2017)
# turned out to be unreadable under this project's real pandas (2.3.3, this
# venv) -- those pickles were written by a pandas 3.0.2 environment (a WSL
# testing env used for verification), whose newer string-dtype pickle format
# pandas 2.3.3 can't deserialize (NotImplementedError in
# NDArrayBacked.__setstate__). The version bump forces every pickle to be
# treated as stale and regenerated fresh the next time it's touched, by
# whichever environment actually runs the code -- no manual file deletion
# needed, and no false negatives from pickles that already carried a
# version=1 sidecar despite being unreadable here.

import numpy as np
import pandas as pd

# matplotlib is NOT imported here. Every plotting use in this module lives
# inside gen_fft, which imports it locally -- see the note below on why module
# scope is the wrong place for it.
#
# scipy was imported here too (`import scipy`, `from scipy.signal import
# windows`, `from scipy.stats import binned_statistic_2d`) and NONE of the
# three was ever referenced: the FFT window is np.hanning (see
# _compute_file_ffts), and the other two had no use site at all. That mattered
# far more than a normal dead import, because of how this module is consumed:
#
# ProcessPoolExecutor on Windows uses the 'spawn' start method, so every
# worker process re-imports the entry-point module and its whole dependency
# tree from scratch before it can run a single job. Importing scipy.signal
# costs ~100 MB of commit per process; at 8 workers that is ~800 MB of address
# space reserved purely to satisfy an import nothing called. On a machine
# whose pagefile could not cover it, the workers died during import with
# "DLL load failed ... The paging file is too small for this operation to
# complete", which surfaced only as BrokenProcessPool in the parent -- an
# error that says nothing about its own cause.
#
# The rule this leaves behind: in a module that spawned workers import, a
# module-scope import of a heavy library is paid N times over, whether or not
# any worker uses it. Keep such imports inside the function that needs them.
# The burst sample rate, in Hz, selected once from the BurstRate enum in
# spectro/instrument_spec.py -- see the note there for why it is 8192 and not the
# 8191.887 the timestamps step at. Light import (dataclasses, enum, numpy).
try:
	from spectro.instrument_spec import B_BURST, BurstRate, E_SAMPLE_RATE, SAMPLE_RATE, field_burst
except ImportError:
	from src.spectro.instrument_spec import B_BURST, BurstRate, E_SAMPLE_RATE, SAMPLE_RATE, field_burst

# Worker count for every ProcessPoolExecutor in this module. Each worker is a
# fresh interpreter that re-imports numpy + pandas (+ whatever the entry point
# pulls in), so this multiplies memory, not just CPU. Windows defaults lower
# because 'spawn' pays that import cost per process where fork-based platforms
# largely do not. Override with PLASMA_MAX_WORKERS to trade memory for speed
# in either direction -- lower it if you see BrokenProcessPool.
MAX_POOL_WORKERS = int(os.environ.get('PLASMA_MAX_WORKERS', 4 if os.name == 'nt' else 8))


def _pool_map(fn, jobs, max_workers, what):
	"""ProcessPoolExecutor.map with a diagnosable failure.

	A worker that dies during its own import (see the scipy note above) reaches
	the parent only as BrokenProcessPool, which names neither the cause nor the
	fix. This turns it into an error that does.
	"""
	try:
		with ProcessPoolExecutor(max_workers=max_workers) as pool:
			return list(pool.map(fn, jobs))
	except BrokenProcessPool as exc:
		raise RuntimeError(
			f'{what}: a worker process died before returning a result '
			f'(max_workers={max_workers}). The usual cause is memory: each worker '
			f'is a separate interpreter that re-imports this module tree, so N '
			f'workers cost roughly N times one process. Retry with a smaller pool '
			f'(PLASMA_MAX_WORKERS=2) or raise the Windows pagefile.'
		) from exc


def _pool_imap(fn, jobs, max_workers, what, cancelled=None, ahead=None, poll_s=0.25):
	"""fn(job) for each job on a process pool, YIELDED in job order, and stoppable (2026-11-23, Graham: "the
	program should wait for the workers to finish, delete their output, then cancel mid run").

	At most `ahead` jobs (default two per worker) are handed to the pool beyond the one being waited for, so the
	results waiting in this process stay bounded however many jobs there are. cancelled() is asked between
	results and every poll_s while one is awaited; once it is true nothing more is submitted, the jobs not yet
	started are cancelled, the ones a worker already holds are waited for (their results dropped), and the
	generator returns. The pool is always shut down and its workers joined before this returns or raises, so no
	worker outlives the call. A worker that dies is reported as _pool_map reports it."""
	from collections import deque
	from concurrent.futures import wait as _wait
	ahead = ahead or 2 * max_workers
	jobs = iter(jobs)
	pending = deque()
	pool = ProcessPoolExecutor(max_workers=max_workers)
	try:
		def fill():
			while len(pending) < ahead:
				for job in jobs:
					pending.append(pool.submit(fn, job))
					break
				else:
					return
		fill()
		while pending:
			head = pending[0]
			while True:
				if cancelled is not None and cancelled():
					return
				if head.done():
					break
				_wait((head,), timeout=poll_s)
			pending.popleft()
			result = head.result()
			fill()
			yield result
	except BrokenProcessPool as exc:
		raise RuntimeError(
			f'{what}: a worker process died before returning a result (max_workers={max_workers}). The usual '
			f'cause is memory -- see _pool_map. Retry with a smaller pool (PLASMA_MAX_WORKERS=2) or raise the '
			f'Windows pagefile.') from exc
	finally:
		pool.shutdown(wait=True, cancel_futures=True)


def _bounds_key(inst):
	"""Map a discovery instrument name to the key cdf_time_bounds expects."""
	return 'E' if inst == 'edp' else ('F' if inst == 'fgm' else 'B')


def _do_download(inst, mode, probe, trange, data_dir):
	"""
	Runs in its own subprocess (see attempt_download) so a stalled call can be
	killed outright on timeout instead of just abandoned-but-still-running.
	Top-level (picklable) function, required for multiprocessing.Process with
	the 'spawn' start method Windows uses.
	"""
	import time as _time_mod  # TEMP diagnostic (see prints below) -- remove once the B->E download gap is root-caused
	_t_import0 = _time_mod.monotonic()
	try:
		try:
			from pyspedas import mms          # pyspedas < ~2.x: mms at the top level
		except ImportError:
			from pyspedas.projects import mms  # pyspedas >= ~2.x: moved under .projects
		try:
			from pytplot import del_data      # older releases: separate pytplot package
		except ImportError:
			from pyspedas import del_data      # newer releases: bundled into pyspedas itself
	except ImportError:
		return
	print(f'[_do_download] {inst} {trange} -- pyspedas/pytplot import took {_time_mod.monotonic() - _t_import0:.2f}s', flush=True)  # TEMP diagnostic

	# pyspedas has its OWN notion of where to save data (mms_config.CONFIG
	# ['local_data_dir']), completely independent of data_dir/DATA_ROOT --
	# and it defaults to the bare relative string 'pydata', resolved against
	# whatever this subprocess's CWD happens to be. Left alone, a download can
	# silently land somewhere find_cdf_files/ensure_data_pickle will never
	# look (they always scan the absolute data_dir passed in), so it looks
	# like it succeeded but is never discovered afterward. Force it to match
	# data_dir explicitly every time, in this subprocess, before downloading.
	try:
		try:
			from pyspedas.mms_config import CONFIG as mms_cfg  # pyspedas < ~2.x
		except ImportError:
			from pyspedas.projects.mms.mms_config import CONFIG as mms_cfg  # pyspedas >= ~2.x
		mms_cfg['local_data_dir'] = data_dir
	except Exception:
		pass  # best-effort; a failed override still leaves the download attempt itself intact

	_t_call0 = _time_mod.monotonic()  # TEMP diagnostic
	try:
		# datatype=mode ties the download to the exact same mode value
		# discovery uses (_scan_cdf_files' mms{probe}/{inst}/brst/l2/{mode}
		# glob) -- without it, pyspedas falls back to its own internal
		# default datatype, which for SCM brst data turned out to be schb
		# (high-burst, 16384 Hz), not scb (burst, 8192 Hz) as intended.
		#
		# probe is a list of probe-id strings here (even for a single probe) --
		# pyspedas logs in once per call and loops over probes internally
		# using that same session, so one call with N probes costs one login,
		# not N.
		if inst == 'edp':
			mms.edp(probe=probe, trange=trange, time_clip=True, data_rate='brst', datatype=mode, latest_version=True, no_update=False)
		elif inst == 'fgm':
			mms.fgm(probe=probe, trange=trange, time_clip=True, data_rate='brst', datatype=mode, latest_version=True, no_update=False)
		else:
			mms.scm(probe=probe, trange=trange, time_clip=True, data_rate='brst', datatype=mode, latest_version=True, no_update=False)
		del_data()
	except Exception as e:
		print(f'Download attempt for {inst} {trange} failed: {type(e).__name__}: {e}')
	print(f'[_do_download] {inst} {trange} -- mms.{inst}()+del_data() took {_time_mod.monotonic() - _t_call0:.2f}s', flush=True)  # TEMP diagnostic


def attempt_download(inst, mode, probe, start, end, data_dir):
	"""
	Best-effort pyspedas download of [start, end] for one instrument and one
	or more probes.

	data_dir must be the SAME absolute path the caller scans afterward (e.g.
	find_cdf_files' own data_dir) -- it's forwarded to _do_download so pyspedas
	saves into exactly that directory instead of its own cwd-relative default.

	`probe` may be a single probe (int or str) or a list of probes -- passing
	a list downloads all of them in a single pyspedas call (one login, one
	SDC file-listing check) instead of one call per probe.

	Serialized process-wide via _download_lock -- only one download runs at a
	time, regardless of which thread or call site triggers it. Running
	multiple downloads concurrently has been slower in practice, not faster;
	on top of that, pyspedas's MMS loaders load into a shared global tplot
	store before we call del_data() to clean up, which isn't safe to do from
	two threads at once. The concurrency win should come from overlapping
	ONE download with other work (computation, plotting) happening in
	another thread, not from running several downloads simultaneously.

	The actual pyspedas call runs in a subprocess (_do_download) bounded by
	DOWNLOAD_TIMEOUT_S. If it doesn't finish in time, the subprocess is
	killed and this returns anyway -- a hang somewhere upstream of pyspedas's
	own per-file timeouts (e.g. a swallowed connection) previously blocked
	the whole pipeline for hours with the lock held the entire time.

	Network/archive failures (including pyspedas/pytplot not being installed)
	are swallowed -- the caller re-scans the disk regardless and treats
	"nothing new landed" as a real absence of data, not a crash.

	Public (no leading underscore) because callers outside this module can
	use it directly to prefetch a range in a background thread ahead of when
	find_cdf_files would otherwise block on it -- see plot_biggest_spikes.py.
	"""
	import time as _time_mod  # TEMP diagnostic (see prints below) -- remove once the B->E download gap is root-caused
	_t0 = _time_mod.monotonic()
	probe_list = [str(p) for p in probe] if isinstance(probe, (list, tuple)) else [str(probe)]
	trange = [start.strftime('%Y-%m-%d/%H:%M:%S'), end.strftime('%Y-%m-%d/%H:%M:%S')]

	# Every SDC call goes through acquisition's own fetch (Graham, 2026-10-01: "All
	# files should be checked that they don't just download as rate-limit pages, and
	# if they are, delete them and make the amount of time between requests longer").
	# _fetch takes _download_lock, waits out the shared pacer's gap, repeats a
	# refused listing, and lets what the call delivered set the next gap; a 429 page
	# it delivered is deleted here, uncounted, as acquire() deletes one. One call per
	# probe, as acquire() makes them (Graham, 2026-09-28). Imported here because
	# acquisition imports this module.
	import acquisition
	_t_lock0 = _time_mod.monotonic()  # TEMP diagnostic -- separates lock-wait from spawn+run
	letter = {'scm': 'B', 'edp': 'E', 'fgm': 'F'}.get(inst, inst)
	for p in probe_list:
		result = acquisition.Acquired(letter, (int(p),), start, end)
		# The settings file's source: CDAWeb, 8 files at a time, unless it says 'sdc' (Graham, 2026-10-06:
		# "Switch ALL downloads to CDAWeb at 8 at a time").
		_status, arrived = acquisition._fetch(acquisition._default_downloader(), inst, mode, (int(p),), start, end,
		                                      data_dir, result, lambda: False, print)
		for path in sorted(q for paths in arrived.values() for q in paths):
			if acquisition._is_rate_limit_page(acquisition._head(path)):
				acquisition._remove(path, 'an SDC rate-limit (429) page, not counted', print)
	print(f'[attempt_download] {inst} {trange} -- lock wait {_t_lock0 - _t0:.2f}s, spawn+run {_time_mod.monotonic() - _t_lock0:.2f}s, total {_time_mod.monotonic() - _t0:.2f}s', flush=True)  # TEMP diagnostic


# ---------------------------------------------------------------------------
# the preprocessed archive (preproc_archive): every reader of a B or E burst CDF asks it first
# ---------------------------------------------------------------------------
#
# An archived CDF keeps its path (preproc_archive.virtual_path): _scan_cdf_files lists it where it was, and
# each reader below takes the archive's row when the archive is on and holds one for that path, whether or not
# the CDF is still on disk. The row holds exactly what the CDF readers take from the file (every record's time,
# the quality-good mask, and |field| at each good record), so a reader gives the same values either way. With
# the archive off (the default) none of this opens anything, and every reader is what it was.

def _archive_on():
	import preproc_archive
	return preproc_archive.enabled()


def _archived_row(path, key, signal=False):
	"""preproc_archive.Archived for this B or E burst CDF path, or None: the archive off, a data pickle, FGM, or
	no verified current row for the path."""
	if key[0] not in ('B', 'E') or str(path).endswith('.pkl') or not _archive_on():
		return None
	import preproc_archive
	return preproc_archive.load(path, signal=signal)


def _archived_only(path, key):
	"""True for a B or E burst CDF path the archive stands for while the CDF itself is gone."""
	if key[0] not in ('B', 'E') or not _archive_on() or os.path.exists(path):
		return False
	import preproc_archive
	return preproc_archive.archived_only(path) is not None


@lru_cache(maxsize=1)
def _pydatetime_unit():
	"""The unit pd.to_datetime gives a column of Python datetimes in this pandas ('us' in pandas 3, 'ns'
	before): _epoch_datetime64's unit, so times built from the archive match the CDF readers' exactly."""
	return pd.to_datetime(np.array([datetime(2015, 9, 1)])).unit


def _archive_time_index(ns):
	"""int64 ns times as the DatetimeIndex _epoch_datetime64 returns for the same samples."""
	return pd.DatetimeIndex(np.asarray(ns, dtype=np.int64).view('datetime64[ns]')).as_unit(_pydatetime_unit())


def _source_mtime(src):
	"""os.path.getmtime(src); for a burst CDF the archive holds after its deletion, the mtime that CDF had when it
	was archived -- so a cache built from it keeps going stale exactly as it did while the CDF was on disk.
	Raises OSError when neither is there."""
	try:
		return os.path.getmtime(src)
	except OSError:
		if not _archive_on():
			raise
		import preproc_archive
		meta = preproc_archive.archived_only(src)
		if meta is None:
			raise
		return meta['cdf_mtime_ns'] / 1e9


def _scan_cdf_files(data_dir, inst, mode, probe):
	"""Sorted [(start_time, path)] for every CDF matching inst/mode/probe: on disk, and -- when the archive is
	on -- every B or E file the archive holds, at the path it had (preproc_archive.virtual_path). A name both on
	disk and archived is listed once, at its on-disk path."""
	pattern = os.path.join(data_dir, f'mms{probe}/{inst}/brst/l2/{mode}/**/*.cdf')
	files = []
	for path in glob.glob(pattern, recursive=True):
		m = re.search(r'_(\d{14})_', path)
		if m:
			ftime = datetime.strptime(m.group(1), '%Y%m%d%H%M%S')
			files.append((ftime, path))
	archived = []
	if inst in ('scm', 'edp') and _archive_on():
		import preproc_archive
		on_disk = {os.path.basename(p) for _t, p in files}
		archived = [(t, p) for t, p in preproc_archive.scan(data_dir, inst, mode, probe)
		            if os.path.basename(p) not in on_disk]
	if archived:
		# Sorted on the name, not the whole path: an archived path and an on-disk one may join their folders with
		# different separators on Windows, and the later data version of a stamp must still sort last.
		files += archived
		files.sort(key=lambda fp: (fp[0], os.path.basename(fp[1])))
	else:
		files.sort()
	return files


def _anchor_index(files, start):
	"""Index of the rightmost file whose filename timestamp is <= start, or None."""
	for i in range(len(files) - 1, -1, -1):
		if files[i][0] <= start:
			return i
	return None


def _fall_forward(files, start):
	"""Index of the leftmost file whose filename timestamp is > start, or None
	if no such file exists. Shared fallback for when _anchor_index finds
	nothing <= start -- used by both the verified (_resolve_anchor,
	download=True) and cheap (_pick_start_anchor, download=False) paths, so
	they agree on this case. `start` landing before a day's first file (e.g.
	an exact-midnight request whose first burst that day begins a few hours
	in) is NOT evidence of missing data -- confirmed bug: before this, the
	cheap path had no fallback at all and treated it as one, reporting a
	whole instrument "missing" for a probe whose data was sitting right
	there, just starting a bit later than the requested window's start."""
	return next((i for i, (ftime, _) in enumerate(files) if ftime > start), None)


def _resolve_anchor(scan, start, bounds_resolver, repair=None):
	"""
	Generic anchor selection, shared by CDF discovery (which can repair a gap
	via download) and pickle-cache discovery (local-only, no repair).

	scan()             -> fresh sorted [(ftime, path), ...]
	bounds_resolver(p) -> (real_start, real_end) or (None, None) if unreadable
	repair(gap_start)  -> optional; attempt to fill data from gap_start through
	                       the caller's own known end (e.g. attempt_download
	                       over the whole remaining request range). Called at
	                       most once. Deliberately given only the start of the
	                       gap, not a guessed narrow sub-range: the local scan
	                       can't tell us whether other real gaps exist further
	                       in (e.g. only a handful of files already known,
	                       scattered across a much bigger requested interval)
	                       -- bounding repair to "whatever's next on disk"
	                       would silently skip any real archive data beyond
	                       that point. Re-requesting the full remaining span
	                       is simpler and correct in all cases; pyspedas skips
	                       files it already has, so this isn't wasteful.

	A filename timestamp alone doesn't prove a file's data actually reaches
	`start`: the naive anchor (rightmost file with ftime <= start) could be the
	last file before a real, arbitrarily long gap just as easily as a stale
	leftover from an unrelated, much earlier request/era -- the same risk
	whether the files in question are raw CDFs or cached data pickles. This
	verifies the naive anchor's TRUE recorded end against `start` via
	bounds_resolver; if it falls short and a `repair` is given, tries once to
	fill from `start` onward and re-scans. Either way, a still-confirmed gap
	falls forward to the next real file after `start` rather than trusting
	the stale one.

	Returns (anchor_index_or_None, files) using the latest scan.
	"""
	files = scan()
	if not files:
		return None, files

	def _verified(files):
		anchor = _anchor_index(files, start)
		real_end = bounds_resolver(files[anchor][1])[1] if anchor is not None else None
		return anchor if (anchor is not None and real_end is not None and real_end >= start) else None

	anchor = _verified(files)
	if anchor is not None:
		return anchor, files

	if repair is not None:
		repair(start)
		files = scan()
		anchor = _verified(files)
		if anchor is not None:
			return anchor, files

	# Confirmed gap (or no repair possible): fall forward to the next real file after `start`.
	return _fall_forward(files, start), files


def _pick_start_anchor(data_dir, inst, mode, probe, start, end, download, check=None):
	"""
	Decide which CDF anchors coverage of `start`. Thin CDF-specific wiring over
	the shared _resolve_anchor primitive: supplies the CDF scan, the CDF bounds
	resolver (cdf_time_bounds), and -- only when `download` is True -- a repair
	callback that fills a confirmed gap via attempt_download (widened by
	GAP_FILL_LOOKBACK on the start side) before falling back to whatever's on
	disk. Repair always requests through the request's own `end`, not a
	guessed narrower sub-range -- a local scan can't rule out further real
	gaps between the naive anchor and `end` (e.g. only a few files already
	known, scattered across a much bigger requested interval), so bounding
	the repair download to "whatever's next on disk" would silently skip any
	archive data past that point.

	check: see find_cdf_files. The anchor's bounds are read only once it has
	validated; one that does not is deleted and answers "no data", so the repair
	below refetches it -- or, still bad, the anchor falls forward past it.

	Returns (anchor_index_or_None, files).
	"""
	scan = lambda: _scan_cdf_files(data_dir, inst, mode, probe)
	if not download:
		files = scan()
		anchor = _anchor_index(files, start)
		if anchor is None:
			anchor = _fall_forward(files, start)
		return anchor, files

	key = _bounds_key(inst)
	bounds_resolver = lambda p: (cdf_time_bounds(p, key=key, probe=probe)
	                             if check is None or check([p]) else (None, None))

	def repair(gap_start):
		attempt_download(inst, mode, probe, gap_start - GAP_FILL_LOOKBACK, end, data_dir)

	return _resolve_anchor(scan, start, bounds_resolver, repair=repair)


def find_cdf_files(data_dir, inst, mode, start, end, probe: int = 1, download: bool = True,
                   check=None):
	"""
	Return a sorted list of CDF paths under data_dir/mms{probe}/{inst}/brst/l2/{mode}
	covering [start, end].

	Scans all files, sorts by timestamp, then anchors to the rightmost file whose
	start time is <= start (see _pick_start_anchor). All files from that anchor up
	through end are returned. When `download` is True (the default), a matching
	check runs at the `end` boundary too: if the last included file's true
	recorded data falls short of `end`, one targeted download is tried for
	exactly that gap before accepting whatever is on disk.

	check: a cdf_check(...). Given one, discovery opens nothing it has not
	validated -- the anchor and end-boundary bounds reads included -- and returns
	only files that validated, the rejected ones deleted, logged and counted. Every
	pipeline caller passes one; None keeps the plain filename listing, which opens
	nothing when download is False.
	"""
	def validated(pairs):
		if check is None or not pairs:
			return pairs
		ok = set(check([path for _ftime, path in pairs]))
		return [(ftime, path) for ftime, path in pairs if path in ok]

	if download:
		attempt_download(inst, mode, probe, start, end, data_dir)

	anchor, all_files = _pick_start_anchor(data_dir, inst, mode, probe, start, end, download,
	                                       check=check)
	if anchor is None:
		return []

	# Validated BEFORE the end check below reads the last file's bounds.
	in_range = validated([(ftime, path) for ftime, path in all_files[anchor:] if ftime <= end])

	if download and in_range:
		key = _bounds_key(inst)
		last_ftime, last_path = in_range[-1]
		last_end = cdf_time_bounds(last_path, key=key, probe=probe)[1]
		post_end = next(((ftime, path) for ftime, path in all_files if ftime > end), None)
		if last_end is not None and last_end < end and post_end is not None:
			attempt_download(inst, mode, probe, last_end, post_end[0], data_dir)
			all_files = _scan_cdf_files(data_dir, inst, mode, probe)
			anchor = _anchor_index(all_files, start)
			if anchor is not None:
				in_range = validated([(ftime, path) for ftime, path in all_files[anchor:] if ftime <= end])

	return [path for ftime, path in in_range]


def read_data_pickle(path):
	"""Load a per-file data pickle, normalizing 'time' to datetime64[ns].

	The pickle cache spans sessions run under different pandas/numpy versions
	-- some older cached pickles have 'time' stored as datetime64[us], not
	[ns]. Downstream code (_compute_file_ffts) takes a raw
	`.values.astype(np.int64)` view of that column and treats it as
	nanoseconds unconditionally, since that's what a fresh datetime64[ns]
	column gives you. On a [us] column, the same cast silently returns
	MICROSECONDS instead -- every timestamp ends up ~1000x too small, which
	surfaced as `ValueError: negative dimensions are not allowed` (the file's
	own real data looked like it started near the 1970 epoch, wildly earlier
	than its own nominal file_time). Normalizing once, at this single read
	choke point, means every downstream raw int64 cast is safe regardless of
	which pandas version originally wrote a given cached pickle.
	"""
	df = pd.read_pickle(path)
	if not df.empty and df['time'].dtype != 'datetime64[ns]':
		df['time'] = df['time'].astype('datetime64[ns]')
	return df


def _pickle_time_bounds(path):
	"""(first, last) timestamp in a data pickle's 'time' column, or (None, None)
	if unreadable/empty. Cheap -- pickles are already local, no CDF parse."""
	try:
		df = read_data_pickle(path)
		if df.empty:
			return None, None
		t = df['time']
		return t.iloc[0].to_pydatetime(), t.iloc[-1].to_pydatetime()
	except Exception:
		return None, None


def _segments_from_pickle(pickle_path, fs, gap_factor=None):
	"""Contiguous good-data runs [(start, end), ...] recovered from an
	already-good-filtered data pickle, WITHOUT opening the CDF -- the
	trust-the-cache counterpart to _scm_good_segments / _edp_good_segments.

	Both instruments' data pickles contain ONLY quality-good samples (SCM
	qf == 'GGG', EDP quality >= EDP_MIN_QUALITY -- see _cdf_to_df): the bad stretches are
	physically absent from the file, not stored as gaps, zeros, or any kind of
	placeholder row. So the runs are just the surviving samples, and the
	boundaries between them are wherever consecutive timestamps sit further
	apart than one sample period at the instrument's sample rate `fs`. Nothing
	about where the holes are needs storing -- the timestamps that remain
	already say it.

	`gap_factor` (default 1.5) is the margin on that 1/fs threshold: big enough
	to absorb timestamp jitter between genuinely consecutive samples, small
	enough that the tightest real hole still trips it (dropping even a single
	sample leaves a 2/fs spacing).
	"""
	try:
		df = read_data_pickle(pickle_path)
	except Exception:
		return []
	if df.empty:
		return []

	t = df['time'].values.astype(np.int64)
	if len(t) == 1:
		only = pd.Timestamp(t[0])
		return [(only, only)]

	gap_factor = GOOD_RUN_GAP_FACTOR if gap_factor is None else gap_factor
	max_contiguous_ns = gap_factor * 1e9 / fs
	dt = np.diff(t)
	breaks = np.flatnonzero(dt > max_contiguous_ns)
	starts = np.concatenate(([0], breaks + 1))
	ends   = np.concatenate((breaks, [len(t) - 1]))
	return [(pd.Timestamp(t[s]), pd.Timestamp(t[e])) for s, e in zip(starts, ends)]


def find_data_pickles(pickle_root, inst, mode, start, end, probe: int = 1, verify: bool = True):
	"""
	Same discovery logic as find_cdf_files but for per-file data pickles, wired
	into the same _resolve_anchor primitive -- just with a pickle-glob scan, a
	pickle bounds resolver (_pickle_time_bounds), and no repair (this is a local
	cache scan, not a discovery+fetch path; a confirmed gap just falls forward
	to whatever real data exists next).

	Returns sorted list of (file_start_time, path) pairs. Pickle filenames
	follow the convention: {YYYY}/{MM}/{DD}/{HHMMSS}.pkl

	This scans EVERY pickle ever cached for this probe/inst/mode -- potentially
	years of history -- not just the ones from one request. verify=True
	(default) is what protects against the naive anchor landing on a stale
	pickle left over from a wholly unrelated earlier request/era -- the exact
	bug class find_cdf_files' anchor already guarded against, now shared code.
	"""
	def scan():
		pattern = os.path.join(pickle_root, f'mms/{probe}/{inst}/{mode}/*/*/*/*.pkl')
		files = []
		for path in glob.glob(pattern):
			m = re.search(r'(\d{4})[/\\](\d{2})[/\\](\d{2})[/\\](\d{6})\.pkl$', path)
			if m:
				ftime = datetime.strptime(''.join(m.groups()), '%Y%m%d%H%M%S')
				files.append((ftime, path))
		files.sort()
		return files

	if verify:
		anchor, all_files = _resolve_anchor(scan, start, _pickle_time_bounds)
	else:
		all_files = scan()
		anchor = _anchor_index(all_files, start)

	if anchor is None:
		return []

	return [(ftime, path) for ftime, path in all_files[anchor:] if ftime <= end]


# TT2000 counts nanoseconds from J2000 in TT. Between leap seconds, UTC is that
# count plus a constant: this instant -- 2000-01-01T11:58:55.816 UTC, J2000 when
# TAI-UTC was 32 s -- in Unix nanoseconds, less 1 s per leap second since.
_J2000_UTC_NS = 946_727_935_816_000_000


def _epoch_datetime64(cdf, epoch_var):
	"""pd.to_datetime(cdf[epoch_var][:]) -- same values, same dtype and unit --
	without building one Python datetime per sample.

	That per-sample build (pycdf's tt2000_to_datetime, one ctypes call each) was
	~95% of _cdf_to_df's time: 2.8 of 3.0 s on a 1M-sample SCM file and 9.7 of
	11 s on a 3.7M-sample one (measured 2026-09-14 on E:, Windows venv). Between
	leap seconds TT2000 -> UTC is one constant offset, so the column converts with
	an integer add, rounded to the microsecond exactly as pycdf rounds
	(nsec/1000 + 0.5 -- truncating instead disagreed on half the samples).
	Verified bit-identical on B, E and FGM burst files, 18-26x faster.

	Only the first and last samples go through pycdf: they give the offset and
	the datetime unit this pandas produces. The original full conversion runs
	instead when the column is empty, holds the TT2000 fill value, or its two
	ends disagree on the offset (a leap second inside the file).
	"""
	raw = np.asarray(cdf.raw_var(epoch_var)[:], dtype=np.int64)
	if raw.size == 0 or raw.min() == np.iinfo(np.int64).min:
		return pd.to_datetime(cdf[epoch_var][:])
	var = cdf[epoch_var]
	ends = pd.to_datetime(np.concatenate([np.asarray(var[0:1]), np.asarray(var[raw.size - 1:raw.size])]))
	offsets = {int(round((int(e) - int(r) - _J2000_UTC_NS) / 1e9))
	           for e, r in zip(ends.as_unit('ns').asi8, raw[[0, -1]])}
	if len(offsets) != 1:
		return pd.to_datetime(cdf[epoch_var][:])
	utc = raw + (_J2000_UTC_NS + offsets.pop() * 1_000_000_000)
	utc = ((utc + 500) // 1000) * 1000
	out = pd.DatetimeIndex(utc.view('datetime64[ns]')).as_unit(ends.unit)
	if out[0] != ends[0] or out[-1] != ends[-1]:
		return pd.to_datetime(cdf[epoch_var][:])
	return out


def _cdf_to_df(cdf_path, key, probe: int = 1, n_records=None):
	"""Load a single CDF file into a DataFrame.

	B/E: columns time, {f}x, {f}y, {f}z (vector field for the FFT path).
	F (FGM background field): columns time, Bmag — the |B| magnitude the
	electron cyclotron frequency depends on. FGM stores [Bx, By, Bz, |B|] in
	the gse vector; we keep only column 3 (the precomputed magnitude, nT).

	SCM (B) quality filtering: the CDF's own `..._qf_scm123_...` variable
	(CATDESC "Quality Factor") flags each sample 'G'=good per antenna, or
	'Z'/'S'/'X'/'B' for convolution-boundary/saturated/out-of-range/fillvalue
	data -- confirmed via the variable's VAR_NOTES and cross-checked against
	15 real files (all 4 probes, two dates): every non-'GGG'-flagged sample
	exactly matched a literal all-zero Bx/By/Bz reading, a 1:1 correspondence,
	not a coincidence. Typically the first ~16384-32768 samples of a file
	right after a burst-trigger boundary, from the L1B->L2 frequency filter's
	zero-padded convolution edge (see the CDF's FREQUENCY_FILTER global attr).
	Dropped here so no downstream FFT window is ever computed over this
	instrument artifact.

	EDP (E) quality filtering: the symmetric filter on the EDP side, using that
	instrument's own `..._edp_quality_...` variable (== 3, "good") -- the exact
	same tier _edp_good_segments already treats as good, so the two instruments
	now agree on what "good data" means and a data pickle contains ONLY good
	samples for either field. EDP shows no all-zero convolution-edge artifact
	the way SCM does (which is why this filter wasn't here originally), but the
	quality flag marks other unusable spans -- 2 is "use with caution", 1 is
	bad, 0 is no data -- and those were previously flowing straight into the E
	FFTs while the manifest reported them as excluded. Filtering here is what
	makes a good-only pickle the single source of truth for BOTH fields: the
	contiguous good runs are recoverable from the surviving time column alone
	(see _segments_from_pickle), so no separate qf/quality artifact has to be
	stored or kept in sync.

	n_records: read only the file's first n_records records, every variable
	alike -- read_burst_data's seam read. None reads the whole file.
	"""
	from spacepy import pycdf
	f = key[0]
	with pycdf.CDF(cdf_path) as cdf:
		if n_records is not None:
			cdf = _CdfRecords(cdf, 0, n_records)
		if f == 'F':
			var  = cdf[f'mms{probe}_fgm_b_gse_brst_l2'][:]   # (N, 4): Bx,By,Bz,|B| (nT)
			time = _epoch_datetime64(cdf, 'Epoch')
			return pd.DataFrame({'time': time, 'Bmag': var[:, 3].copy()})
		time, var, good = _burst_records(cdf, cdf_path, key, probe)
		var, time = var[good], time[good]
	return pd.DataFrame({
		'time': time,
		f'{f}x': var[:, 0].copy(),
		f'{f}y': var[:, 1].copy(),
		f'{f}z': var[:, 2].copy(),
	})


def _burst_records(cdf, cdf_path, key, probe):
	"""(time, var, good) of EVERY record of an open B or E burst CDF, before the
	quality filter: the epoch times, the (N, 3) field vectors and the quality-good
	mask. _cdf_to_df keeps var[good], time[good]; preproc_archive.preprocess takes
	all three from this one read, so the variables read and the quality test live
	here alone."""
	if key[0] == 'B':
		mode = _scm_mode(cdf_path)
		var  = cdf[f'mms{probe}_scm_acb_gse_{mode}_brst_l2'][:]
		time = _epoch_datetime64(cdf, 'Epoch')
		qf_var = f'mms{probe}_scm_qf_scm123_{mode}_brst_l2'
		try:
			# raw_var: see _qf_good_mask -- same ~30x-faster fix as
			# _scm_good_segments, applied here too since this is the
			# actual per-file FFT data load, called far more often (once
			# per file the FFT pipeline ever touches) than the
			# manifest-only _scm_good_segments.
			good = _qf_good_mask(cdf.raw_var(qf_var)[:])
		except Exception:
			qf = np.asarray(cdf[qf_var][:])
			good = qf == 'GGG'
	else:
		var  = cdf[f'mms{probe}_edp_dce_gse_brst_l2'][:]
		time = _epoch_datetime64(cdf, f'mms{probe}_edp_epoch_brst_l2')
		# quality is a small numeric type (CDF_UINT1), not CDF_CHAR like
		# SCM's qf -- no raw_var/string-marshalling fast path needed here
		# (see _edp_good_segments), a plain read is already cheap.
		quality = np.asarray(cdf[f'mms{probe}_edp_quality_brst_l2'][:])
		good = quality >= EDP_MIN_QUALITY
	return time, var, good


def cdf_time_bounds(cdf_path, key='B', probe: int = 1):
	"""Return (first, last) recorded sample times of a CDF as Timestamps.

	Reads ONLY the first and last Epoch records, not the whole array, so it is
	cheap. This is the source of E(file): the true end of a file's data. The
	filename does not give it — burst files are not contiguous, so the next
	file's start overestimates this end. Returns (None, None) if empty.
	"""
	from spacepy import pycdf
	f = key[0]
	a = _archived_row(cdf_path, key)
	if a is not None:
		if a.recorded.size == 0:
			return None, None
		ends = _archive_time_index(a.recorded[[0, -1]])
		return ends[0], ends[1]
	epoch_var = 'Epoch' if f in ('B', 'F') else f'mms{probe}_edp_epoch_brst_l2'
	with pycdf.CDF(cdf_path) as cdf:
		ev = cdf[epoch_var]
		n = len(ev)
		if n == 0:
			return None, None
		first = pd.to_datetime(ev[0])
		last  = pd.to_datetime(ev[n - 1])
	return first, last


def _qf_good_mask(raw):
	"""True where a raw (untranslated) qf record reads as 'GGG', given
	whatever spacepy's CDF.raw_var() hands back for a CDF_CHAR variable --
	see _scm_good_segments. raw_var() skips pycdf's normal CHAR-type
	Unicode-decode-on-read translation (spacepy docs: a 'raw' Var "does not
	perform this translation, on read or write" -- normally applied "to/from
	Unicode for CHAR variables on Py3k"), so this compares numeric byte
	codes directly instead of ever materializing a Python str per record --
	fully vectorized, no per-record object creation, where the string path
	(qf == 'GGG') requires pycdf to have already built n Python str objects
	before that comparison ever runs.

	Handles the two shapes that translation-free CDF_CHAR data could
	plausibly come back as (byte-string elements, or one int column per
	character -- exact shape isn't nailed down without spacepy available to
	test against directly): anything else raises ValueError so the caller's
	try/except falls back to the proven (slower) string-decoded comparison
	instead of silently computing a wrong mask."""
	arr = np.asarray(raw)
	if arr.ndim == 1 and arr.dtype.kind == 'S':
		return arr == b'GGG'
	if arr.ndim == 2 and arr.dtype.kind in 'iu':
		return (arr == 71).all(axis=1)   # 'G' == 71 (ASCII); one int column per char
	raise ValueError(f'unrecognized raw qf shape/dtype: {arr.shape} {arr.dtype}')


def _record_runs(time, good):
	"""[(start, end)] of each run of consecutive quality-good RECORDS (record index, not time gaps), as times:
	what _scm_good_segments and _edp_good_segments return, from a CDF or from the archive alike."""
	if not good.any():
		return []
	padded = np.concatenate(([False], good, [False]))
	diffs  = np.diff(padded.astype(np.int8))
	run_starts = np.flatnonzero(diffs == 1)
	run_ends   = np.flatnonzero(diffs == -1) - 1
	return [(time[s], time[e]) for s, e in zip(run_starts, run_ends)]


@lru_cache(maxsize=None)
def _scm_good_segments(cdf_path, probe):
	"""List of (start, end) Timestamp pairs, one per contiguous run of
	quality-good samples in an SCM/B CDF file -- the exact same qf == 'GGG'
	filter _cdf_to_df's SCM path applies (kept in lockstep on purpose: this
	function exists so the interval manifest reports precisely the spans
	that actually survive into the FFT computation, not the file's raw
	Epoch span). A leading/trailing convolution-edge artifact (the common
	case -- see _cdf_to_df's docstring) trims the reported span at that
	end; any other bad stretch inside a file (e.g. saturation/fill-value
	flags away from the boundary) splits it into multiple disjoint
	segments, which downstream (cdf_data_intervals + merge_bursts) become
	real gaps in the manifest instead of being silently spanned over.

	Empty list if the file has no good samples at all, or is empty --
	callers should treat that as "this file contributes nothing".

	Reads the FULL Epoch and quality-flag arrays (needed to find exact run
	boundaries, not just first/last) -- more than cdf_time_bounds' 2-sample
	read, but the quality-flag array is already full-length by necessity,
	so this adds one comparably-sized array read, not a new order of cost.

	@lru_cache'd on (cdf_path, probe): plot_intervals/collect_fft_data call
	this once per (interval, probe) grid cell with NO cache of their own
	(unlike prewarm's local segments_cache), so a file spanning several
	short intervals was being re-opened and fully re-scanned -- Epoch
	convert + qf read -- once per interval that touched it, which read as
	an infinite loop over the SCM files. Caching here
	makes every call site (batched or not) pay the cost once per file for
	the life of the process; a file that's rewritten on disk mid-run
	(corrupted-then-redownloaded under the same path) would need the
	process restarted to pick up the new content, which is already true of
	the pickle caches elsewhere in this module.
	"""
	a = _archived_row(cdf_path, 'B')
	if a is not None:
		return _record_runs(_archive_time_index(a.recorded), a.good) if a.recorded.size else []
	from spacepy import pycdf
	import time as _time_mod  # TEMP diagnostic (see the qf prints below) -- remove with the prints
	with pycdf.CDF(cdf_path) as cdf:
		n = len(cdf['Epoch'])
		if n == 0:
			return []
		time = _epoch_datetime64(cdf, 'Epoch')
		qf_var = f'mms{probe}_scm_qf_scm123_{_scm_mode(cdf_path)}_brst_l2'
		_t_qf0 = _time_mod.monotonic()  # TEMP diagnostic -- qf (CDF_CHAR/string) had NO timing before; suspected the real cost
		try:
			# raw_var: see _qf_good_mask -- skips pycdf's per-record CDF_CHAR
			# Unicode decode entirely, comparing byte codes instead of strings.
			print(f'[_scm_good_segments] {cdf_path} -- reading qf ({n} samples) via raw_var...', flush=True)  # TEMP diagnostic
			good = _qf_good_mask(cdf.raw_var(qf_var)[:])
			print(f'[_scm_good_segments] {cdf_path} qf read+compared (raw_var path) in {_time_mod.monotonic() - _t_qf0:.2f}s', flush=True)  # TEMP diagnostic
		except Exception as exc:
			# Any surprise (older spacepy without raw_var, an unexpected
			# shape/dtype _qf_good_mask didn't recognize, etc.) falls back
			# to the original, proven-correct string comparison -- never
			# silently computes a wrong mask just to save time.
			print(f'[_scm_good_segments] {cdf_path} raw_var qf path failed ({type(exc).__name__}: {exc}) -- falling back to string decode', flush=True)  # TEMP diagnostic
			_t_qf0 = _time_mod.monotonic()  # TEMP diagnostic
			qf = np.asarray(cdf[qf_var][:])
			good = (qf == 'GGG')
			print(f'[_scm_good_segments] {cdf_path} qf read+compared (string fallback) in {_time_mod.monotonic() - _t_qf0:.2f}s', flush=True)  # TEMP diagnostic
	return _record_runs(time, good)


def _scm_good_segments_job(job):
	cdf_path, probe = job
	try:
		return cdf_path, _scm_good_segments(cdf_path, probe)
	except Exception:
		return cdf_path, []


def _scm_good_segments_batch(jobs, max_workers=MAX_POOL_WORKERS):
	"""Batch, parallel version of _scm_good_segments -- runs every
	(cdf_path, probe) job across ONE shared ProcessPoolExecutor pool, same
	pattern (and now same MAX_POOL_WORKERS) as ensure_data_pickles/compute_ffts.
	Was dropped to 4 while the qf (quality-flag) read was suspected of
	disk-queue-contending, multi-minute-per-file marshalling overhead (see
	_scm_good_segments' git history / prior investigation); confirmed +
	fixed via CDF.raw_var (see _qf_good_mask -- ~30x faster, verified
	against real files across all 4 probes), so each job is cheap again and
	8 concurrent workers scaling with core count is back to being the right
	call, same as the other pools. Read-only (no Manager/lock needed, unlike
	ensure_data_pickles): each job just opens its own CDF file and returns
	that file's segments, independent of every other job -- no shared state
	to protect.

	jobs is deduplicated by (cdf_path, probe) so a path pulled in by two
	overlapping windows is only opened once. Returns {cdf_path: [(start,
	end), ...]}; a job whose CDF read raised is mapped to [] (same
	fail-open behavior _cached_segments' try/except gave each file before)."""
	jobs = list({job for job in jobs})
	if not jobs:
		return {}
	return dict(_pool_map(_scm_good_segments_job, jobs, max_workers,
	                      'SCM quality-segment scan'))


@lru_cache(maxsize=None)
def _edp_good_segments(cdf_path, probe):
	"""List of (start, end) Timestamp pairs, one per contiguous run of
	quality-good samples in an EDP/E CDF file -- the EDP analog of
	_scm_good_segments, characterized against real files the same way
	(see the mms{probe}_edp_quality_brst_l2 CATDESC, confirmed identical
	across all 4 probes / multiple dates via a real-file scratch
	investigation): "Quality indicator (3 good), (2 ok data, use with
	some caution), (1 bad data, use with caution), (0 Really bad data or
	no data at all)". 'Good' here means quality >= EDP_MIN_QUALITY (2),
	i.e. every tier that is a real measurement rather than a defect --
	see that constant for why tier 2 qualifies and tier 1 does not.

	This deliberately does NOT mirror SCM's qf == 'GGG' strictness. The
	two flags are not comparable: 'ZZZ' marks samples that are literally
	all-zero, so excluding them removes non-data, whereas EDP tier 2
	marks fully-valid samples carrying a documented caveat. Matching the
	strictness of the two flags -- the earlier rationale here -- matched
	their FORM while inverting their EFFECT, and cost 298 of 300 files on
	the dates actually being studied.

	The mms{probe}_edp_bitmask_brst_l2 variable is now the reason tier 2
	can be accepted with confidence: the EDP Data Products Guide v1.3
	Table 5 documents each bit and the tier it forces, and decoding the
	real files shows every tier-2 sample on these dates is bit 6 (ASPOC
	emitting current) alone. Per-bit meanings are absent from the CDF's
	own CATDESC ("Status bitmask"), which is why the earlier note here
	said they were undocumented -- they are documented, just not in-file.

	quality is a small numeric type (CDF_UINT1), not CDF_CHAR like SCM's
	qf -- never had the string-marshalling cost _qf_good_mask/raw_var
	exists for, so no analogous fast-path dance needed here.

	quality gates _cdf_to_df's actual EDP data too, using this same == 3
	test, so this function and the FFT input agree on what "good" means --
	exactly mirroring _scm_good_segments' relationship to the qf filter.
	(It did not always: the filter was originally SCM-only, because the
	investigation that produced it was chasing a raw all-zero-reading
	artifact that EDP doesn't exhibit -- a DIFFERENT question from this
	quality variable. Until that gap was closed, EDP FFTs were computed
	over quality-bad data the manifest simultaneously reported as unused.)

	Empty list if the file has no sample at or above EDP_MIN_QUALITY, or is empty.
	"""
	a = _archived_row(cdf_path, 'E')
	if a is not None:
		return _record_runs(_archive_time_index(a.recorded), a.good) if a.recorded.size else []
	from spacepy import pycdf
	epoch_var = f'mms{probe}_edp_epoch_brst_l2'
	with pycdf.CDF(cdf_path) as cdf:
		if len(cdf[epoch_var]) == 0:
			return []
		time = _epoch_datetime64(cdf, epoch_var)
		quality = np.asarray(cdf[f'mms{probe}_edp_quality_brst_l2'][:])
	good = quality >= EDP_MIN_QUALITY
	return _record_runs(time, good)


def _edp_good_segments_job(job):
	cdf_path, probe = job
	try:
		return cdf_path, _edp_good_segments(cdf_path, probe)
	except Exception:
		return cdf_path, []


def _edp_good_segments_batch(jobs, max_workers=MAX_POOL_WORKERS):
	"""Batch, parallel version of _edp_good_segments -- same pooling
	pattern as _scm_good_segments_batch. quality is already a cheap
	numeric read (see _edp_good_segments), so this never had SCM's
	disk/marshalling cost concern to begin with -- MAX_POOL_WORKERS from the
	start, no history of being throttled down."""
	jobs = list({job for job in jobs})
	if not jobs:
		return {}
	return dict(_pool_map(_edp_good_segments_job, jobs, max_workers,
	                      'EDP quality-segment scan'))


def _version_path(pickle_path):
	return pickle_path + '.ver'


def _read_version(pickle_path):
	"""Stored version for pickle_path, or None if missing/unreadable. Cheap --
	reads a few bytes from a sidecar file, never unpickles the actual data."""
	try:
		with open(_version_path(pickle_path)) as f:
			return int(f.read().strip())
	except (OSError, ValueError):
		return None


def _write_version(pickle_path, version):
	with open(_version_path(pickle_path), 'w') as f:
		f.write(str(version))


def _invalidate_if_stale(pickle_path, expected_version, source_paths=()):
	"""Delete pickle_path and its version sidecar if it is out of date against
	EITHER of the two independent staleness signals -- checked in this order:

	1. SOURCE MTIME (first, and the one that matters most). If any path in
	   `source_paths` is newer than the pickle, the pickle was derived from
	   content that has since changed on disk, so it is wrong no matter what
	   version wrote it. This is checked ahead of the version because it
	   answers a question the version cannot: the version says "which CODE
	   built this", mtime says "was the INPUT the same input". A pickle can be
	   at the current version and still be built from a CDF that has since
	   been re-downloaded -- which used to go completely undetected, because
	   existence + version were the only tests. Re-downloading a corrupt or
	   truncated file over its old path is exactly the case that produced it,
	   and nothing anywhere in the pipeline noticed.
	2. VERSION. Different generating logic (a new filter, a changed column, a
	   fixed unit bug) -- see DATA_PICKLE_VERSION / FFT_PICKLE_VERSION.

	The comparison is strict (`>`), so equal mtimes never invalidate: a pickle
	written in the same second as its source is not treated as stale, and the
	check cannot thrash on coarse filesystem timestamp granularity.

	This composes down the chain by construction. A CDF newer than its data
	pickle regenerates that pickle, which makes the data pickle newer than
	every FFT pickle derived from it, which regenerates those in turn -- so
	one changed CDF invalidates exactly the artifacts that actually depend on
	it, with no manual cache-hunting and no global version bump.

	Explicit deletion (not just letting the regenerate step overwrite it) so a
	stale file is never left readable under a mismatched version if
	regeneration is interrupted or fails partway (e.g. a corrupt CDF on the
	raw-pickle path).
	"""
	if not os.path.exists(pickle_path):
		return

	stale = False
	try:
		pkl_mtime = os.path.getmtime(pickle_path)
	except OSError:
		pkl_mtime = None
	if pkl_mtime is not None:
		for src in source_paths:
			if not src:
				continue
			try:
				if _source_mtime(src) > pkl_mtime:     # an archived CDF's mtime from its row
					stale = True
					break
			except OSError:
				# Source vanished -- not evidence the pickle is wrong, and
				# deleting a usable cache entry over a missing input would be
				# strictly worse than keeping it. Fall through to the version.
				continue

	if not stale and _read_version(pickle_path) != expected_version:
		stale = True

	if not stale:
		return

	try:
		os.remove(pickle_path)
	except OSError:
		pass
	try:
		os.remove(_version_path(pickle_path))
	except OSError:
		pass


def ensure_data_pickle(cdf_path, pickle_root, key, corrupted_log=None, lock=None, probe: int = 1):
	"""
	Ensure a per-file data pickle exists for cdf_path.
	If not, generate it from the CDF and save it.
	Pickle path: {pickle_root}/mms/{probe}/{inst}/{mode}/{14-digit-timestamp}.pkl
	Returns the pickle path.
	"""
	m = re.search(r'_(\d{4})(\d{2})(\d{2})(\d{6})_', cdf_path)
	if not m:
		raise ValueError(f'Cannot parse timestamp from CDF path: {cdf_path}')
	timestamp = m.group(4)

	f = key[0]
	if f == 'F':                       # FGM background field (for f_ce)
		inst, mode = 'fgm', 'brst'
	elif f == 'B':
		inst, mode = 'scm', _scm_mode(cdf_path)
	else:
		inst, mode = 'edp', 'dce'

	pickle_path = os.path.join(pickle_root,
	                           f'mms/{probe}/{inst}/{mode}/{m.group(1)}/{m.group(2)}/{m.group(3)}/{timestamp}.pkl')

	# Source is the CDF this pickle was converted from: re-download it (a
	# repaired or replaced file under the same name) and this regenerates.
	_invalidate_if_stale(pickle_path, DATA_PICKLE_VERSION[inst], source_paths=(cdf_path,))

	if not os.path.exists(pickle_path):
		print(f'Generating data pickle from {os.path.basename(cdf_path)}...')
		try:
			df = _cdf_to_df(cdf_path, key, probe=probe)
		except Exception as e:
			size = os.path.getsize(cdf_path)
			print(f'Bad CDF, deleting: {os.path.basename(cdf_path)} — {size} bytes, {type(e).__name__}: {e}')
			try:
				os.remove(cdf_path)
			except PermissionError:
				print(f'Cannot delete locked file: {os.path.basename(cdf_path)}')
			if corrupted_log:
				with (lock or _corrupted_log_lock):
					with open(corrupted_log, 'a') as f:
						f.write(cdf_path + '\n')
			return None
		os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
		df.to_pickle(pickle_path)
		_write_version(pickle_path, DATA_PICKLE_VERSION[inst])
		print(f'Saved: {pickle_path}')

	return pickle_path


# ---------------------------------------------------------------------------
# Reading burst CDFs directly
#
# The pipeline used to convert every CDF to a data pickle before touching it
# (ensure_data_pickle, above). On a real day -- MMS1 B+E, 2021-02-23, 62 CDFs, 4
# workers -- that conversion was 19.1 s of a 34.4 s first run and wrote 1.73 GB,
# while FFTs computed straight from the CDFs matched row for row and to the
# checksum (_wsl_scratch/direct_cdf_bench.py). So every reader below takes the
# CDF itself. The one thing the pickles still gave a repeat run, good-run
# boundaries without rereading a file, is kept as a sidecar of boundaries only
# (_segments_via_sidecar). The pickle helpers stay defined; the pipeline no
# longer calls them.
# ---------------------------------------------------------------------------

# mms{probe}_{inst}_brst_l2_[{mode}_]{YYYYMMDDHHMMSS}_v... -- FGM names carry no mode.
_BURST_CDF_NAME = re.compile(r'mms(\d)_([a-z]+)_brst_l2_(?:([a-z]+)_)?(\d{14})_')

# Version of the good-run sidecars (_segments_via_sidecar). Bump it when the
# quality test or the run rule changes. A changed CDF is caught by its mtime, and
# a different rate or gap factor by the sidecar's directory (_segments_sidecar_path).
# ONE gap threshold for deciding where data stops being continuous, applied
# identically INSIDE a file and ACROSS a file boundary (Graham, 2026-09-25: "the
# segment threshold should absolutely apply over file boundaries").
#
# It was two different numbers before, and neither was chosen: _good_run_bounds
# split inside a file at 1.5 sample periods while merge_bursts joined across a
# boundary at 2.0, so the same hole was a break in one place and not in the other
# depending only on whether a file happened to end in the middle of it. At 1.5
# periods (183 us) ONE missing sample already started a new segment.
#
# Measured on EVERY burst CDF on E: (2026-09-25), which is what settles it:
#
#   SCM  11,449 files, 12,819,497,216 good samples
#        gaps inside a file : ZERO. Not one, anywhere.
#        gaps between files : 5,370, narrowest 4.000 s exactly
#   EDP   1,013 files,  1,155,441,687 good samples
#        gaps inside a file : 99,586, widest 647.8 ms
#        gaps between files : 282 -- 15 of them at 103 ms or less, then NOTHING
#                             until 10.936 s (a 106x jump), then 267 at 4 s+
#
# Two populations, and they do not overlap:
#
#   <= 647.8 ms   EDP quality dropouts. The instrument chattering inside one
#                 continuous burst. They land inside a file AND straddle file
#                 boundaries (hence the 1-103 ms boundary gaps: an EDP file's
#                 leading and trailing samples are often quality-trimmed), which
#                 is exactly why one threshold has to cover both cases.
#   >= 4.000 s    genuine loss -- a burst segment that was never recorded.
#
# So the empty band runs 647.8 ms to 4.000 s, a factor of 6.2, and its geometric
# centre is 1.61 s. 1.5 s is the round number closest to that: 2.3x above the
# widest dropout, 2.7x below the narrowest real gap. Every value in the band
# gives identical results on everything on E:, so the choice is robust within
# those bounds -- but the bounds are 6x, not the 32,000x an SCM-only look
# suggested, so it is not arbitrary either.
#
# In SECONDS, not sample periods, because the decision is physical ("is this the
# same burst?") and must not change meaning when the rate does. scb/dce at 8192
# Hz and schb at 16384 Hz get the same 1.5 s.
SEGMENT_GAP_SECONDS = 1.5


def segment_gap_factor(fs):
	"""SEGMENT_GAP_SECONDS expressed in sample periods, the unit _good_run_bounds
	and the sidecar path speak -- for a caller that wants runs cut at the event
	threshold itself rather than at GOOD_RUN_GAP_FACTOR."""
	return SEGMENT_GAP_SECONDS * fs


def segment_gap_tol(fs):
	"""The event threshold as a timedelta: merge_bursts' tol when a manifest groups
	good runs into physical events. The same rule judges a hole inside a file and a
	seam between two files, since both are just a gap between two good runs."""
	return timedelta(seconds=SEGMENT_GAP_SECONDS)


# The sidecar covers ALL good data (Graham, 2026-09-29: "The manifest is supposed to
# measure continuous events, the sidecar is supposed to cover all good data"). So a
# good run ends wherever a good sample is missing: the next one sits more than 1.5
# sample periods on, which timestamp jitter never reaches (neighbours are 122000 or
# 123000 ns apart) and a single dropped sample (2 periods) always does. Grouping the
# runs into physical events is the manifest's job, at SEGMENT_GAP_SECONDS
# (segment_gap_tol). Same number as CADENCE_JITTER_FACTOR below, but a different
# question -- see there -- so each keeps its own name.
GOOD_RUN_GAP_FACTOR = 1.5


# Graham, 2026-10-09: "A lone good sample should never count since we're doing math
# that relies on continuous data. Single digit numbers of samples do not count as
# events." A good run of fewer samples than this is no part of any event: find_events,
# every manifest's used table and the plot's chunks drop it before grouping runs at
# SEGMENT_GAP_SECONDS. The sidecars still hold it, since they cover all good data, and
# no FFT window ever holds one: a window is filled by ~4095 good samples in a row.
MIN_GOOD_RUN_SAMPLES = 10


def good_run_rule(fs):
	"""MIN_GOOD_RUN_SAMPLES as data_intervals.drop_short_runs takes it at rate fs:
	{'min_run', 'run_gap'}. A run of n samples spans n - 1 periods, so the cut sits
	half a period below MIN_GOOD_RUN_SAMPLES - 1 (8.5 periods: 9 samples span 8, 10
	span 9, whatever the microsecond rounding of their times). Rows chain into one
	run across up to GOOD_RUN_GAP_FACTOR periods -- the sidecar's own split -- so a
	run a file boundary cuts in two counts whole."""
	return {'min_run': timedelta(seconds=(MIN_GOOD_RUN_SAMPLES - 1.5) / fs),
	        'run_gap': timedelta(seconds=GOOD_RUN_GAP_FACTOR / fs)}


# NOT the segment threshold, and it must never be wired to it. This one answers a
# completely different question in _compute_file_ffts: "is this gap between two
# consecutive timestamps real, or is it the microsecond quantisation of a 122072
# ns period?" It exists only to keep a hole out of the MEAN sample period that
# sizes a full window (see n_full).
#
# Sharing SEGMENT_GAP_SECONDS here would be a silent regression: at 1 s, every
# EDP dropout up to 125 ms would count as normal spacing, dragging the measured
# period from 122072 ns to roughly 127600 and putting n_full near 3919 instead of
# 4095 -- which would let windows missing ~177 real samples pass the fill test.
# A jitter margin is 1.5 periods because two adjacent samples are 122000 or
# 123000 ns apart and nothing legitimate is wider.
CADENCE_JITTER_FACTOR = 1.5

SEGMENTS_VERSION = 1

# Version of the recorded-run sidecars (_segments_via_sidecar(..., recorded=True)):
# every run of samples a CDF holds, whatever their quality, for the manifest's
# recorded table (Graham, 2026-09-30). Bump it when the run rule changes; the
# quality test does not touch these.
RECORDED_VERSION = 1


def _burst_cdf_name(cdf_path):
	"""(probe, inst, mode, file_time) from a burst CDF's filename; mode is ''
	for FGM, whose files carry none."""
	m = _BURST_CDF_NAME.search(os.path.basename(cdf_path))
	if not m:
		raise ValueError(f'Not a burst CDF filename: {cdf_path}')
	return int(m.group(1)), m.group(2), m.group(3) or '', datetime.strptime(m.group(4), '%Y%m%d%H%M%S')


def _scm_mode(cdf_path):
	"""'scb' or 'schb' for an SCM burst CDF, from its own name: a file is read with
	the variable names of the mode it is, whichever mode B_BURST selects. B_BURST's
	own mode when the name does not carry one (a scratch copy, say)."""
	m = _BURST_CDF_NAME.search(os.path.basename(cdf_path))
	return m.group(3) if m and m.group(3) else B_BURST.mode


class _CdfRecords:
	"""A pycdf CDF seen through records [start, stop) of every variable.

	Covers exactly what _cdf_to_df and _epoch_datetime64 do with a CDF --
	cdf[name] and cdf.raw_var(name), then len() and slices of those -- so both
	run unchanged on part of a file."""

	def __init__(self, cdf, start, stop):
		self._cdf, self._start, self._stop = cdf, start, stop

	def raw_var(self, name):
		return _VarRecords(self._cdf.raw_var(name), self._start, self._stop)

	def __getitem__(self, name):
		return _VarRecords(self._cdf[name], self._start, self._stop)


class _VarRecords:
	"""One pycdf variable seen through records [start, stop). Slices only."""

	def __init__(self, var, start, stop):
		self._var, self._start, self._stop = var, start, stop

	def __len__(self):
		return self._stop - self._start

	def __getitem__(self, key):
		lo, hi, step = key.indices(len(self))
		if step != 1:
			raise ValueError('only unit-step slices are supported')
		return self._var[self._start + lo:self._start + max(lo, hi)]


def _records_before(cdf_path, key, probe, until_ns, fs):
	"""How many leading records of a CDF certainly hold every sample earlier
	than until_ns (int ns), or None when the answer is the whole file.

	Sized from the file's own first sample, not its filename stamp: 25 of 60 real
	files begin more than 1.25 s before their stamp. At the nominal rate fs that
	many records reach until_ns, a gap only stretches them further in time, and a
	quarter second more is margin. The last record's time is checked all the same;
	a head that still stops short means the whole file."""
	from spacepy import pycdf
	epoch_var = 'Epoch' if key[0] in ('B', 'F') else f'mms{probe}_edp_epoch_brst_l2'
	with pycdf.CDF(cdf_path) as cdf:
		n = len(cdf[epoch_var])
		if n == 0:
			return None
		first_ns = int(_epoch_datetime64(_CdfRecords(cdf, 0, 1), epoch_var).as_unit('ns').asi8[0])
		k = int(np.ceil(max(until_ns - first_ns, 0) / 1e9 * fs)) + int(0.25 * fs)
		if k >= n:
			return None
		last_ns = int(_epoch_datetime64(_CdfRecords(cdf, k - 1, k), epoch_var).as_unit('ns').asi8[0])
	return k if last_ns >= until_ns else None


def read_burst_data(path, key, probe=None, until_ns=None, fs=None):
	"""One burst file's quality-good samples as the FFT code reads them: time
	(datetime64[ns]) plus the field's columns -- _cdf_to_df's frame.

	path      a burst CDF. A .pkl is still read, through read_data_pickle, so
	          old data pickles -- and tests or scratch code built on them -- work.
	probe     taken from the CDF's filename when not given.
	until_ns  (int ns) the caller needs only samples earlier than this. The FFT
	          worker reads the NEXT file for exactly that: a window borrows from
	          it only once it reaches past the current file, and every borrowed
	          sample is earlier than next_file_time + pad (see
	          _compute_file_ffts). Then only the leading records are read
	          (_records_before). They hold every sample before until_ns, in the
	          same rows a whole read gives, so no FFT can tell the difference:
	          60 of 60 real seam jobs were identical and next-file read CPU fell
	          24.6 -> 17.3 s (_wsl_scratch/next_head_check.py). Ignored for a pickle.
	fs        the nominal rate that sizes that head.

	'time' is normalised to datetime64[ns] for the reason read_data_pickle
	gives: _compute_file_ffts casts it straight to int64 nanoseconds, and pandas 3
	builds this column in microseconds.
	"""
	if path.endswith('.pkl'):
		return read_data_pickle(path)
	if probe is None:
		probe = _burst_cdf_name(path)[0]
	if fs is None:
		# The field's own rate (E never takes B's selection). FGM keeps the one it
		# always had here; it only sizes a head read, where more records is the safe side.
		fs = field_burst(key)[2] if key[0] in ('B', 'E') else SAMPLE_RATE
	n_records = None if until_ns is None else _records_before(path, key, probe, until_ns, fs)
	df = _cdf_to_df(path, key, probe=probe, n_records=n_records)
	if df['time'].dtype != 'datetime64[ns]':
		df['time'] = df['time'].astype('datetime64[ns]')
	return df


def read_burst_signal(path, key, probe=None, until_ns=None, fs=None):
	"""(t, s) of one B or E burst file's quality-good samples: t int64 ns, s the FFT input |field| at each,
	sqrt(z**2 + y**2 + x**2) -- the expression _compute_file_ffts applied window by window, applied once to
	the whole file. Every operation in it is elementwise and correctly rounded, so a window's slice of s is
	bit for bit what the window computed from its own x, y and z (proven on real jobs, 2026-11-03).

	From the archive when it holds the file (preproc_archive stores exactly this s, float32, and t), else
	from read_burst_data, whose arguments these are; until_ns only shortens a CDF read."""
	a = _archived_row(path, key, signal=True)
	if a is not None:
		return a.times, a.signal
	df = read_burst_data(path, key, probe=probe, until_ns=until_ns, fs=fs)
	f = key[0]
	x, y, z = df[f + 'x'].values, df[f + 'y'].values, df[f + 'z'].values
	return df['time'].values.astype(np.int64), np.sqrt(z**2 + y**2 + x**2)


def _good_sample_times(cdf_path, key, probe):
	"""int64 ns times of a B or E CDF's quality-good samples -- the time column
	_cdf_to_df keeps -- read from the time and quality variables only. The data
	variable, most of the file, is never touched. Raises when unreadable. The
	archive's row, when it holds this file."""
	a = _archived_row(cdf_path, key)
	if a is not None:
		return a.times
	from spacepy import pycdf
	with pycdf.CDF(cdf_path) as cdf:
		if key[0] == 'B':
			time = _epoch_datetime64(cdf, 'Epoch')
			qf_var = f'mms{probe}_scm_qf_scm123_{_scm_mode(cdf_path)}_brst_l2'
			try:
				good = _qf_good_mask(cdf.raw_var(qf_var)[:])      # same fast path and fallback as _cdf_to_df
			except Exception:
				good = np.asarray(cdf[qf_var][:]) == 'GGG'
		else:
			time = _epoch_datetime64(cdf, f'mms{probe}_edp_epoch_brst_l2')
			good = np.asarray(cdf[f'mms{probe}_edp_quality_brst_l2'][:]) >= EDP_MIN_QUALITY
	return time.as_unit('ns').asi8[good]


def _recorded_sample_times(cdf_path, key, probe):
	"""int64 ns times of EVERY sample a B or E CDF holds, whatever its quality --
	where the instrument recorded, which the manifest's recorded table reports
	beside the good data (Graham, 2026-09-30). The time variable alone is read.
	Raises when unreadable. The archive's row, when it holds this file."""
	a = _archived_row(cdf_path, key)
	if a is not None:
		return a.recorded
	from spacepy import pycdf
	epoch_var = 'Epoch' if key[0] == 'B' else f'mms{probe}_edp_epoch_brst_l2'
	with pycdf.CDF(cdf_path) as cdf:
		return _epoch_datetime64(cdf, epoch_var).as_unit('ns').asi8


def _good_run_bounds(t, fs, gap_factor=None):
	"""(n, 2) int64 ns [start, end] of each contiguous run of sorted sample
	times t, split wherever neighbours sit more than gap_factor sample periods
	apart -- _segments_from_pickle's rule. A lone sample is a run of its own."""
	if len(t) == 0:
		return np.empty((0, 2), dtype=np.int64)
	gap_factor = GOOD_RUN_GAP_FACTOR if gap_factor is None else gap_factor
	breaks = np.flatnonzero(np.diff(t) > gap_factor * 1e9 / fs)
	starts = np.concatenate(([0], breaks + 1))
	ends   = np.concatenate((breaks, [len(t) - 1]))
	return np.stack([t[starts], t[ends]], axis=1).astype(np.int64)


def _as_segments(runs):
	return [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in runs]


def _segments_from_cdf(cdf_path, key, fs, probe, gap_factor=None):
	"""_segments_from_pickle's answer for the data pickle this CDF would have
	become -- its quality-good runs [(start, end), ...] -- read from the CDF's
	time and quality variables alone. [] when the file has no good sample or
	cannot be read, as _segments_from_pickle answers for an unreadable pickle.

	Identical to the pickle-derived runs on 62 real B and E files: 5,371
	segments, one digest (_wsl_scratch/direct_cdf_bench.py)."""
	if key[0] not in ('B', 'E'):
		raise ValueError(f'quality-good runs exist for B and E, not {key!r}')
	try:
		t = _good_sample_times(cdf_path, key, probe)
	except Exception:
		return []
	return _as_segments(_good_run_bounds(t, fs, gap_factor))


def _segments_sidecar_path(pickle_root, cdf_path, fs, gap_factor=None, kind='segments'):
	"""{pickle_root}/mms/{probe}/{inst}/{mode}/{kind}/gap{N}ns/{YYYY}/{MM}/{DD}/{HHMMSS}.npy

	kind is 'segments' for the good runs and 'recorded' for every recorded run
	(see _segments_via_sidecar); the two never share a file.

	N is the split threshold in whole nanoseconds, floor(gap_factor * 1e9 / fs):
	183105 at 8192 Hz for the default GOOD_RUN_GAP_FACTOR, one missing sample.
	Sample times are whole nanoseconds, so N decides every split, and a changed
	threshold lands in its own directory instead of reusing runs cut at the old
	one -- a parameter change neither staleness signal can see (the trap
	_fft_pickle_path documents). The 1.5 s runs written before 2026-09-29 sit in
	gap1500000000ns and are no longer read. Probe, instrument and mode come from
	the CDF's own name, so an scb file and an schb file never share an entry."""
	probe, inst, mode, file_time = _burst_cdf_name(cdf_path)
	gap_factor = GOOD_RUN_GAP_FACTOR if gap_factor is None else gap_factor
	gap_ns = int(np.floor(gap_factor * 1e9 / fs))
	return os.path.join(pickle_root, f'mms/{probe}/{inst}/{mode}/{kind}/gap{gap_ns}ns',
	                    file_time.strftime('%Y/%m/%d/%H%M%S.npy'))


def _segments_via_sidecar(cdf_path, key, fs, probe, pickle_root, gap_factor=None, recorded=False):
	"""_segments_from_cdf, remembered per CDF in a tiny .npy under pickle_root:
	each run's start and end as int64 ns, shape (n, 2), and nothing else.

	It covers ALL of the file's good data: one run per stretch of good samples,
	split wherever one is missing (GOOD_RUN_GAP_FACTOR). It does not decide what
	a physical event is -- a manifest groups these runs, clipped to its request,
	at the event threshold (segment_gap_tol), so `recorded` counts good data
	only and `event` spans the whole event.

	This is what the data pickles still bought after a first run. Event discovery
	read good-run boundaries off them at ~1.3 s per day of data, where rereading
	every CDF's time and quality variables takes ~8 s; the sidecar keeps that and
	drops the data. It goes stale the way every cache here does
	(_invalidate_if_stale): a CDF newer than it, or a SEGMENTS_VERSION it was not
	written under.

	Nothing is remembered for a file that cannot be read right now -- a CDF held
	open by another process must not be recorded as holding no good data. A failed
	write costs only the next call's time, so it is not raised.

	recorded=True answers the manifest's other question (Graham, 2026-09-30: "the
	recorded should show where it was recorded and not necessarily when good quality
	data was recorded"): every run of samples the file holds, whatever their quality,
	from its time variable alone (_recorded_sample_times), split by the same rule.
	Those runs have their own sidecar (kind 'recorded') and RECORDED_VERSION."""
	if key[0] not in ('B', 'E'):
		raise ValueError(f'quality-good runs exist for B and E, not {key!r}')
	version = RECORDED_VERSION if recorded else SEGMENTS_VERSION
	path = _segments_sidecar_path(pickle_root, cdf_path, fs, gap_factor,
	                              kind='recorded' if recorded else 'segments')
	_invalidate_if_stale(path, version, source_paths=(cdf_path,))
	if os.path.exists(path):
		try:
			return _as_segments(np.load(path))
		except (OSError, ValueError):
			pass                                    # unreadable sidecar: rebuilt below
	try:
		t = (_recorded_sample_times if recorded else _good_sample_times)(cdf_path, key, probe)
	except Exception:
		return []
	runs = _good_run_bounds(t, fs, gap_factor)
	tmp = f'{path}.{os.getpid()}.tmp'
	try:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(tmp, 'wb') as fh:
			np.save(fh, runs)
		os.replace(tmp, path)
		_write_version(path, version)
	except OSError:
		try:
			os.remove(tmp)            # never read, but not left behind either
		except OSError:
			pass
	return _as_segments(runs)


def _good_time_bounds(cdf_path, key, probe, pickle_root=None, fs=None):
	"""(first, last) quality-good sample time of a CDF as datetimes, or
	(None, None) when it has none or cannot be read -- what _pickle_time_bounds
	gave for the data pickle made from it, which held only good samples.

	_resolve_anchor needs these rather than cdf_time_bounds' raw ones: a file whose
	raw data reaches `start` while its good data stops short contributes nothing
	there, so it must not anchor the request. With raw bounds it would, and gain an
	FFT job the pickle pipeline never ran.

	B and E: the first run's start and the last run's end, from the sidecar when
	pickle_root is given. F (FGM) has no quality filter, so its good bounds are
	its raw first and last records. fs defaults to the field's own rate: E's
	sidecar never takes B's selection (find_local_cdfs calls this for both)."""
	try:
		if key[0] == 'F':
			first, last = cdf_time_bounds(cdf_path, key='F', probe=probe)
			return (None, None) if first is None else (first.to_pydatetime(), last.to_pydatetime())
		if fs is None:
			fs = field_burst(key)[2]
		segs = (_segments_via_sidecar(cdf_path, key, fs, probe, pickle_root) if pickle_root is not None
		        else _segments_from_cdf(cdf_path, key, fs, probe))
	except Exception:
		return None, None
	if not segs:
		return None, None
	return segs[0][0].to_pydatetime(), segs[-1][1].to_pydatetime()


def find_local_cdfs(data_dir, inst, mode, start, end, probe: int = 1, verify: bool = True, pickle_root=None,
                    check=None):
	"""find_data_pickles over the CDFs themselves: sorted [(file_time, cdf_path)]
	for the local burst CDFs covering [start, end], picked by the same shared
	_resolve_anchor rule with the same bounds -- the naive anchor's GOOD sample
	bounds (_good_time_bounds), which are exactly the bounds its data pickle had.
	Local only: nothing is downloaded, and a confirmed gap falls forward to the
	next real file.

	One file per filename stamp. A data pickle was keyed by its stamp, so two
	versions of one CDF on disk (v2.2.0 and v2.2.1) were one entry; here the later
	one in sort order stands for it.

	pickle_root: where the good-run sidecars live (see _segments_via_sidecar);
	None reads the anchor's time and quality variables instead.

	check: see find_cdf_files. The anchor is probed only once it has validated --
	when the naive anchor was rejected upstream, the next one back is a file
	nothing has checked yet -- and the files returned are validated before the
	one-per-stamp pick, so a rejected v2.2.1 leaves its good v2.2.0 standing."""
	key = _bounds_key(inst)
	scan = lambda: _scan_cdf_files(data_dir, inst, mode, probe)
	if verify:
		anchor, all_files = _resolve_anchor(
			scan, start, lambda p: (_good_time_bounds(p, key, probe, pickle_root=pickle_root)
			                        if check is None or check([p]) else (None, None)))
	else:
		all_files = scan()
		anchor = _anchor_index(all_files, start)
	if anchor is None:
		return []
	candidates = [(ftime, path) for ftime, path in all_files[anchor:] if ftime <= end]
	if check is not None:
		ok = set(check([path for _ftime, path in candidates]))
		candidates = [(ftime, path) for ftime, path in candidates if path in ok]
	by_time = {}
	for ftime, path in candidates:
		by_time[ftime] = path
	return sorted(by_time.items())


def pair_with_next_cdf(data_files, data_dir, inst, mode, probe: int = 1, check=None):
	"""Pair each discovered file with its successor, taking the LAST file's
	successor from the CDFs ON DISK rather than from the request.

	A file's FFT pickle must not depend on which request computed it. Taking
	every next file from the request meant the last file of a narrow request
	got next_path=None and so lost the seam samples a wider request would have
	borrowed -- and because a CDF's mtime never moves when a later file first
	joins a run, the stale pickle was then reused forever (Graham, 2026-09-15:
	"If the file ends during the FFT, grab the required data from the next
	[file]"). Looking the successor up on disk makes the pickle a function of
	the CDFs alone.

	Cheap: filename stamps only, no CDF opens. The successor is picked by the
	existing _fall_forward rule (leftmost file whose stamp is > this one), so
	"the file after X" is defined in exactly one place. One file per stamp,
	later in sort order winning, exactly as find_local_cdfs does. A successor
	that is nowhere near the seam is harmless -- _compute_file_ffts only
	borrows once next_file_time falls inside the window.

	The FIRST file's PREDECESSOR comes off disk for exactly the same reason, now
	that a window reaches backwards as well as forwards (see _compute_file_ffts).
	Without it the leading windows of a request's first file would be dropped
	while the identical file mid-request kept them, and the pickle would again be
	a function of the request rather than of the CDFs.

	check: see find_cdf_files. These two neighbours come off disk, not out of the
	request, so nothing upstream has validated them -- and the FFT worker reads
	them. Each is validated before it is handed on; a rejected one is deleted and
	the next file out takes its place. That file is not adjacent, so it lends no
	samples and the windows it would have filled are dropped, as at any gap. The
	pickle stays a function of the CDFs: when the neighbour is refetched, its new
	mtime marks the pickle stale (get_or_compute_fft's source_paths).

	Returns [(file_time, path, prev_file_time, prev_path, next_file_time, next_path)]."""
	if not data_files:
		return []

	by_time = {}
	for ftime, path in _scan_cdf_files(data_dir, inst, mode, probe):
		by_time[ftime] = path
	on_disk = sorted(by_time.items())

	def usable(k):
		return check is None or bool(check([on_disk[k][1]]))

	first_time = data_files[0][0]
	# Rightmost file on disk strictly before this one -- _anchor_index' rule,
	# nudged off the boundary because it is inclusive. Stamps are whole seconds,
	# so a microsecond cannot skip a real neighbour.
	j = _anchor_index(on_disk, first_time - timedelta(microseconds=1))
	while j is not None and not usable(j):
		j = j - 1 if j > 0 else None
	prev_first = on_disk[j] if j is not None else (None, None)

	last_time, last_path = data_files[-1]
	i = _fall_forward(on_disk, last_time)
	while i is not None and not usable(i):
		i = i + 1 if i + 1 < len(on_disk) else None
	next_last = on_disk[i] if i is not None else (None, None)

	paired = []
	for k, (ft, p) in enumerate(data_files):
		prev = prev_first if k == 0 else data_files[k - 1]
		nxt  = next_last  if k == len(data_files) - 1 else data_files[k + 1]
		paired.append((ft, p) + tuple(prev) + tuple(nxt))
	return paired


def _check_cdf(cdf_path, key, probe):
	"""Raise unless the first and last record of every variable _cdf_to_df
	reads from this CDF can be read."""
	from spacepy import pycdf
	f = key[0]
	if f == 'F':
		names = ('Epoch', f'mms{probe}_fgm_b_gse_brst_l2')
	elif f == 'B':
		mode = _scm_mode(cdf_path)
		names = ('Epoch', f'mms{probe}_scm_acb_gse_{mode}_brst_l2', f'mms{probe}_scm_qf_scm123_{mode}_brst_l2')
	else:
		names = (f'mms{probe}_edp_epoch_brst_l2', f'mms{probe}_edp_dce_gse_brst_l2',
		         f'mms{probe}_edp_quality_brst_l2')
	with pycdf.CDF(cdf_path) as cdf:
		n = len(cdf[names[0]])
		for name in names:
			var = cdf.raw_var(name)
			if n:
				var[0], var[n - 1]


# Every CDF version check_cdfs has passed in this process: (path, field letter,
# mtime_ns, size). Converting a file to a pickle happened once per file version, and
# so did the check that came with it. Checking on every call instead costs one CDF
# open per file per call -- ~72 ms, nearly all of it the CDF library validating the
# file -- which made a repeat Find events 3-5 s slower than the pickles were. A
# re-downloaded file has a new mtime and size, so it is checked again.
_CHECKED_CDFS = set()


def check_cdfs(jobs, corrupted_log=None, pickle_root=None):
	"""Check every (cdf_path, key, probe) job's CDF can be read, deleting and
	logging any that cannot exactly as ensure_data_pickle did when a conversion
	failed -- same messages, same corrupted_log line -- so download.py's
	re-download loop still finds them. Returns the paths in job order, None
	where a file was bad.

	The pipeline used to get this check as a side effect of converting every file
	to a pickle. Reading the first and last record of each variable _cdf_to_df
	needs catches the bad files downloads actually leave -- an HTML error page
	saved as .cdf, or a truncated file -- at a few records' cost instead of a whole
	read. Runs in this process: a check is cheaper than starting a worker pool.
	Each file version is checked once per process (_CHECKED_CDFS) and once ever
	across runs (cdf_validation). The database is consulted BEFORE the file is
	opened, which is the entire point: the ~60 ms is the library validating the
	file at open time, so asking afterwards would already have paid it. A version
	that passed in any earlier run costs one indexed lookup here instead.

	A file that fails is deleted and logged exactly as before -- that is how the
	redownload loop refetches it -- but the failure is COUNTED in the database
	first, keyed on the name so the count survives the deletion. Once a name has
	failed GIVE_UP_AFTER times, download.py stops refetching it and the name is
	written into the interval manifest's corrupted section instead (Graham,
	2026-09-25).

	No corrupt file outlives its detection (Graham, 2026-09-27: "the file should
	only be deleted and notated in a manifest that it's been deleted"). So bytes
	already recorded bad -- still here only because the delete failed last time,
	a locked file -- are deleted now without being reopened, and logged again so
	the redownload loop retries them, but not counted again: a failed delete is
	not a failed download."""
	checked = []
	newly_passed = []
	for cdf_path, key, probe in jobs:
		if _archived_only(cdf_path, key):
			# The archive holds it and the CDF is gone: verified when archived (and per request), nothing to open.
			checked.append(cdf_path)
			continue
		try:
			st = os.stat(cdf_path)
			version = (cdf_path, key[0], st.st_mtime_ns, st.st_size)
		except OSError:
			version = None
		if version in _CHECKED_CDFS:
			checked.append(cdf_path)
			continue
		# Before the open, not after it.
		if cdf_validation.passed(cdf_path, key, pickle_root=pickle_root):
			if version is not None:
				_CHECKED_CDFS.add(version)
			checked.append(cdf_path)
			continue
		if cdf_validation.failed(cdf_path, key, pickle_root=pickle_root):
			# These exact bytes are already known bad; re-opening them would only
			# buy the same answer at 60 ms.
			_delete_bad_cdf(cdf_path, 'already recorded bad', corrupted_log)
			checked.append(None)
			continue
		try:
			_check_cdf(cdf_path, key, probe)
		except Exception as e:
			import acquisition       # here, not at the top: acquisition imports this module
			if acquisition._is_rate_limit_page(acquisition._head(cdf_path)):
				# Not a download that came down wrong: the SDC refused (429) and pyspedas
				# saved its reply under the CDF's name. Deleted and logged for the
				# redownload loop like any bad file, but not counted -- acquisition
				# treats a 429 page the same way (Graham, 2026-10-01).
				_delete_bad_cdf(cdf_path, 'an SDC rate-limit (429) page, not counted', corrupted_log)
				checked.append(None)
				continue
			n = cdf_validation.record_failure(cdf_path, key, pickle_root=pickle_root)
			attempt = f'attempt {n}/{cdf_validation.GIVE_UP_AFTER}' if n else 'attempt uncounted'
			_delete_bad_cdf(cdf_path, f'{attempt}, {type(e).__name__}: {e}', corrupted_log)
			checked.append(None)
			continue
		if version is not None:
			_CHECKED_CDFS.add(version)
		newly_passed.append((cdf_path, key))
		checked.append(cdf_path)
	# One transaction for the batch rather than one fsync per file.
	if newly_passed:
		cdf_validation.record_passes(newly_passed, pickle_root=pickle_root)
	return checked


def _delete_bad_cdf(cdf_path, why, corrupted_log=None):
	"""Delete a CDF check_cdfs has rejected and, given a log, hand it to the
	redownload loop -- the messages and log line ensure_data_pickle used."""
	size = os.path.getsize(cdf_path) if os.path.exists(cdf_path) else 0
	print(f'Bad CDF, deleting: {os.path.basename(cdf_path)} — {size} bytes, {why}')
	try:
		os.remove(cdf_path)
	except PermissionError:
		print(f'Cannot delete locked file: {os.path.basename(cdf_path)}')
	except FileNotFoundError:
		pass
	if corrupted_log:
		with _corrupted_log_lock:
			with open(corrupted_log, 'a') as f:
				f.write(cdf_path + '\n')


def cdf_check(key, probe, pickle_root=None, corrupted_log=None):
	"""check_cdfs bound to one field and probe: check(paths) -> the paths that
	validated, in order, every rejected one deleted, logged and counted.

	Discovery takes this as `check` (find_cdf_files, find_local_cdfs,
	pair_with_next_cdf) and calls it before it opens a file and on everything it
	hands back, so nothing downstream of discovery ever meets a file validation
	has not passed (Graham, 2026-09-27: "Anything after that initial validation
	should just access it and never have access to corrupt files in the first
	place"). A file already passed in this process costs a stat and a set lookup."""
	def check(paths):
		# A path no longer on disk was rejected (and counted) already, or has
		# gone; either way there is nothing to validate, and passing it to
		# check_cdfs would count it again. download.py's retry rounds call
		# check_cdfs directly, where a refetch that never arrived SHOULD count.
		# An archived CDF that is gone from disk is still there to read (preproc_archive).
		paths = [p for p in paths if os.path.exists(p) or _archived_only(p, key)]
		if not paths:
			return []
		ok = check_cdfs([(p, key, probe) for p in paths], corrupted_log=corrupted_log,
		                pickle_root=pickle_root)
		return [p for p in ok if p is not None]
	return check


# ---------------------------------------------------------------- FFT scratch: no FFT cache across requests
#
# Graham, 2026-11-19: "Toss the cache honestly. Producing FFTs is the quickest processing step that actually
# constitutes as 'processing', and can be done on the fly per request." FFT pickles used to persist under
# <pickle_root>/mms/<p>/<inst>/<mode>/fft/ from one request to the next, and grew to 2.4-3.7x the CDF bytes (41 GB
# for one month, 2022-08). They now live in a scratch folder per process,
#     <pickle_root>/fft_scratch/<os.name>-<pid>-<stamp>/mms/...    (the old layout below that point)
# deleted when the last fft_scratch_scope on the root ends (the event window's Run), by drop_fft_scratch (a census
# day), at interpreter exit, and -- for a process that died without cleaning -- by the next process to use the root.
# Within one request the pickles are still how the worker pool hands spectra to the export and image steps, so
# they are kept until the request ends; a second request computes its FFTs again.
FFT_SCRATCH_DIR = 'fft_scratch'
FFT_SCRATCH_ENV = 'PLASMACOOP_FFT_SCRATCH_TAG'
_fft_scratch_lock = threading.Lock()
_fft_tag_lock = threading.Lock()   # its own lock: fft_scratch_dir asks for the tag while holding _fft_scratch_lock
_fft_scratch_tag = None
_fft_scratch_roots = set()     # normalised roots this process has handed a scratch path under
_fft_scratch_users = {}        # normalised root -> open fft_scratch_scope count


def _scratch_tag():
	"""<os.name>-<pid>-<stamp>, one per process. A multiprocessing worker takes its parent's from the environment, so
	any path it computes names the parent's folder. Any other program that inherits the variable (a subprocess the
	app starts) ignores it and makes its own."""
	global _fft_scratch_tag
	if _fft_scratch_tag is None:
		with _fft_tag_lock:            # two threads asking first must not make two tags a second apart
			if _fft_scratch_tag is None:
				inherited = os.environ.get(FFT_SCRATCH_ENV)
				if inherited and multiprocessing.parent_process() is not None:
					_fft_scratch_tag = inherited
				else:
					_fft_scratch_tag = f'{os.name}-{os.getpid()}-{datetime.now():%Y%m%dT%H%M%S}'
					os.environ[FFT_SCRATCH_ENV] = _fft_scratch_tag
	return _fft_scratch_tag


def _scratch_owner(tag):
	"""(os.name, pid) a scratch folder name was made by, or (None, None) for a name that is not a scratch tag."""
	parts = tag.split('-')
	try:
		return parts[0], int(parts[1])
	except (IndexError, ValueError):
		return None, None


def _owns_fft_scratch():
	"""True in the process that made this process's tag; False in a multiprocessing worker, which never deletes."""
	return _scratch_owner(_scratch_tag()) == (os.name, os.getpid())


def _rmtree_quiet(path):
	"""Delete a folder tree without ever raising, and return the bytes removed. A file another process still holds
	open (Windows) stays, and so does its folder; the next sweep or drop takes it."""
	freed = 0
	for dirpath, _dirs, names in os.walk(path, topdown=False):
		for name in names:
			p = os.path.join(dirpath, name)
			try:
				size = os.path.getsize(p)
				os.remove(p)
				freed += size
			except OSError:
				pass
		try:
			os.rmdir(dirpath)
		except OSError:
			pass
	return freed


def _sweep_dead_fft_scratch(root):
	"""Delete the scratch folders of processes that ended without cleaning (a crash, a kill): made on this OS, by a
	pid that is not running. post_verify._alive is the liveness check (OpenProcess on Windows, where os.kill(pid, 0)
	would end the process). A WSL folder and a Windows folder on the same drive never judge each other, since their
	pids come from different systems. A pid reused by a new process keeps its folder until that one ends too, which
	only delays the delete."""
	base = os.path.join(root, FFT_SCRATCH_DIR)
	try:
		names = os.listdir(base)
	except OSError:
		return
	from post_verify import _alive
	for name in names:
		osn, pid = _scratch_owner(name)
		if osn != os.name or pid is None or pid == os.getpid():
			continue
		if not _alive(pid):
			_rmtree_quiet(os.path.join(base, name))


def fft_scratch_dir(pickle_root):
	"""This process's FFT scratch folder under pickle_root. Creates nothing (the writer makes the folders it needs).
	The first call for a root registers it for deletion at exit and sweeps dead processes' scratch from it."""
	key = os.path.normpath(pickle_root)
	if key not in _fft_scratch_roots:
		with _fft_scratch_lock:
			if key not in _fft_scratch_roots:
				_fft_scratch_roots.add(key)
				if _owns_fft_scratch():
					_sweep_dead_fft_scratch(pickle_root)
	return os.path.join(pickle_root, FFT_SCRATCH_DIR, _scratch_tag())


def drop_fft_scratch(pickle_root, force=False):
	"""Delete this process's FFT scratch under pickle_root now and return the bytes freed. Only the process that owns
	it deletes (a multiprocessing worker gets 0). While a fft_scratch_scope is open on the root nothing is deleted
	(0): that request is still reading its pickles, and the scope deletes when it ends. force: delete regardless
	(exit only)."""
	if not _owns_fft_scratch():
		return 0
	if not force and _fft_scratch_users.get(os.path.normpath(pickle_root), 0) > 0:
		return 0
	# The empty fft_scratch parent stays: removing it could race another process making its first folder there.
	return _rmtree_quiet(os.path.join(pickle_root, FFT_SCRATCH_DIR, _scratch_tag()))


@contextmanager
def fft_scratch_scope(pickle_root):
	"""One request's use of the FFT scratch (the event window's Run). The scratch is deleted when the last open scope
	on the root ends, so two requests overlapping in one process never delete each other's FFTs. Yields a dict whose
	'freed' is the bytes deleted at the end (0 when another scope was still open)."""
	key = os.path.normpath(pickle_root)
	result = {'freed': 0}
	with _fft_scratch_lock:
		_fft_scratch_users[key] = _fft_scratch_users.get(key, 0) + 1
	try:
		yield result
	finally:
		with _fft_scratch_lock:
			_fft_scratch_users[key] -= 1
			last = _fft_scratch_users[key] == 0
		if last:
			result['freed'] = drop_fft_scratch(pickle_root)


def _drop_all_fft_scratch():
	for key in list(_fft_scratch_roots):
		try:
			drop_fft_scratch(key, force=True)
		except Exception:
			pass


atexit.register(_drop_all_fft_scratch)


def _fft_pickle_path(pickle_root, inst, mode, pad, file_time, probe: int = 1, step: float = 0.05):
	"""
	Path for the per-file FFT result pickle.
	{scratch}/mms/{probe}/{inst}/{mode}/fft/pad{pad:.3f}[_step{step:.3f}]_full/{YYYY}/{MM}/{DD}/{HHMMSS}.pkl
	where {scratch} is this process's fft_scratch_dir(pickle_root), deleted when the request ends (see the FFT
	scratch block above; until 2026-11-19 it was pickle_root itself, and the pickles stayed between requests).
	Each pad value gets its own subdirectory since it determines frequency resolution.

	step is in the key too, whenever it is not 0.05, because the rows sit one window
	centre per step. A pickle built at one step and read at another hands the binner
	FFTs at the wrong spacing: a 0.05 s grid over 0.10 s rows leaves every other column
	empty, and a 0.10 s grid over 0.05 s rows averages two FFTs into each column. Either
	way it breaks the rule that every display column holds exactly one FFT (Graham,
	2026-09-29). Keyed on pad alone, spike_biggest_burst_fft (STEP 0.10) and
	plot_biggest_spikes / event_export (0.05) shared every pad-0.25 pickle, and whoever
	built a file first decided its step for both. 0.05 keeps the bare pad{pad}_full
	name, so the existing cache -- all of it built at 0.05 -- stays valid.

	lpf_lim is in the path because it determines the pickle's CONTENT, not just
	its presentation: _compute_file_ffts masks rows to frequency <= lpf_lim, so a
	pickle built at 1000 Hz holds strictly fewer rows than one built at 3000 Hz.
	Keyed on pad alone, an entry written under one ceiling was silently reused
	under another -- raising FIELD_LPF_LIM in plot_biggest_spikes.py from
	{'B': 1000, 'E': 500} to {'B': 3000, 'E': 3000} left every already-cached
	pickle looking valid while truncating its spectrum at the OLD ceiling.
	Nothing detected it: _invalidate_if_stale compares source mtimes (the CDF
	had not changed) and FFT_PICKLE_VERSION (the code had not changed), and
	neither signal can see that a PARAMETER changed. Confirmed on disk before
	the fix -- .../scm/scb/fft/pad0.250/2020/01/31/210323.pkl held 558,978 rows
	topping out at 1000 Hz while its 209 siblings reached 3000 Hz.

	The ceiling is no longer part of this key, and the paragraph above is why it
	once had to be: while a pickle was truncated at lpf_lim, the ceiling really
	did decide its content. The pickles now hold the FULL spectrum to Nyquist,
	so there is exactly one entry per (pad, file) whatever ceiling is asked for,
	and the ceiling is applied where the rows are binned instead (see
	_FftGrid.add and _bin_fft_dataframe in short_interval.py). One consequence
	worth stating: raising the ceiling no longer rebuilds anything.

	The suffix is _full and NOT a bare pad{pad}, even though nothing else uses
	that name now. The pre-lpf layout was pad-only, so reusing it would make a
	legacy pickle -- truncated at whatever ceiling was current when it was
	written -- indistinguishable on disk from a full-spectrum one, and it would
	pass every existence check while silently cutting the spectrum short.
	"""
	sub = f'pad{pad:.3f}_full' if f'{step:.3f}' == '0.050' else f'pad{pad:.3f}_step{step:.3f}_full'
	return os.path.join(
		fft_scratch_dir(pickle_root),
		f'mms/{probe}/{inst}/{mode}/fft/{sub}',
		file_time.strftime('%Y/%m/%d/%H%M%S.pkl'),
	)


# Adopt pre-fix (pad-only) FFT pickles into the lpf-keyed layout instead of
# recomputing them. Set False to skip the migration entirely and just rebuild.
# OFF as of the full-spectrum change. Adoption exists to migrate pre-lpf-key
# pickles into the lpf-keyed layout, and both of those layouts are now history:
# FFT pickles hold the whole spectrum under a pad{pad}_full key. A legacy entry
# truncated at some old ceiling must never be adopted as a full-spectrum one --
# it would pass every existence check while silently stopping the spectrum
# short. Left in place rather than deleted so the migration stays readable if an
# old cache ever has to be reconstructed.
ADOPT_LEGACY_FFT_CACHE = False


def _legacy_fft_path(fft_pkl):
	"""The pre-fix pad-only path corresponding to an lpf-keyed one, or None.

	String surgery on the pad directory rather than re-deriving from
	(pickle_root, inst, mode, pad, file_time, probe): the callers that need this
	(_compute_and_save_fft, get_or_compute_fft) are handed a finished path and
	never see those components separately.
	"""
	head, tail = os.path.split(fft_pkl)                       # .../HHMMSS.pkl
	rest, dd   = os.path.split(head)
	rest, mm   = os.path.split(rest)
	rest, yyyy = os.path.split(rest)
	base, sub  = os.path.split(rest)                           # sub = padX_lpfY
	if '_lpf' not in sub:
		return None
	return os.path.join(base, sub.split('_lpf')[0], yyyy, mm, dd, tail)


def _adopt_legacy_fft_pickle(fft_pkl, pad, lpf_lim):
	"""Move a pre-fix FFT pickle into its lpf-keyed home when its content really
	was computed at this ceiling, so adding lpf_lim to the cache key does not
	discard a cache that took hours to build.

	Adoptable in exactly two cases:
	  * non-empty and its max frequency equals the top grid frequency for this
	    (pad, lpf_lim) -- proof it was built at THIS ceiling, since
	    _compute_file_ffts always emits every grid bin up to lpf_lim (Nyquist,
	    4096 Hz at SAMPLE_RATE, is above every ceiling in use, so the mask and not the
	    data decides the top bin);
	  * empty -- an empty result is empty at any ceiling. It means no window in
	    that file cleared the sample threshold (see _compute_and_save_fft), which
	    is a property of the data, not of lpf_lim.

	Anything else is left where it is for the normal compute path to rebuild.
	The legacy file is MOVED, not copied (these run to gigabytes), and never
	deleted: a non-matching legacy entry stays readable at its old path.
	"""
	if not ADOPT_LEGACY_FFT_CACHE:
		return False
	legacy = _legacy_fft_path(fft_pkl)
	if not legacy or not os.path.exists(legacy):
		return False

	Df = 1.0 / (pad * 2)
	expected_max = int(np.floor(lpf_lim / Df) * Df)
	try:
		ldf = pd.read_pickle(legacy)
		if not ldf.empty and int(ldf['frequency'].max()) != expected_max:
			return False
	except Exception:
		return False

	os.makedirs(os.path.dirname(fft_pkl), exist_ok=True)
	try:
		os.replace(legacy, fft_pkl)
	except OSError:
		return False
	try:
		os.replace(_version_path(legacy), _version_path(fft_pkl))
	except OSError:
		pass
	return True


def _compute_file_ffts(curr_path, next_path, key, inst, mode, fs, pad, n_expected, Df, file_time, next_file_time, step=0.05, jitter_factor=None, prev_path=None, prev_file_time=None):
	"""
	Compute FFT windows whose centers lie in [file_time, next_file_time).

	Three files are in play and the middle one owns the centres: curr_path
	supplies the samples, and prev_path and next_path are opened on demand only
	when a window overhangs one end of curr or the other. All three are burst
	CDFs read by read_burst_signal -- |field| per sample, from the preprocessed
	archive when it holds the file (a data pickle still works). Nothing more than
	the two neighbours is ever held, which is the whole design -- a window is at
	most 2*pad wide and files are far longer, so it can never reach past the file
	next door.

	The next file is read only as far as next_file_time + pad, so no borrowed
	sample is later than the centre range. The previous file is read whole,
	because read_burst_data can stop a read early but cannot start one late; it
	is read at most once per file and only when a window actually reaches back.

	Per-window policy -- a window is computed ONLY when real samples fill it
	(Graham, 2026-09-15: "there shouldn't be a cutoff in data in the MIDDLE of
	the data used for the FFT"), and it reaches in BOTH directions to fill it
	(Graham, 2026-09-25: "on that first FFT should also absolutely reach back for
	the needed points and similarly drop the window if it isn't full enough"):
	  n == 0                          skip entirely (no data in window)
	  window starts before curr's first sample -> borrow curr's head from prev
	  window ends after curr's last sample     -> borrow curr's tail from next
	    combined n >= n_full          compute FFT with frequency snapping
	    combined n <  n_full          skip
	A window can be short at both ends at once (a file shorter than 2*pad), so
	both borrows run, back first then forward.

	Every row therefore stands for a full 2*pad window, whatever produced it.
	Windows used to be computed on as little as half a window's samples when
	no next file was in range, so a row's meaning depended on where it sat.
	Borrowing was forward-only until 2026-09-25, which cost the first pad/step
	windows of EVERY file -- 0.3 s of columns at each seam at pad 0.25, step
	0.05, on data with no gap in it at all.

	n_full, NOT the n_expected argument, is what "filled" means: n_expected
	comes from the nominal sample rate, which is 14 ppm faster than the clock
	that actually wrote the timestamps, so it overstates a gap-free window's
	sample count by one about 6% of the time. n_full is measured from this
	file's own timestamps instead; see where it is derived below. gap_factor is
	the sample-period margin that tells a real hole from timestamp jitter,
	deliberately NOT the segment threshold -- see CADENCE_JITTER_FACTOR for why
	sharing them would quietly loosen the fill test.
	"""
	# |field| per sample, once per file (read_burst_signal): from the archive when it holds the file.
	curr_time_ns, curr_s_arr = read_burst_signal(curr_path, key, fs=fs)
	if curr_time_ns.size == 0:
		return pd.DataFrame(columns=['time', 'frequency', 'power'])

	next_read         = False
	next_time_ns      = None
	next_s_arr        = None

	prev_time_ns      = None
	prev_s_arr        = None
	prev_tried        = False         # a prev that turns out empty must not be reread

	step_ns           = int(step * 1e9)
	pad_ns            = int(pad  * 1e9)
	file_time_ns      = np.datetime64(file_time, 'ns').astype(np.int64).item()
	# int(...) on curr_time_ns[0] forces a plain Python int, not numpy.int64 --
	# under numpy 2.x, np.datetime64(x, 'ns') raises "Could not convert object
	# to NumPy datetime" for a bare numpy.int64 scalar (confirmed: works for a
	# plain int, fails for np.int64(same value)). start_ns feeds center_ns
	# through the whole sliding-window loop below via center_ns += step_ns, so
	# whichever type wins this max() propagates through every np.datetime64()
	# call downstream -- must stay a plain int unconditionally, not just when
	# file_time_ns (already plain, via .item()) happens to be the larger one.
	start_ns          = max(file_time_ns, int(curr_time_ns[0]) - pad_ns)
	next_file_time_ns = (np.datetime64(next_file_time, 'ns').astype(np.int64).item()
	                     if next_file_time is not None else None)
	data_end_ns = int(curr_time_ns[-1]) + pad_ns + step_ns
	end_ns      = min(next_file_time_ns, data_end_ns) if next_file_time_ns is not None else data_end_ns

	# How many samples a GAP-FREE 2*pad window actually holds, measured from
	# this file's own timestamps instead of assumed from the nominal rate.
	#
	# n_expected is int(fs * 2 * pad) -- 4096 at fs=8192, pad=0.25 -- and fs is
	# the NOMINAL 8192. The spacecraft clock runs ~14 ppm slow, so the real
	# period is 122072 ns / 8191.887 Hz (see the SAMPLE_RATE comment), which makes a
	# 5e8 ns window 4095.94 periods wide. The sample count inside it is
	# therefore 4095 or 4096 purely according to where its edges fall between
	# two samples -- never a fixed 4096. Measured on a real file
	# (mms1 scb 20190816010003, one 97.9 s gap-free run): 119 of 1949 interior
	# windows hold 4095, 1830 hold 4096. Testing "is this window filled?"
	# against the nominal 4096 therefore threw away 6.1% of windows that had no
	# gap in them at all.
	#
	# The drop rule itself is right and stays -- a window that cannot be filled
	# with real samples is not computed. This only stops the nominal rate from
	# lying about what "filled" means.
	#
	# MEAN, not median. Epoch timestamps reach us quantised to whole
	# microseconds, so the real 122072 ns period arrives as a mix of 122000
	# (93%) and 123000 (7%). The median is the mode, 122000, which implies 4098
	# samples per window and would drop EVERY window; the mean recovers
	# 122072.4 and implies 4095, which drops none. Verified both ways on the
	# file above before choosing.
	#
	# Gaps are excluded by CADENCE_JITTER_FACTOR -- 1.5 sample periods, which is
	# the widest two ADJACENT microsecond-quantised samples can sit apart -- so a
	# hole in this file cannot inflate the period. This is not the segment
	# threshold and must not be wired to it; see that constant.
	# min() keeps the threshold from ever rising above the nominal count, so a
	# mis-measured period can only ever be more permissive, never stricter. A
	# real hole is a whole missing burst, thousands of samples wide, and cannot
	# hide inside the one-sample spread between 4095 and 4096.
	n_full = n_expected
	if curr_time_ns.size > 1:
		_d      = np.diff(curr_time_ns)
		_jf     = CADENCE_JITTER_FACTOR if jitter_factor is None else jitter_factor
		_d      = _d[_d <= _jf * 1e9 / fs]
		if _d.size:
			n_full = min(n_expected, int((2 * pad_ns) // _d.mean()))

	# The whole one-sided grid, DC to Nyquist, with no ceiling applied: this
	# sizes the row preallocation below, so it has to cover every bin the mask
	# can possibly keep.
	freq_grid = (np.round(np.arange(0, (fs / 2) / Df + 1)) * Df).astype(int)

	n_steps      = (end_ns - start_ns) // step_ns + 2
	max_per_step = len(freq_grid)
	max_rows     = n_steps * max_per_step
	t_arr = np.empty(max_rows, dtype='datetime64[ns]')
	f_arr = np.empty(max_rows, dtype=np.int16)
	p_arr = np.empty(max_rows, dtype=np.float32)
	row   = 0

	center_ns = start_ns
	print(f'Computing FFTs for {file_time}...')

	while center_ns < end_ns:
		fft_start_ns = center_ns - pad_ns
		fft_end_ns   = center_ns + pad_ns

		lo  = np.searchsorted(curr_time_ns, fft_start_ns, side='left')
		hi  = np.searchsorted(curr_time_ns, fft_end_ns,   side='right')
		n   = hi - lo
		center_dt64 = np.datetime64(center_ns, 'ns')

		if n == 0:
			center_ns += step_ns
			continue

		if n < n_full:
			# Head of the window: does it start before this file's first sample?
			# Only then is there anything for the previous file to supply.
			back_in_window = (prev_path is not None and fft_start_ns < int(curr_time_ns[0]))

			if back_in_window:
				if not prev_tried:
					prev_tried = True
					_t, _s = read_burst_signal(prev_path, key, fs=fs)
					if _t.size:
						prev_time_ns, prev_s_arr = _t, _s
				if prev_time_ns is not None:
					first_time_ns = int(curr_time_ns[lo])
					n_needed  = n_full - n
					lo_p = np.searchsorted(prev_time_ns, fft_start_ns,  side='left')
					hi_p = np.searchsorted(prev_time_ns, first_time_ns, side='left')
					# Take the samples NEAREST the seam, so what is borrowed is
					# contiguous with curr's head rather than a detached lump
					# from the far side of a hole in the previous file.
					lo_p_clip = max(lo_p, hi_p - n_needed)
					if lo_p_clip < hi_p:
						s_win = np.concatenate([prev_s_arr[lo_p_clip:hi_p], curr_s_arr[lo:hi]])
						n = len(s_win)

			# Tail of the window, on whatever the head borrow left short. A file
			# shorter than 2*pad is short at both ends, so this is not an elif.
			next_in_window = (next_file_time_ns is not None and next_file_time_ns <= fft_end_ns)

			if n < n_full and next_in_window:
				if not next_read:
					next_read = True
					next_time_ns, next_s_arr = read_burst_signal(next_path, key, until_ns=next_file_time_ns + pad_ns,
					                                             fs=fs)   # seam only: nothing later is borrowed
				last_time_ns = curr_time_ns[hi - 1]
				n_needed     = n_full - n
				lo_n = np.searchsorted(next_time_ns, last_time_ns, side='right')
				hi_n = np.searchsorted(next_time_ns, fft_end_ns,   side='right')
				hi_n_clip = min(lo_n + n_needed, hi_n)
				if lo_n < hi_n_clip:
					# head is whatever the backward borrow produced, if any
					head_s = s_win if n != hi - lo else curr_s_arr[lo:hi]
					s_win = np.concatenate([head_s, next_s_arr[lo_n:hi_n_clip]])
					n = len(s_win)

			if n < n_full:
				# Still short after reaching both ways: either the burst
				# genuinely starts or ends here, or the shortfall is a gap
				# inside this file. Either way the window cannot be filled with
				# real samples, so it is not computed (Graham, 2026-09-21: drop
				# it, rather than keep a half-window as this branch used to).
				center_ns += step_ns
				continue

		if n == hi - lo:  # no concatenation happened, use a direct slice
			s_win = curr_s_arr[lo:hi]
		signal = s_win - np.mean(s_win)

		w = np.hanning(n)
		u = np.mean(w**2)  # mean square of Hann window

		signal_fft      = np.fft.rfft(signal * w)
		signal_fft_freq = np.fft.rfftfreq(n, d=1/fs)
		signal_fft_freq = (np.round(signal_fft_freq / Df) * Df).astype(int)

		signal_psd     = (np.abs(signal_fft)**2) / (fs * u * n)
		# One-sided correction: double each bin that has a negative-frequency
		# twin. DC never does, and neither does the Nyquist bin (n/2) of an
		# even-length window -- doubling it overstated that bin 2x and put
		# Parseval off by its share. Odd n has no Nyquist bin.
		top = len(signal_psd) - 1 if n % 2 == 0 else len(signal_psd)
		signal_psd[1:top] = signal_psd[1:top] * 2

		# Keep the entire one-sided spectrum. The ceiling used to be applied
		# right here, which is precisely what baked it into the cache; it is
		# applied at binning time now. The upper bound is the grid's own top bin
		# so a snapped frequency can never fall outside the preallocation.
		mask  = (signal_fft_freq >= 0) & (signal_fft_freq <= freq_grid[-1])
		f_sel = signal_fft_freq[mask]
		P_sel = signal_psd[mask]

		k = len(f_sel)
		t_arr[row:row+k] = center_dt64
		f_arr[row:row+k] = f_sel
		p_arr[row:row+k] = P_sel
		row += k

		center_ns += step_ns

	if row == 0:
		return pd.DataFrame(columns=['time', 'frequency', 'power'])

	return pd.DataFrame({
		'time':      t_arr[:row].copy(),
		'frequency': f_arr[:row].copy(),
		'power':     p_arr[:row].copy(),
	})


def _compute_and_save_fft(file_args, key, inst, mode, fs, pad, n_expected, Df, step=0.05):
	"""
	Worker function for parallel FFT computation.
	file_args: (curr_path, next_path, file_time, next_file_time, fft_pkl,
	            prev_path, prev_file_time) -- the last two optional

	Always writes fft_pkl (+ version sidecar), even when the result has zero
	rows -- an empty pickle on disk IS the cache marker for "already
	computed, no FFT windows here". Short/edge-of-burst files legitimately
	produce zero rows now that insufficient-data windows are skipped rather
	than zero-filled (see _compute_file_ffts' "no zero-power gaps" policy).
	Previously the write was skipped whenever the result was empty, so
	os.path.exists(fft_pkl) was never true for those files -- every re-run
	recomputed them from scratch, printing "Computing FFTs for ..." again
	despite having already done the work. Returns fft_pkl; callers read it
	and skip empty DataFrames themselves (e.g. gen_fft's
	`if file_fdf.empty: continue`).
	"""
	curr_path, next_path, file_time, next_file_time, fft_pkl = file_args[:5]
	# prev came later than the rest of this tuple, so a 5-element file_args from
	# older code (or a scratch script) still works and simply gets no backward
	# borrow. See pair_with_next_cdf for the shape callers build now.
	prev_path, prev_file_time = file_args[5:7] if len(file_args) >= 7 else (None, None)

	# Sources are ALL THREE burst CDFs this FFT reads: its own file, the next
	# one, whose leading samples close this file's trailing window, and the
	# previous one, whose trailing samples open its leading window (see
	# _compute_file_ffts). Including the neighbours is what makes a newly
	# arrived adjacent file correctly invalidate this file's FFT -- that seam
	# was previously computed against "no such file" and silently kept.
	# No legacy adoption -- see ADOPT_LEGACY_FFT_CACHE. An entry written under an
	# older layout was truncated at whatever ceiling was current when it was
	# built, so it can never stand in for a full-spectrum one.
	_invalidate_if_stale(fft_pkl, FFT_PICKLE_VERSION[inst],
	                     source_paths=tuple(
	                         q for q in (curr_path, next_path, prev_path) if q))

	if os.path.exists(fft_pkl):
		return fft_pkl
	file_fdf = _compute_file_ffts(
		curr_path, next_path, key, inst, mode, fs,
		pad, n_expected, Df, file_time, next_file_time, step,
		prev_path=prev_path, prev_file_time=prev_file_time
	)
	os.makedirs(os.path.dirname(fft_pkl), exist_ok=True)
	file_fdf.to_pickle(fft_pkl)
	_write_version(fft_pkl, FFT_PICKLE_VERSION[inst])
	if not file_fdf.empty:
		print(f'Saved FFT pickle: {fft_pkl}')
	return fft_pkl


def get_or_compute_fft(file_args, key, inst, mode, fs, pad, n_expected, Df, step=0.05):
	"""In-process sibling of _compute_and_save_fft that returns the actual
	FFT DataFrame instead of just the pickle path.

	_compute_and_save_fft returns a path because it's built for the
	ProcessPoolExecutor path (compute_ffts/gen_fft/prewarm) -- the DataFrame
	lives in a worker process and would have to be re-serialized over IPC to
	get it back regardless, so callers there read the (already-necessary)
	pickle themselves, on their own schedule. This function is for the
	SINGLE-PROCESS synchronous path (short_interval.py, few files, no pool):
	there the computed DataFrame is already sitting in this process's memory
	the moment _compute_file_ffts returns it, so writing it to disk and then
	immediately reading it straight back (as gen_fft-style callers must) is
	pure waste -- serializing for the cache is still done as a side effect
	(future runs/processes still hit it), but THIS call uses the in-memory
	result directly. On a cache HIT, a disk read is unavoidable (that data
	was computed by someone else -- a prior run, or a separate worker
	process -- and was never in this process's memory to begin with).
	"""
	curr_path, next_path, file_time, next_file_time, fft_pkl = file_args[:5]
	# prev came later than the rest of this tuple, so a 5-element file_args from
	# older code (or a scratch script) still works and simply gets no backward
	# borrow. See pair_with_next_cdf for the shape callers build now.
	prev_path, prev_file_time = file_args[5:7] if len(file_args) >= 7 else (None, None)

	# Sources are ALL THREE burst CDFs this FFT reads: its own file, the next
	# one, whose leading samples close this file's trailing window, and the
	# previous one, whose trailing samples open its leading window (see
	# _compute_file_ffts). Including the neighbours is what makes a newly
	# arrived adjacent file correctly invalidate this file's FFT -- that seam
	# was previously computed against "no such file" and silently kept.
	# No legacy adoption -- see ADOPT_LEGACY_FFT_CACHE. An entry written under an
	# older layout was truncated at whatever ceiling was current when it was
	# built, so it can never stand in for a full-spectrum one.
	_invalidate_if_stale(fft_pkl, FFT_PICKLE_VERSION[inst],
	                     source_paths=tuple(
	                         q for q in (curr_path, next_path, prev_path) if q))

	if os.path.exists(fft_pkl):
		return pd.read_pickle(fft_pkl)

	file_fdf = _compute_file_ffts(
		curr_path, next_path, key, inst, mode, fs,
		pad, n_expected, Df, file_time, next_file_time, step,
		prev_path=prev_path, prev_file_time=prev_file_time
	)
	os.makedirs(os.path.dirname(fft_pkl), exist_ok=True)
	file_fdf.to_pickle(fft_pkl)
	_write_version(fft_pkl, FFT_PICKLE_VERSION[inst])
	if not file_fdf.empty:
		print(f'Saved FFT pickle: {fft_pkl}')
	return file_fdf


def _parse_cdf_time(path):
	"""14-digit filename timestamp embedded in a burst CDF path, as a datetime."""
	m = re.search(r'_(\d{14})_', path)
	return datetime.strptime(m.group(1), '%Y%m%d%H%M%S')


def slice_cdf_files(cdf_files, start, end):
	"""Slice an already-discovered cdf_files list (e.g. the other half of
	prewarm()'s return value) down to the files relevant to [start, end],
	using the cheap filename-timestamp anchor (_parse_cdf_time +
	_anchor_index) -- no filesystem re-scan, no CDF opens. Manifest-writing
	still clips to the exact window afterward regardless, but pre-slicing
	here avoids opening (and quality-checking) files from far outside the
	narrow window just because they were part of a broader prewarmed
	request."""
	timed = sorted((_parse_cdf_time(p), p) for p in cdf_files)
	anchor = _anchor_index(timed, start)
	if anchor is None:
		return [p for _, p in timed]
	return [p for ftime, p in timed[anchor:] if ftime <= end]


def _ensure_data_pickle_job(job, pickle_root, corrupted_log, lock):
	cdf_path, key, probe = job
	return ensure_data_pickle(cdf_path, pickle_root, key, corrupted_log=corrupted_log, lock=lock, probe=probe)


def ensure_data_pickles(jobs, pickle_root, corrupted_log=None, max_workers=MAX_POOL_WORKERS):
	"""Batch, deduplicated version of ensure_data_pickle -- runs every
	(cdf_path, key, probe) job across ONE shared ProcessPoolExecutor pool.

	Callers spanning multiple overlapping requests/fields/probes (see
	prewarm below) must deduplicate `jobs` by (cdf_path, key, probe)
	themselves before calling -- the same underlying pickle must never be
	submitted, and never written, twice concurrently. This function does not
	dedupe on your behalf; it just fans whatever you give it out to workers.

	Returns pickle paths in the same order as jobs (None where the source
	CDF was corrupt)."""
	with Manager() as manager:
		lock = manager.Lock()
		fn = partial(_ensure_data_pickle_job, pickle_root=pickle_root, corrupted_log=corrupted_log, lock=lock)
		return _pool_map(fn, jobs, max_workers, 'CDF -> data pickle conversion')


def _compute_fft_job(job):
	file_args, key, inst, mode, fs, pad, n_expected, Df, step, stream = job
	file_time = file_args[2]
	if stream:
		df = get_or_compute_fft(file_args, key=key, inst=inst, mode=mode, fs=fs,
		                        pad=pad, n_expected=n_expected, Df=Df, step=step)
		return file_time, df
	# Cache-warm only -- no window asked for this file's data back in memory.
	# Still computes and writes the FFT pickle (same as the stream branch),
	# but returns None instead of the DataFrame so the (potentially huge,
	# see _compute_file_ffts' max_rows preallocation) array never gets
	# pickled over the IPC boundary or accumulated in the parent process.
	# The array still exists transiently inside THIS worker while
	# _compute_and_save_fft builds and writes it, same peak per-worker cost
	# as the stream branch -- what's eliminated is the parent-side pileup
	# across every file in a wide cache-only window (the MemoryError cause).
	_compute_and_save_fft(file_args, key=key, inst=inst, mode=mode, fs=fs,
	                      pad=pad, n_expected=n_expected, Df=Df, step=step)
	return file_time, None


def compute_ffts(jobs, max_workers=MAX_POOL_WORKERS):
	"""Batch, deduplicated version of get_or_compute_fft -- runs every FFT
	job across ONE shared ProcessPoolExecutor pool. Each job is
	(file_args, key, inst, mode, fs, pad, n_expected, Df, step, stream).

	Returns [(file_time, dataframe_or_None), ...] in job order. When a job's
	`stream` flag is True, the actual computed (or cache-read) DataFrame
	comes back -- crossing the process boundary already means the result
	gets pickled once over IPC regardless, so handing back the real
	DataFrame there is the ONLY serialization round trip: a caller that
	wants the data (e.g. prewarm, assembling it for a plotter) never has to
	separately re-open the very file this call just wrote. When `stream` is
	False, the file is still computed and cached to disk, but None comes
	back instead -- the caller only wanted the cache warmed, not the data
	returned, so paying to ship a large array over IPC (and hold it in this
	process alongside every other job's result) would be pure waste, and at
	wide-window scale is what exhausts memory.

	Callers spanning multiple overlapping requests must deduplicate jobs by
	the fft_pkl path inside file_args first (see prewarm below) -- this
	function just fans out whatever it's given."""
	return _pool_map(_compute_fft_job, jobs, max_workers, 'FFT computation')


def stream_ffts(jobs, max_workers=MAX_POOL_WORKERS):
	"""Streaming sibling of compute_ffts: YIELDS each (file_time,
	dataframe_or_None) the instant its worker finishes, instead of listing the
	whole pool's output before returning any of it. Same job tuple, same pool,
	same per-job result -- the only difference is `yield from` in place of
	`list(...)`.

	ProcessPoolExecutor.map already submits every job eagerly (workers run
	ahead) and returns a lazy iterator that hands results back in job order as
	they complete, so a consumer can start drawing one burst's chunk while the
	pool is still computing the next files. That overlap is the whole point:
	the per-good-chunk plotting path (plot_good_chunks in short_interval.py)
	draws one figure per qf-good data chunk, and drawing is far slower than the
	FFT read/compute -- so the plotting thread stays busy on the chunk just
	handed to it while the pool keeps the next files warming in the background,
	instead of the old compute-EVERYTHING-then-draw-EVERYTHING two-phase wait.

	In-order (not completion-order) delivery is deliberate: a caller that
	submits a burst's files time-ordered gets them back time-ordered, so a
	streaming clusterer can close and emit each burst as its last file lands.

	Memory note: map submits everything up front, so completed results pile up
	as fast as workers produce them if the consumer lags (the MemoryError class
	this project has hit -- see plot_biggest_spikes.py). Fine for per-chunk
	mode, which draws-and-discards each chunk and is meant for inspecting a
	handful of bursts, not a multi-day span; a caller pointing this at hundreds
	of files would batch the job list over one reused pool instead."""
	try:
		with ProcessPoolExecutor(max_workers=max_workers) as pool:
			yield from pool.map(_compute_fft_job, jobs)
	except BrokenProcessPool as exc:
		# Streamed sibling of _pool_map -- same failure, same guidance; it
		# cannot use that helper because results are yielded, not collected.
		raise RuntimeError(
			f'streamed FFT computation: a worker process died before returning '
			f'a result (max_workers={max_workers}). See _pool_map for the usual '
			f'cause and fix (PLASMA_MAX_WORKERS, or the Windows pagefile).'
		) from exc


def _cluster_good_segments(segs, tol):
	"""Cluster (good_start, good_end, file_time, probe) qf-good segments into
	bursts by TIME-SPAN OVERLAP within `tol`, across every probe and file --
	the robust "same burst" test when per-probe file start times differ. MMS
	commands all four probes over the same interval, so their good-data spans
	overlap even when the individual start stamps don't; distinct bursts sit
	far apart in time, so nothing false-merges. A later segment joins the
	running cluster when it starts no more than `tol` after the cluster's
	current end; a gap larger than `tol` starts a new cluster. There is NO
	reference probe -- the cluster is defined purely by which spans touch.

	Returns a list of (x0, x1, needs):
	  x0    -- earliest good start in the cluster (the image's t=0 origin)
	  x1    -- latest good end in the cluster
	  needs -- set of (file_time, probe) whose FFT DataFrame the cluster's
	           image is drawn from (a file split into two qf runs by a bad
	           stretch wider than `tol` appears in two adjacent clusters)."""
	clusters = []
	x0 = x1 = None
	needs = set()
	for cs, ce, ft, probe in sorted(segs, key=lambda s: s[0]):
		if x1 is not None and cs <= x1 + tol:
			x1 = max(x1, ce)
			needs.add((ft, probe))
		else:
			if x1 is not None:
				clusters.append((x0, x1, needs))
			x0, x1, needs = cs, ce, {(ft, probe)}
	if x1 is not None:
		clusters.append((x0, x1, needs))
	return clusters


def _plan_fft_jobs(data_dir, pickle_root, specs, step=0.05, max_workers=MAX_POOL_WORKERS,
                   corrupted_log=None, download=False):
	"""Discovery + CDF checks + per-file FFT job construction, shared by prewarm
	(batch compute) and stream_good_chunk_ffts (streamed compute) so neither
	re-implements it. Each spec is (key, probe, pad, lpf_lim, windows), windows
	an iterable of (st, ts, stream): find_cdf_files across every spec, validating
	as it discovers (see cdf_check -- an unreadable CDF is deleted and logged
	before any job is planned on it, and before the manifest step sees the lists
	returned here), then find_local_cdfs per window and one FFT job per unique
	file, its seam neighbours validated too. Jobs carry CDF paths: the FFT workers read the
	CDFs directly, and no data pickle is written. pickle_root is where the FFT
	pickles and good-run sidecars live. max_workers sized the old conversion pool;
	the check runs in this process, so it is unused and kept for callers.

	Returns (fft_jobs, fft_owner, manifest_specs):
	  fft_jobs       {fft_pkl: job_tuple}    -- deduped across overlapping
	                                            windows/specs (same file -> one job)
	  fft_owner      {fft_pkl: (key, probe)} -- to regroup flat results afterward
	  manifest_specs [(key, inst, mode, probe, pad, fs, window_files), ...]
	                                         -- carries the discovered cdf lists
	                                            for the manifest / qf-segment step
	A job's `stream` flag is the OR across every window that needs the file."""
	any_files = False
	per_spec = []             # (key, probe, pad, lpf_lim, inst, mode, fs, windows, window_files, check)

	for key, probe, pad, lpf_lim, windows in specs:
		inst, mode, fs = field_burst('B' if key[0] == 'B' else 'E')   # E never follows B's selection

		check = cdf_check(key, probe, pickle_root=pickle_root, corrupted_log=corrupted_log)
		window_files = []
		for st, ts, stream in windows:
			files = find_cdf_files(data_dir, inst, mode, st, ts, probe=probe, download=download,
			                       check=check)
			window_files.append((st, ts, stream, files))
			any_files = any_files or bool(files)

		per_spec.append((key, probe, pad, lpf_lim, inst, mode, fs, windows, window_files, check))

	if not any_files:
		return {}, {}, []

	fft_jobs = {}             # fft_pkl -> job tuple, dict so re-adding the same key is a no-op dedup
	fft_owner = {}            # fft_pkl -> (key, probe), to regroup compute_ffts' flat results afterward
	manifest_specs = []       # (key, inst, mode, probe, pad, fs, window_files)

	for key, probe, pad, lpf_lim, inst, mode, fs, windows, window_files, check in per_spec:
		# A later window's discovery can reject a file an earlier window already
		# listed (two windows sharing an anchor), so the lists handed to the
		# manifest step are cut back to what is still on disk and has passed.
		window_files = [(st, ts, stream, check(files)) for st, ts, stream, files in window_files]
		n_expected = int(fs * (pad * 2))
		Df = 1 / (pad * 2)

		# Verified discovery per window -- matches gen_fft's own
		# find_local_cdfs call (NOT the raw find_cdf_files(download=False)
		# union above). find_cdf_files with download=False has no anchor
		# verification and can include a wildly-distant stale file (a real,
		# accepted quirk -- see _pick_start_anchor); find_local_cdfs checks
		# the naive anchor's good recorded bounds and falls forward past
		# it. Skipping this step was a real bug: it let a stale file that
		# only got checked above (harmless, matches old behavior) go
		# all the way through to actual FFT computation (not harmless -- a
		# real regression, caught before this shipped).
		data_by_time = {}
		stream_by_time = {}   # file_time -> True if ANY window needing this file wants it streamed back
		for st, ts, stream, files in window_files:
			for ftime, path in find_local_cdfs(data_dir, inst, mode, st, ts, probe=probe, pickle_root=pickle_root,
			                                   check=check):
				data_by_time[ftime] = path
				stream_by_time[ftime] = stream_by_time.get(ftime, False) or stream
		data_files = sorted(data_by_time.items())
		if not data_files:
			continue

		# The last file's successor comes from disk, not from this request, so
		# the FFT pickle it produces does not depend on where the request ended.
		for (file_time, file_path, prev_file_time, prev_path,
		     next_file_time, next_path) in pair_with_next_cdf(
				data_files, data_dir, inst, mode, probe, check=check):
			fft_pkl = _fft_pickle_path(pickle_root, inst, mode, pad, file_time, probe=probe, step=step)
			file_args = (file_path, next_path, file_time, next_file_time, fft_pkl,
			             prev_path, prev_file_time)
			fft_jobs.setdefault(fft_pkl, (file_args, key, inst, mode, fs, pad, n_expected, Df, step, stream_by_time[file_time]))
			fft_owner.setdefault(fft_pkl, (key, probe))

		manifest_specs.append((key, inst, mode, probe, pad, fs, window_files))

	return fft_jobs, fft_owner, manifest_specs


def stream_good_chunk_ffts(data_dir, pickle_root, key, probes, pad, start, end,
                           lpf_lim=1000, step=0.05, merge_tol=None,
                           download=False, max_workers=MAX_POOL_WORKERS,
                           trust_good_pickles=False, segments_sink=None, recorded_sink=None):
	"""Discover every CDF file covering [start, end] across `probes`, then
	STREAM back one group per qf-GOOD DATA CHUNK -- a contiguous quality-good
	run as defined by the CDF's own quality flag (SCM qf == 'GGG' for B, EDP
	quality >= EDP_MIN_QUALITY for E), NOT a burst window of our own invention. Built for the
	per-chunk image mode (one PNG per chunk, probes as rows) -- see
	plot_good_chunks in short_interval.py.

	Yields (chunk_start, chunk_end, {probe: [fft_pickle_path, ...]}, n_files) per
	good chunk, in strictly ascending chunk_start order:
	  chunk_start -- the EARLIEST qf-good start among the probes observing this
	                 chunk. There is NO reference probe: chunk_start is the only
	                 anchor, and the plotter makes every probe row's x-axis
	                 relative to it, so a probe whose good data starts later
	                 simply begins further right.
	  chunk_end   -- the latest qf-good end among those probes.
	  {probe: paths} -- each probe with good data in this chunk maps to its
	                 FFT pickle paths, in file-time order. PATHS, not frames:
	                 the consumer folds them into its display grid one file at
	                 a time (see plot_good_chunks), which is what keeps memory
	                 flat in burst length. A probe with no good data here isn't
	                 a key (the plotter draws a 'No data' row).
	  n_files     -- how many distinct source files (counted per probe, so the
	                 same burst seen by 4 probes counts 4) fed this chunk. The
	                 driver records it in the run's interval manifest, which
	                 reports coverage as start/end/n_files.

	Identifying "the same burst" across probes whose files start at different
	times is done by TIME-SPAN OVERLAP, not by matching start stamps (see
	_cluster_good_segments) -- overlapping good-data spans are the same burst
	regardless of start jitter, and `merge_tol` (default one FFT window, 2*pad)
	bridges a small edge gap between two probes' good runs of the same burst.

	REUSES the existing pipeline rather than re-scanning or re-deriving:
	  * find_cdf_files -> check_cdfs (one pass over every probe's files) ->
	    find_local_cdfs for discovery, all inside _plan_fft_jobs.
	  * _scm_good_segments_batch / _edp_good_segments_batch (the SAME qf-good
	    segment finders prewarm's manifest step uses) for chunk boundaries --
	    "the qf flag, already found somewhere". The CDF filename's 14-digit
	    stamp is the job's file_time (see find_local_cdfs), so a segment joins
	    its streamed FFT by file_time.
	  * stream_ffts for the compute/plot overlap.

	download=False by default: the driver (plot_biggest_spikes.py) runs its own
	dedicated download worker up front, so discovery here is a local-cache scan.

	trust_good_pickles: where the qf-good boundaries come from.
	  False (default) -- read them from the CDFs' own quality variables via
	    seg_batch, exactly as before. Authoritative, and the only choice that
	    re-derives the boundaries from scratch on every run.
	  True -- read them from each CDF's good-run sidecar instead
	    (_segments_via_sidecar): the same quality test, computed from the time
	    and quality variables once and remembered, and recomputed whenever the
	    CDF is newer than its sidecar. The name is from when the boundaries came
	    off the data pickles. It skips a full Epoch + quality-array read per
	    file per probe on every later run, which is the dominant cost of a wide
	    chunks request.
	    Boundaries are resolved per (field, probe, file) from that file's own
	    contents, so B and E are found INDEPENDENTLY -- they only ever end up
	    in the same image when their good spans genuinely overlap in time, via
	    the same _cluster_good_segments overlap test the CDF path uses. Nothing
	    is inherited from one field to the other.
	"""
	seg_batch = _scm_good_segments_batch if key[0] == 'B' else _edp_good_segments_batch
	if merge_tol is None:
		# One FFT window bridges two probes' edges of the same burst (see the
		# docstring). The sidecar now holds every good run, so a dropout inside a
		# burst is a gap here too; the event threshold keeps it inside the chunk,
		# as a manifest keeps it inside the event (Graham, 2026-09-29).
		merge_tol = max(timedelta(seconds=2 * pad), segment_gap_tol(SAMPLE_RATE))

	# Discovery + CDF checks + per-file FFT jobs: REUSED from prewarm's
	# pipeline via _plan_fft_jobs (one window per probe, stream=True so the
	# DataFrames come back to draw), not re-implemented here.
	# stream=False: workers compute and WRITE each FFT pickle but return None
	# instead of the DataFrame (see _compute_fft_job). This path used to ask for
	# the frames back (stream=True) and concatenate them per probe, which meant
	# the parent held every file's rows for a cluster until its last file
	# arrived, plus a full second copy during pd.concat. A single long
	# quality-good burst reached ~50M rows for ONE probe -- the parent died
	# allocating the concatenated array, and shipping frames that size over IPC
	# was its own cost. The pickles are on disk either way, so the consumer
	# reads them back one at a time instead (see plot_good_chunks).
	specs = [(key, probe, pad, lpf_lim, [(start, end, False)]) for probe in probes]
	fft_jobs, fft_owner, manifest_specs = _plan_fft_jobs(
		data_dir, pickle_root, specs, step=step, max_workers=max_workers, download=download)
	if not fft_jobs:
		return

	# Index the verified jobs by (file_time, probe): a job tuple's file_args[2]
	# is the file_time, fft_owner gives the probe. A stale anchor find_cdf_files
	# surfaced never reached fft_jobs (find_local_cdfs dropped it), so this is
	# the authoritative file set the qf segments get filtered against.
	job_by_ft_probe = {}
	for fft_pkl, job in fft_jobs.items():
		job_by_ft_probe[(job[0][2], fft_owner[fft_pkl][1])] = job

	# qf-good chunk boundaries from the qf flag itself (reusing the batch
	# finders), joined to the jobs by file_time (the CDF's filename stamp).
	# window_files -- the discovered cdf lists -- come straight from _plan_fft_jobs.
	# trust_good_pickles skips this entirely -- the batch is the CDF-opening
	# step, so building its job set at all would be wasted work.
	if trust_good_pickles:
		seg_by_cdf = {}
	else:
		seg_jobs = {
			(c, probe)
			for _key, _inst, _mode, probe, _pad, _fs, window_files in manifest_specs
			for _st, _ts, _stream, files in window_files
			for c in files
		}
		seg_by_cdf = seg_batch(seg_jobs) if seg_jobs else {}
	all_segs = []
	recorded_seen = set()
	for _key, _inst, _mode, probe, _pad, fs, window_files in manifest_specs:
		for _st, _ts, _stream, files in window_files:
			for c in files:
				ft = _parse_cdf_time(c)
				if ft is None or (ft, probe) not in job_by_ft_probe:
					continue
				if trust_good_pickles:
					# job[0] is file_args; file_args[0] is this file's CDF, as
					# verified by discovery (see _plan_fft_jobs). key is this
					# call's own field, so B and E runs never mix.
					# fs comes from the same manifest spec the FFT jobs were
					# built from, so the gap threshold always matches the rate
					# this file is actually being computed at.
					segs = _segments_via_sidecar(job_by_ft_probe[(ft, probe)][0][0], key, fs, probe, pickle_root)
				else:
					segs = seg_by_cdf.get(c, [])
				for cs, ce in segs:
					all_segs.append((cs, ce, ft, probe))
				# recorded_sink: the same files' recorded runs, whatever their
				# quality, for the chunk manifest's recorded table (Graham,
				# 2026-09-30). Once per file, from the CDF discovery verified.
				if recorded_sink is not None and (ft, probe) not in recorded_seen:
					recorded_seen.add((ft, probe))
					for rs, re_ in _segments_via_sidecar(job_by_ft_probe[(ft, probe)][0][0], key, fs, probe,
					                                     pickle_root, recorded=True):
						recorded_sink.setdefault(probe, []).append((rs, re_, ft))
	# A good run of fewer than MIN_GOOD_RUN_SAMPLES samples is no part of any chunk
	# (Graham, 2026-10-09), measured per probe, across file boundaries.
	try:
		from spectro.data_intervals import drop_short_runs
	except ImportError:
		from src.spectro.data_intervals import drop_short_runs
	all_segs = [seg for probe in sorted({s[3] for s in all_segs})
	            for seg in drop_short_runs(sorted((s for s in all_segs if s[3] == probe), key=lambda s: s[0]),
	                                       **good_run_rule(field_burst(key)[2]))]
	# Per-probe view for the caller's manifest. The clusters built below
	# deliberately span probes (that is what a drawn chunk IS -- one PNG with the
	# probes stacked as rows), so their bounds are the UNION across probes and
	# cannot answer "when did probe 3 start and stop". These per-probe segments can,
	# and they are the same numbers the clustering consumes, so a manifest built
	# from them describes exactly the data that was streamed rather than a separate
	# re-derivation of it.
	if segments_sink is not None:
		for cs, ce, ft, probe in all_segs:
			segments_sink.setdefault(probe, []).append((cs, ce, ft))
	if not all_segs:
		return

	# --- Identify bursts (global overlap clustering) -----------------------
	clusters = _cluster_good_segments(all_segs, merge_tol)

	# Stream ONLY the files the clusters draw from, file_time-ordered so bursts
	# close promptly.
	needed = sorted({fp for _x0, _x1, needs in clusters for fp in needs})
	jobs = [job_by_ft_probe[fp] for fp in needed]
	job_meta = needed

	# Reverse index (file_time, probe) -> clusters still waiting on it, plus a
	# per-cluster remaining-need counter, so a cluster emits the instant its
	# last file's FFT arrives (a file feeding two clusters decrements both).
	waiting = {}
	remaining = [len(needs) for _x0, _x1, needs in clusters]
	for ci, (_x0, _x1, needs) in enumerate(clusters):
		for fp in needs:
			waiting.setdefault(fp, set()).add(ci)

	# (file_time, probe) -> the FFT pickle that file's rows live in. file_args[4]
	# is that path (see _plan_fft_jobs); by the time a cluster completes below,
	# every one of its files has been written by a worker.
	fft_path_by_ft_probe = {fp: job[0][4] for fp, job in job_by_ft_probe.items()}

	# --- Stream compute; emit each burst as it completes -------------------
	# Nothing is retained here: the loop tracks only completion counts, so the
	# parent's memory is independent of how long a burst runs.
	for (ft, probe), _result in zip(job_meta, stream_ffts(jobs, max_workers=max_workers)):
		# sorted, not list: clusters are built in ascending start order (from
		# time-sorted segments), and one file can complete two of them at once
		# -- iterating the waiting SET directly would then emit those two in
		# arbitrary order. Sorting the indices makes emission strictly
		# ascending, which the per-day CSV writer downstream relies on to know
		# a calendar day is finished once a later day appears.
		for ci in sorted(waiting.get((ft, probe), ())):
			remaining[ci] -= 1
			if remaining[ci] > 0:
				continue
			x0, x1, needs = clusters[ci]
			probe_paths = {}
			# sorted so each probe's paths come out in file-time order, which
			# is the order the consumer folds them into its grid.
			for fp in sorted(needs):
				probe_paths.setdefault(fp[1], []).append(fft_path_by_ft_probe[fp])
			yield x0, x1, probe_paths, len(needs)


def prewarm(data_dir, pickle_root, specs, corrupted_log=None, manifest_dir=None,
           write_manifest=True, resolve_bounds=True, step=0.05, max_workers=MAX_POOL_WORKERS,
           manifest_batch_dir=None, trust_good_pickles=False, manifest_duration_copy=False,
           event_gap_s=None):
	"""Warm the FFT-pickle cache for MULTIPLE (key, probe) combinations, each
	spanning MULTIPLE possibly-overlapping [st, ts] windows, via one shared
	FFT job pool (compute_ffts) -- so every unique underlying file is computed
	exactly once no matter how many overlapping windows or (field, probe)
	combos need it. (A second pool used to convert CDFs to data pickles first;
	the FFT workers read the CDFs directly now -- see _plan_fft_jobs.)

	This is the fix for the hazard found parallelizing plot_biggest_spikes.py
	per-request: REQUESTS' date ranges actually overlap (the 'calculate'
	request spans across both 'spike' requests' days), so per-request or
	per-(field,probe) parallel workers touching the SAME file concurrently
	would race on ensure_data_pickle/_compute_and_save_fft's unlocked
	check-then-act cache logic. Deduplicating at the FILE-JOB level instead
	of the request/field/probe level makes that race structurally
	impossible: each file is exactly one job, submitted once, into one pool.

	specs: iterable of (key, probe, pad, lpf_lim, windows), windows itself an
	       iterable of (st, ts, stream) triples -- stream=True means this
	       window's underlying files' computed DataFrames should come back
	       in the returned dict; stream=False means cache-warm only (the
	       FFT pickle still gets written, but the computed array is
	       discarded inside its worker instead of crossing back over IPC
	       and piling up in this process). A file needed by BOTH a
	       stream=True and a stream=False window within the same spec is
	       streamed (OR, not AND) -- see the stream_by_time merge below.

	Returns {(key, probe): dataframe} -- ONE concatenated, time-sorted
	DataFrame per (key, probe), covering every STREAMED window in that spec,
	deduplicated by fft_pkl (an overlap between two windows contributes its
	file's rows exactly once). A (key, probe) whose windows were ALL
	stream=False (pure cache-warming, e.g. a wide 'calculate'-only request
	with no plot coming out of it) maps to an empty DataFrame -- its files
	are cached on disk exactly as before, just not returned here. This is
	the ACTUAL computed data for streamed windows, not a file list or a
	path: compute_ffts already has to ship each streamed worker's result
	back over IPC to leave its process, so returning that DataFrame here
	costs nothing extra -- it's the SAME serialization that already has to
	happen, just not thrown away afterward. A caller plotting a window
	nested inside one of these (st, ts) windows (e.g. plot_biggest_spikes.py
	plotting a spike-search result found within one of its own request
	windows) should time-filter this DataFrame directly (cheap, in-memory,
	exact -- real computed timestamps, not a filename guess) instead of
	touching the filesystem at all.

	trust_good_pickles: same meaning as in stream_good_chunk_ffts -- read the
	manifest's qf-good spans from each CDF's good-run sidecar
	(_segments_via_sidecar) rather than the batch readers of every CDF's quality
	variable. Only reaches anything when write_manifest is on (the manifest step
	is the sole quality reader on this path; the FFT computation reads the CDF
	data, which _cdf_to_df has already quality-filtered).

	manifest_dir: root of the cross-run manifest CACHE (data_paths.
	INTERVAL_MANIFEST). Entries are nested by probe/inst/mode and their
	filenames carry _v<MANIFEST_VERSION>, which is what makes a stale one
	detectable. A rerun whose cached entry is newer than every source CDF
	recomputes nothing.

	manifest_batch_dir: this run's <batch>/intervals. Each cached manifest is
	copied here as <st>-<ts>/<field>_mms<probe>.txt -- one folder per request,
	at most 4 probes x 2 fields = 8 files inside it, and no version in the name
	(the batch folder's run stamp already identifies it). None -> cache only.

	event_gap_s: gap threshold (seconds) below which two intervals are the same
	physical event and merge into one manifest row. None keeps the old
	sample-adjacency behaviour (2 sample periods).

	manifest_duration_copy: also emit the legacy duration-only view
	(<field>_mms<probe>.duration.txt) beside each batch copy. Off by default --
	write_burst_manifest's format now carries start, end AND duration in one
	file, so the second copy only doubled the file count.
	"""
	# Discovery + CDF->pickle warming + per-file FFT job construction, shared
	# with stream_good_chunk_ffts via _plan_fft_jobs (the streamed per-qf-chunk
	# path submits the SAME job tuples to stream_ffts instead of compute_ffts).
	fft_jobs, fft_owner, manifest_specs = _plan_fft_jobs(
		data_dir, pickle_root, specs, step=step, max_workers=max_workers,
		corrupted_log=corrupted_log, download=False)

	empty_fft = pd.DataFrame(columns=['time', 'frequency', 'power'])

	combined = {}   # (key, probe) -> concatenated, time-sorted DataFrame
	if fft_jobs:
		results = compute_ffts(list(fft_jobs.values()), max_workers=max_workers)
		rows_by_owner = {}   # (key, probe) -> [(file_time, df), ...]
		for fft_pkl, (file_time, df) in zip(fft_jobs.keys(), results):
			rows_by_owner.setdefault(fft_owner[fft_pkl], []).append((file_time, df))
		for owner, rows in rows_by_owner.items():
			rows.sort(key=lambda r: r[0])
			frames = [df for _, df in rows if df is not None and not df.empty]
			combined[owner] = pd.concat(frames, ignore_index=True) if frames else empty_fft

	if write_manifest:
		try:
			from spectro.data_intervals import (
				write_request_manifest, write_burst_manifest_readable, read_burst_manifest,
			)
		except ImportError:
			from src.spectro.data_intervals import (
				write_request_manifest, write_burst_manifest_readable, read_burst_manifest,
			)

		# Cached across the WHOLE prewarm call, not just one window: a file
		# in the overlap between two requests (e.g. an evening-of-Feb-23 CDF
		# covered by both the day-1 request and the week-long request) would
		# otherwise get its bounds/segments re-read via a real CDF open once
		# per overlapping window/spec -- redundant, and each open is a slow
		# round-trip. A path uniquely belongs to one probe/inst/mode already
		# (baked into its own directory), so caching by path alone is safe.
		bounds_cache       = {}   # F (FGM) only:  path -> (start, end)
		segments_cache     = {}   # B/SCM only:    path -> [(start, end), ...]
		edp_segments_cache = {}   # E/EDP only:    path -> [(start, end), ...]

		def _cached_bounds(p, _key, _probe):
			if p not in bounds_cache:
				try:
					bounds_cache[p] = cdf_time_bounds(p, key=_key, probe=_probe)
				except Exception:
					bounds_cache[p] = (None, None)
			return bounds_cache[p]

		# trust_good_pickles: resolve a CDF path to the verified CDF discovery
		# kept for it, so the segment finders below read good spans from its
		# sidecar (_segments_via_sidecar) instead of the CDF-quality pools.
		# _plan_fft_jobs carries that path for every file that survived
		# discovery; a stale anchor that did not gets no segments.
		#
		# Keyed by (file_time, probe, FIELD) -- the field letter is load-bearing,
		# not decoration. SCM and EDP burst files covering the same burst share
		# the SAME 14-digit filename stamp (mms1_scm_..._20210223032923 and
		# mms1_edp_..._20210223032923 both exist), so keying on (file_time,
		# probe) alone made one entry for both instruments. A prewarm call
		# carrying B and E specs together -- which is exactly what phase 1 of
		# plot_biggest_spikes does -- then had E overwrite B, and every B
		# manifest that call produced reported EDP's good spans under SCM's
		# name: byte-identical Bz_*/Ez_* files, B's 13 real bursts replaced by
		# E's 4865 quality-shredded fragments. Per-field prewarm calls and
		# stream_good_chunk_ffts never tripped it (they only ever carry one
		# field), which is why the narrow spike-window manifests looked fine
		# while the wide ones did not.
		cdf_by_ft_probe = {}
		if trust_good_pickles:
			cdf_by_ft_probe = {
				(job[0][2], fft_owner[fft_pkl][1], fft_owner[fft_pkl][0][0]): job[0][0]
				for fft_pkl, job in fft_jobs.items()
			}

		def _pickle_segments(p, _probe, _fs, _field):
			ft = _parse_cdf_time(p)
			cdf = cdf_by_ft_probe.get((ft, _probe, _field)) if ft is not None else None
			return _segments_via_sidecar(cdf, _field, _fs, _probe, pickle_root) if cdf is not None else []

		def _cached_segments(p, _probe, _fs):
			if p not in segments_cache:
				try:
					segments_cache[p] = (
						_pickle_segments(p, _probe, _fs, 'B') if trust_good_pickles
						else _scm_good_segments(p, _probe)
					)
				except Exception:
					segments_cache[p] = []
			return segments_cache[p]

		def _cached_edp_segments(p, _probe, _fs):
			if p not in edp_segments_cache:
				try:
					edp_segments_cache[p] = (
						_pickle_segments(p, _probe, _fs, 'E') if trust_good_pickles
						else _edp_good_segments(p, _probe)
					)
				except Exception:
					edp_segments_cache[p] = []
			return edp_segments_cache[p]

		recorded_cache = {}

		def _cached_recorded(p, _probe, _fs, _field):
			"""Every recorded run of p, whatever its quality, for the manifest's
			recorded table (Graham, 2026-09-30) -- read from the same CDF the used
			table's runs come from, through its own sidecar."""
			if (p, _field) not in recorded_cache:
				if trust_good_pickles:
					ft = _parse_cdf_time(p)
					cdf = cdf_by_ft_probe.get((ft, _probe, _field)) if ft is not None else None
				else:
					cdf = p
				try:
					recorded_cache[(p, _field)] = (
						_segments_via_sidecar(cdf, _field, _fs, _probe, pickle_root, recorded=True)
						if cdf is not None else [])
				except Exception:
					recorded_cache[(p, _field)] = []
			return recorded_cache[(p, _field)]

		# Prefetch every B/SCM and E/EDP file's segments in one parallel pass
		# each instead of opening them one at a time inside the per-window
		# loop below -- each is a read-only, independent CDF open (see
		# _scm_good_segments_batch / _edp_good_segments_batch), so this is
		# strictly a speedup, not a behavior change: the _cached_* functions
		# above still fill in anything a pass missed.
		# Both prefetch pools are pure CDF-opening work, so trust_good_pickles
		# skips them outright -- the _cached_* resolvers above then fill each
		# path in lazily from its data pickle, which is the whole point of the
		# flag. (They were only ever a speedup over the same lazy path.)
		if not trust_good_pickles:
			segment_jobs = {
				(p, probe)
				for key, inst, mode, probe, pad, fs, window_files in manifest_specs
				if key[0] == 'B'
				for st, ts, stream, files in window_files
				for p in files
			}
			if segment_jobs:
				segments_cache.update(_scm_good_segments_batch(segment_jobs))

			edp_segment_jobs = {
				(p, probe)
				for key, inst, mode, probe, pad, fs, window_files in manifest_specs
				if key[0] == 'E'
				for st, ts, stream, files in window_files
				for p in files
			}
			if edp_segment_jobs:
				edp_segments_cache.update(_edp_good_segments_batch(edp_segment_jobs))

		for key, inst, mode, probe, pad, fs, window_files in manifest_specs:
			if not resolve_bounds:
				_resolver, _seg_resolver = None, None
			elif key[0] == 'B':
				# Quality-bad spans (whole file, or just a sub-span, e.g. the
				# usual leading convolution-edge artifact) never contribute
				# an FFT window anyway -- _cdf_to_df drops those rows,
				# _compute_file_ffts short-circuits on the resulting empty
				# data (skipped for free). Reporting only the quality-good
				# runs here means the manifest shows a real gap wherever
				# data existed but wasn't used, not just gaps between files.
				_resolver, _seg_resolver = None, (lambda p, _probe=probe, _fs=fs: _cached_segments(p, _probe, _fs))
			elif key[0] == 'E':
				# EDP (E) now gets segments too, via mms{probe}_edp_quality_
				# brst_l2 (quality >= EDP_MIN_QUALITY -- see _edp_good_segments,
				# characterized against real files across all 4 probes: the
				# variable's own CATDESC is authoritative, "3 good / 2 ok,
				# use with caution / 1 bad, use with caution / 0 really bad
				# or no data"). _cdf_to_df now applies this SAME threshold
				# filter on the compute side too, so the manifest and the FFT
				# input finally agree for E the way they always did for B --
				# a span this reports as a gap is a span no E FFT window was
				# ever computed over. (Before that filter existed, this was a
				# fact about the manifest only, and _compute_file_ffts ran
				# over quality-bad EDP data regardless.)
				_resolver, _seg_resolver = None, (lambda p, _probe=probe, _fs=fs: _cached_edp_segments(p, _probe, _fs))
			else:
				# F (FGM) falls here, unreached today since prewarm's only
				# caller never requests it -- plain whole-file bounds.
				_resolver, _seg_resolver = (lambda p, _key=key, _probe=probe: _cached_bounds(p, _key, _probe)), None
			# The recorded table: every recorded run, whatever its quality, from the
			# same CDFs (Graham, 2026-09-30). Whole-file bounds rows already are that.
			_rec_resolver = ((lambda p, _probe=probe, _fs=fs, _field=key[0]: _cached_recorded(p, _probe, _fs, _field))
			                 if _seg_resolver is not None else None)
			# Two destinations, two different jobs -- and only ONE of them
			# carries a version.
			#
			# 1. The CACHE, under manifest_dir:
			#      .manifest/mms/<probe>/<inst>/<mode>/intervals/
			#         <key>_<st>_<ts>_pad<pad>_v<MANIFEST_VERSION>.txt
			#    Path is stable for a given request, so a rerun that finds it
			#    newer than every source CDF skips recomputation entirely (see
			#    write_request_manifest). Resolving segments here means re-reading
			#    every data pickle in the parent process -- work the FFT pass
			#    itself does not repeat -- so the skip is worth real minutes on a
			#    wide request. The VERSION LIVES HERE AND ONLY HERE: it is the
			#    staleness mechanism. A format or logic change bumps it, which
			#    changes the filename, which makes the old cache entry invisible
			#    rather than silently reused.
			#
			# 2. The BATCH copy, under manifest_batch_dir:
			#      <batch>/intervals/<st>-<ts>/<field>_mms<probe>.txt
			#    One folder per request, one file per (probe, field) inside it --
			#    4 x 2 = 8 at most, because the probes are the only thing whose
			#    data start/stop times differ within a request. NO version in the
			#    name: a batch is a snapshot of one run, and there is nothing to
			#    tell it apart from -- the run stamp in the folder above already
			#    says which run produced it. A version here would be noise that
			#    reads like a choice the reader has to make.
			for st, ts, _stream, files in window_files:
				# event_gap_s None -> the old sample-adjacency tol (2 sample
				# periods), which only merges files that literally tile. Set it
				# and rows become physical events instead: gaps narrower than it
				# are dropouts inside one burst, wider ones are real boundaries.
				mpath = write_request_manifest(
					manifest_dir or pickle_root, files, key, inst, mode, probe, pad, st, ts,
					fallback_end=ts, bounds_resolver=_resolver, segments_resolver=_seg_resolver,
					tol=(timedelta(seconds=event_gap_s) if event_gap_s is not None
					     else segment_gap_tol(fs)),
					corrupted=cdf_validation.corrupted_names(key, pickle_root=pickle_root),
					recorded_resolver=_rec_resolver, **good_run_rule(fs),
				)
				if manifest_batch_dir:
					req_dir = os.path.join(manifest_batch_dir,
					                       f'{st:%Y%m%dT%H%M%S}-{ts:%Y%m%dT%H%M%S}')
					os.makedirs(req_dir, exist_ok=True)
					shutil.copy2(mpath, os.path.join(req_dir, f'{key[0]}_mms{probe}.txt'))
					# Legacy duration-only view (start + duration, no end column).
					# OFF by default: write_burst_manifest's own format now carries
					# start, end AND duration in the one file, so this only ever
					# doubled the count. Kept behind the flag for anything that
					# still parses the old shape. Derived from the cached manifest
					# so it always reflects exactly what was recorded.
					if manifest_duration_copy:
						write_burst_manifest_readable(
							os.path.join(req_dir, f'{key[0]}_mms{probe}.duration.txt'),
							read_burst_manifest(mpath),
						)

	return combined


def collect_fft_data(data_dir, pickle_root, key, pad, st, ts, lpf_lim=1000, corrupted_log=None, step=0.05, probe: int = 1, manifest_dir=None, write_manifest=True, resolve_bounds=True, download=True):
	"""Synchronous, single-process sibling of gen_fft for SHORT intervals --
	the harness ShortIntervalPlotter (short_interval.py) calls instead of
	doing its own CDF discovery / pickling / FFT computation. Owning this
	here (not in the plotter) is the point: a plotting class has no business
	finding CDFs, writing pickles, or running FFT math -- it should take
	already-computed FFT data and draw it.

	No ProcessPoolExecutor (that's gen_fft's job for long intervals with
	many files) -- for a handful of files, pool overhead dwarfs any win, and
	staying single-process means get_or_compute_fft can hand back a freshly
	computed DataFrame directly, no write-then-immediately-reread round trip.

	Returns one concatenated DataFrame (columns time/frequency/power) over
	[st, ts], already time-filtered. Empty (0-row, same columns) if no data.
	"""
	inst, mode, fs = field_burst('B' if key[0] == 'B' else 'E')   # E never follows B's selection

	n_expected = int(fs * (pad * 2))
	Df         = 1 / (pad * 2)

	empty = pd.DataFrame(columns=['time', 'frequency', 'power'])

	# Validation comes first and travels with discovery, so every file below --
	# the manifest's, the FFT's and their seam neighbours -- has already passed
	# (see cdf_check). The manifest used to open the discovered files BEFORE they
	# were checked, and one bad file killed the request with it still on disk.
	check = cdf_check(key, probe, pickle_root=pickle_root, corrupted_log=corrupted_log)
	cdf_files = find_cdf_files(data_dir, inst, mode, st, ts, probe=probe, download=download,
	                           check=check)
	if not cdf_files:
		return empty

	if write_manifest:
		try:
			from spectro.data_intervals import write_request_manifest
		except ImportError:
			from src.spectro.data_intervals import write_request_manifest

		_rec_resolver = None
		if not resolve_bounds:
			_resolver, _seg_resolver = None, None
		elif key[0] in ('B', 'E'):
			# Every good run from each CDF's sidecar -- all its good data -- which
			# write_request_manifest clips to [st, ts] and groups at the event
			# threshold below: the manifest measures events, the sidecar covers the
			# good data (Graham, 2026-09-29). Same rows prewarm writes. The recorded
			# table gets every recorded run the same way (Graham, 2026-09-30).
			_resolver, _seg_resolver = None, (lambda p: _segments_via_sidecar(p, key, fs, probe, pickle_root))
			_rec_resolver = lambda p: _segments_via_sidecar(p, key, fs, probe, pickle_root, recorded=True)
		else:
			_resolver, _seg_resolver = (lambda p: cdf_time_bounds(p, key=key, probe=probe)), None
		write_request_manifest(
			manifest_dir or pickle_root, cdf_files, key, inst, mode, probe, pad, st, ts,
			fallback_end=ts, bounds_resolver=_resolver, segments_resolver=_seg_resolver,
			tol=segment_gap_tol(fs),
			# Names given up on after GIVE_UP_AFTER failed refetches: their data is
			# absent from the rows, and nothing else on disk still records that.
			corrupted=cdf_validation.corrupted_names(key, pickle_root=pickle_root),
			recorded_resolver=_rec_resolver, **good_run_rule(fs),
		)

	data_files = find_local_cdfs(data_dir, inst, mode, st, ts, probe=probe, pickle_root=pickle_root,
	                             check=check)
	if not data_files:
		return empty

	frames = []
	# Neighbours from disk at both ends -- see pair_with_next_cdf.
	for (file_time, curr_path, prev_file_time, prev_path,
	     next_file_time, next_path) in pair_with_next_cdf(
			data_files, data_dir, inst, mode, probe, check=check):
		fft_pkl = _fft_pickle_path(pickle_root, inst, mode, pad, file_time, probe=probe, step=step)
		fdf = get_or_compute_fft(
			(curr_path, next_path, file_time, next_file_time, fft_pkl,
			 prev_path, prev_file_time),
			key=key, inst=inst, mode=mode, fs=fs, pad=pad,
			n_expected=n_expected, Df=Df, step=step,
		)
		# The ceiling is applied HERE now, not baked into the pickle at compute
		# time -- the pickle spans DC to Nyquist. This function's contract is
		# unchanged: callers still get rows no higher than lpf_lim.
		fdf = fdf[
			(fdf['time'] >= np.datetime64(st, 'ns')) &
			(fdf['time'] <= np.datetime64(ts, 'ns')) &
			(fdf['frequency'] <= lpf_lim)
		]
		if not fdf.empty:
			frames.append(fdf)

	if not frames:
		return empty
	return pd.concat(frames, ignore_index=True)


def gen_fft(data_dir, pickle_root, key, pad, st, ts, lpf_lim=1000, corrupted_log=None, step=0.05, probe: int = 1, manifest_dir=None, write_manifest=True, resolve_bounds=True, download=True):
	"""
	Compute a sliding-window PSD spectrogram over [st, ts].

	Parameters
	----------
	data_dir    : str      Root data directory, e.g. './pydata'
	pickle_root : str      Root cache directory, e.g. './.serialized'
	key         : str      Field prefix: 'B' for magnetic (SCM/SCB), 'E' for electric (EDP/DCE)
	pad         : float    Half-window in seconds; full window = 2*pad s, Df = 1/(2*pad) Hz
	st          : datetime Start time (inclusive)
	ts          : datetime End time (inclusive)
	lpf_lim     : float    Upper frequency cutoff (Hz)
	download    : bool     Forwarded to find_cdf_files. Set False when a caller
	                       has its own dedicated download pipeline and needs a
	                       hard guarantee that gen_fft will never itself trigger
	                       a download (see plot_biggest_spikes.py).

	Returns
	-------
	Generator of pd.DataFrames, one per raw data file, with columns:
	    time      (datetime64[ns])  absolute timestamp
	    frequency (int16)           Hz, snapped to nearest Df multiple
	    power     (float64)         nT²/Hz (or mV²/m²/Hz for E-field)

	Caching
	-------
	Raw data: not cached -- each CDF is read directly (read_burst_data). Only a
	    file's quality-good run boundaries are kept (_segments_via_sidecar).
	    Path: {pickle_root}/mms/{probe}/{inst}/{mode}/segments/gap{N}ns/{YYYY}/{MM}/{DD}/{HHMMSS}.npy

	FFT results: stored per raw data file, keyed by pad value.
	    Path: {pickle_root}/mms/{probe}/{inst}/{mode}/fft/pad{pad:.3f}/{YYYY}/{MM}/{DD}/{HHMMSS}.pkl
	    On-disk time column: datetime64[ns] (absolute, 8 bytes).
	    Filtered to [st, ts] on load; downstream accumulation code unchanged.
	"""
	inst, mode, fs = field_burst('B' if key[0] == 'B' else 'E')   # E never follows B's selection

	n_expected = int(fs * (pad * 2))
	Df         = 1 / (pad * 2)

	# Validation first, carried by discovery -- see collect_fft_data. The manifest
	# below used to open the discovered files before they were checked, and its
	# _safe_bounds answered a bad one with filename bounds, so the manifest claimed
	# a span check_cdfs then deleted and no FFT ever used.
	check = cdf_check(key, probe, pickle_root=pickle_root, corrupted_log=corrupted_log)
	cdf_files = find_cdf_files(data_dir, inst, mode, st, ts, probe=probe, download=download,
	                           check=check)
	if not cdf_files:
		raise FileNotFoundError(
			f'No CDF files found under {data_dir}/mms{probe}/{inst}/brst/l2/{mode} '
			f'for the range {st} to {ts}.'
		)

	# Request-scoped burst manifest: the physical events this request draws on,
	# written up front (before any FFT work) for later pick/filter. With
	# resolve_bounds (default), B and E rows are every good run from each CDF's
	# sidecar, clipped to [st, ts] and grouped at the event threshold -- the same
	# rows collect_fft_data and prewarm write (Graham, 2026-09-29: the manifest
	# measures events, the sidecar covers all good data). F has no quality flag,
	# so it keeps each file's first/last record.
	if write_manifest:
		try:
			from spectro.data_intervals import write_request_manifest
		except ImportError:
			from src.spectro.data_intervals import write_request_manifest
		def _safe_bounds(p):
			try:
				return cdf_time_bounds(p, key=key, probe=probe)
			except Exception:
				return None, None   # corrupt/unreadable: fall back to filename bounds
		_resolver = _safe_bounds if resolve_bounds else None
		_seg_resolver = ((lambda p: _segments_via_sidecar(p, key, fs, probe, pickle_root))
		                 if resolve_bounds and key[0] in ('B', 'E') else None)
		# The recorded table: every recorded run, whatever its quality (Graham,
		# 2026-09-30). F has no quality flag, so its rows already are that.
		_rec_resolver = ((lambda p: _segments_via_sidecar(p, key, fs, probe, pickle_root, recorded=True))
		                 if _seg_resolver is not None else None)
		write_request_manifest(
			manifest_dir or pickle_root, cdf_files, key, inst, mode, probe, pad, st, ts,
			fallback_end=ts, bounds_resolver=_resolver, segments_resolver=_seg_resolver,
			tol=segment_gap_tol(fs),
			# Names given up on after GIVE_UP_AFTER failed refetches: their data is
			# absent from the rows, and nothing else on disk still records that.
			corrupted=cdf_validation.corrupted_names(key, pickle_root=pickle_root),
			recorded_resolver=_rec_resolver, **good_run_rule(fs),
		)

	data_files = find_local_cdfs(data_dir, inst, mode, st, ts, probe=probe, pickle_root=pickle_root,
	                             check=check)
	if not data_files:
		return

	# Neighbours from disk at both ends -- see pair_with_next_cdf.
	file_args = [
		(file_path, next_path, file_time, next_file_time,
		 _fft_pickle_path(pickle_root, inst, mode, pad, file_time, probe=probe, step=step),
		 prev_path, prev_file_time)
		for file_time, file_path, prev_file_time, prev_path, next_file_time, next_path
		in pair_with_next_cdf(data_files, data_dir, inst, mode, probe, check=check)
	]

	worker = partial(_compute_and_save_fft, key=key, inst=inst, mode=mode, fs=fs,
	                 pad=pad, n_expected=n_expected, Df=Df, step=step)

	with ProcessPoolExecutor(max_workers=MAX_POOL_WORKERS) as pool:
		for fft_pkl in pool.map(worker, file_args):
			if not fft_pkl:
				continue

			file_fdf = pd.read_pickle(fft_pkl)
			# Ceiling applied on read -- the pickle spans DC to Nyquist now.
			file_fdf = file_fdf[
				(file_fdf['time'] >= np.datetime64(st, 'ns')) &
				(file_fdf['time'] <= np.datetime64(ts, 'ns')) &
				(file_fdf['frequency'] <= lpf_lim)
			]

			if file_fdf.empty:
				continue

			# Explicit, not just inherited from pool.map()'s input-order
			# guarantee -- callers concatenating chunks across yields (or a
			# future executor swap that doesn't preserve submission order)
			# must still get results sorted by time, not by whatever order
			# work happened to finish in.
			file_fdf = file_fdf.sort_values('time')

			yield file_fdf


paper = 'mms_data/mms1/scm/brst/l2/schb/2019/08/16/mms1_scm_brst_l2_schb_20190816093103_v2.2.0.cdf'
pkl_path_p = '.serialized/mms/1/scm/schb/20190816093145.pkl'

paperb = 'pydata/mms1/scm/brst/l2/scb/2019/08/16/mms1_scm_brst_l2_scb_20190816093103_v2.2.1.cdf'
pkl_path_b = '../.serialized/mms/1/scm/scb/20190816093145.pkl'

pkl_path_pr = '.serialized/mms/1/scm/fft/scb/20190816093145.pkl'
if __name__ == "__main__":
	# Imported HERE, not at module scope: this block is the only matplotlib
	# user in the file, and it runs only when ffttest is the entry point.
	# Spawned workers import ffttest as a normal module, so __name__ is
	# 'ffttest' and none of this executes -- but a module-scope import would
	# still have cost them ~70 MB of commit each. See the import note at the
	# top of the file.
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	from matplotlib.ticker import FuncFormatter, FixedLocator
	from math import log

#	rcParams['path.simplify'] = True
#	rcParams['path.simplify_threshold'] = 0.2
	#rcParams['savefig.format'] = 'svg'
	mpl.use('Qt5Agg')

	lpf_lim = 1000
	hpf_lim = 10
	num_bins = 40
	bin_factor = (lpf_lim) ** (1 / num_bins)

	sig_key = 'Bz'
	inst = 'scm' if sig_key[0] == 'B' else 'edp'
	units = r'\frac{\text{nT}^2}{\text{Hz}}' if sig_key[0] == 'B' else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'

	if sig_key[0] == 'B':
		cdf_path = f'pydata/mms1/{inst}/brst/l2/scb/2019/08/16/mms1_{inst}_brst_l2_scb_20190816093103_v2.2.1.cdf'
		vmi = 1e-6
		vma = 1e-2
	else:
		cdf_path = f'pydata/mms1/{inst}/brst/l2/dce/2019/08/16/mms1_{inst}_brst_l2_dce_20190816093103_v3.0.1.cdf'
		vmi = 1e-4
		vma = 10

	from data_paths import DATA_DIR as data_dir, PICKLE_ROOT as pickle_root  # per-platform E:/ vs /mnt/e
	save = True
	plot = False

	# all scm caches for graphing all files
	# sfc = glob.glob('mms_data/mms1/scm/brst/l2/schb/2020/09/02/*')
	# pkl_paths = glob.glob('.serialized/scm/schb/*')

	datefmt = '%m/%d/%Y-%H:%M:%S'
	start = datetime.strptime('08/01/2019-00:00:00', datefmt)
	#start = datetime.strptime('08/16/2019-09:31:58.45', datefmt)
	end = datetime.strptime('02/01/2020-00:00:00', datefmt)

	if plot:
		# Create two axes: top for Ez vs time, bottom for FFT/PSD
		fig, axes = plt.subplots(
			1, 1, figsize=(8, 4), sharex=True, constrained_layout=True
		)
		axes = list([axes,])

	bins = list()

	def make_log_bins(freqs, period):
		h = len(freqs) - 1
		df = 1 / period

		b = list()
		while freqs[h] >= df and h > 0:
			b.append(int(freqs[h]))
			h = int(h // bin_factor)

		b.append(int(freqs[0]))

		b.reverse()

		return b

	def channelize(x):
		global bins

		if x == bins[0]:
			return 0
		else:
			for i in range(1, len(bins)):
				if bins[i - 1] < x <= bins[i]:
					return i

		if x == bins[-1]:
			return len(bins)

		print(f'wtf: {x}')
		return None


#		 def display_logbin_legend(ax: plt.Axes):
#	        lg_lbls = [f'Channel {i}: [{bins[i - 1]} - {bins[i]}) Hz' for i in range(channelize(10), len(bins) - 1)]
#	        lg_lbls.append(f'Channel {len(bins) - 1}: [{bins[len(bins) - 2]} - {lpf_lim}] Hz')
#	        proxies = [Patch(color='none') for _ in range(len(lg_lbls))]
#	        ax.legend(proxies, lg_lbls, handlelength=0, handletextpad=0)


	vfunc = np.vectorize(channelize)

	#clear_fft(date=start) # deletes all pkl files for start date, must be adjusted for other

	name = './out/sheet.xlsx'
	if os.path.exists(name):
		os.remove(name)

# 	writer = pd.ExcelWriter(name, engine='xlsxwriter', mode='w')

# # 	book = writer.book

# 	sci_notate = book.add_format()
#	sci_notate.set_num_format(11)

#	print('open excel')

	f = lambda x: int(x)
	vf = np.vectorize(f)
	logbf = lambda x: log(x, bin_factor)
	vlogbf = np.vectorize(logbf)

	if plot:
		display_dt = 0.10  # seconds per display bin — reduce for more detail, increase for speed

		fig.suptitle(
			fr'$|{sig_key[0]}|$ Frequency vs. Time vs. $|{sig_key[0]}|$ Power ${units}$ starting from {start.hour:02d}:{start.minute:02d}:{start.second:02d}')

		methods = False # ~~~~~~~~~~~~~~~~~~~~~~~~~~ TRUE IF PLOTTING SPECTRA OF BOTH, FALSE IF PLOTTING POWER OF BOTH TODO:

	for j, i in enumerate([0.1]): #, 0.25, 0.5]):
		print(i*2)

		if not plot:
			a = datetime.now()
			print(f"Started generating FFTs at @ {a.strftime('%H:%M:%S.%f')[:-3]}")

			for _ in gen_fft(data_dir, pickle_root, sig_key, i, start, end, lpf_lim):
				pass
			b = datetime.now()
			print(f"Ended generating FFTs at @ {b.strftime('%H:%M:%S.%f')[:-3]}")
			print(f"Duration {str(b - a)}")
		else:
			ax = axes[j]

			if start == end:
				def make_fft(start, end, method, i, key):
					# method parameter retained for signature compatibility; gen_fft is always used
					fdf = pd.concat(list(gen_fft(data_dir, pickle_root, key, i, start, end, lpf_lim)), ignore_index=True)

					fdf.loc[fdf['frequency'] == 0, 'frequency'] = 1

					print('load')

			#		fq = fdf['frequency'].unique() # TODO: delete when done debugging
					global bins
					bins = make_log_bins(fdf['frequency'].unique(), i * 2)
					print('bins')
					fdf['channel'] = vfunc(fdf['frequency'])
					print('channelize')

					# Convert absolute datetime64[ns] → float seconds since start
					st_ns = np.datetime64(start, 'ns').astype(np.int64)
					fdf['time'] = (fdf['time'].astype(np.int64) - st_ns) / 1e9 + 1e-12
					fdf['frequency'] = fdf['frequency'].astype(float)
					fdf['power'] = fdf['power'].astype(float)

					return fdf

				fdf = make_fft(start, end, gen_fft, i, sig_key)

				if methods:
					ax.plot(fdf['frequency'], fdf['power'], color='red', label='Numpy method')

					ax.xaxis.set_major_locator(FixedLocator([10, 100, 300, 1000]))
					ax.set_xticklabels([r'$10^1$', r'$10^2$', r'$3 \cdot 10^2$', r'$10^3$'])
					ax.set_xlim(left=1, right=300)
					ax.set_yscale('log')
				else:
					ax.plot(fdf['frequency'], fdf['power'], color='red', label='Numpy method')
					ax.tick_params(labelbottom=True)
					ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, -3))
					ax.set_ylim(top=1.2 * 1e-3 + 1e-4)

				ax.grid(linewidth=0.25)
			else:
				st_ns      = np.datetime64(start, 'ns').astype(np.int64)
				norm       = mpl.colors.LogNorm(vmin=vmi, vmax=vma)
				ax.set_facecolor(plt.cm.jet(norm(vmi)))
				cbar_added = False
				n_chan     = None

				for day_fdf in gen_fft(data_dir, pickle_root, sig_key, i, start, end, lpf_lim):
					day_fdf.loc[day_fdf['frequency'] == 0, 'frequency'] = 1

					if n_chan is None:
						bins   = make_log_bins(day_fdf['frequency'].unique(), i * 2)
						n_chan = len(bins) + 1
						ye     = np.arange(n_chan + 1)
						print('load')

					channel_map = {f: channelize(f) for f in day_fdf['frequency'].unique()}
					day_fdf['channel'] = day_fdf['frequency'].map(channel_map)
					t_vals = (day_fdf['time'].values.astype(np.int64) - st_ns) / 1e9 + 1e-12

					t_min = t_vals.min()
					t_max = t_vals.max()
					chunk_time_bins = np.arange(t_min, t_max + display_dt, display_dt)
					if len(chunk_time_bins) < 2:
						continue

					n_time_chunk = len(chunk_time_bins) - 1
					ix    = np.digitize(t_vals, chunk_time_bins) - 1
					iy    = day_fdf['channel'].values.astype(int)
					valid = (ix >= 0) & (ix < n_time_chunk) & (iy >= 0) & (iy < n_chan)
					flat  = ix[valid] * n_chan + iy[valid]

					sum_chunk   = np.bincount(flat, weights=day_fdf['power'].values[valid], minlength=n_time_chunk * n_chan).reshape(n_time_chunk, n_chan)
					count_chunk = np.bincount(flat, minlength=n_time_chunk * n_chan).reshape(n_time_chunk, n_chan)

					mean_chunk = np.full((n_time_chunk, n_chan), vmi)
					mask = count_chunk > 0
					mean_chunk[mask] = sum_chunk[mask] / count_chunk[mask]

					mappable = ax.pcolormesh(chunk_time_bins, ye, mean_chunk.T, norm=norm, cmap='jet')

					if not cbar_added:
						cbar = plt.colorbar(mappable=mappable, ax=ax)
						cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')
						cbar_added = True

					print('plot chunk')

				if n_chan is not None:
					ax.set_xlabel('Time (s)')
					ax.set_ylabel('Frequency (Hz)')

					ax.set_ylim(bottom=channelize(10), top=n_chan - 1)
					ax.yaxis.set_major_locator(FixedLocator(vfunc([10, 100, 1000])))
					ax.set_yticklabels(['10','100','1000'])

			ax.set_title(rf'FFT taken with {i * 2} sec window / $\Delta f = {int((i * 2) ** -1)}$')
			print('stylize\n')

	if plot:
		if start == end:
			if methods:
				plt.legend(loc='upper right')
			else:
				plt.xlabel('Scipy method Power')
				plt.ylabel('Numpy method Power')

		pngpth = './out/fft/'
		os.makedirs(os.path.dirname(pngpth), exist_ok=True)
		if save:
			if sig_key[0] == 'B':
				plt.savefig(pngpth + 'bfield.pdf', format='pdf')
			else:
				plt.savefig(pngpth + 'efield.pdf', format='pdf')
		else:
			plt.show()
			print('\nshow')


# \twriter.close()
