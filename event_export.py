"""
Backend for the event-export UI (event_ui.py): list the quality-good events a
request range holds, and export the ones the user picks -- one folder per
event, holding that event's FFT CSV and spectrogram.

No Qt in here: plain functions over plain data, so every step can be driven
(and tested) without a window.

Output tree for one request:
  <root>/<request folder>/
      B_bins_README.md, E_bins_README.md    one per requested field
      MMS<probe>/<field>/<event folder>/
          ffts.csv                           binned power, this event only
          graph.png                          spectrogram, this event only
Names: see top_folder_name / event_folder_name. Either file can be left out
(ExportSettings.csv / .graph); both are written by default.

Composed from pieces the pipeline already has, not re-derived:
  * Events are the rows plot_biggest_spikes' interval manifests hold --
    quality-good segments (ffttest._segments_via_sidecar, or the CDF quality
    readers when TRUST_GOOD_PICKLES is off) merged by data_intervals.
    merge_bursts at EVENT_GAP_S. One difference: the merge runs BEFORE the
    range clip (the manifest clips first), so an event the range cuts through
    always starts/ends exactly on the range edge and is flagged as clipped,
    even when the edge falls inside one of its sub-EVENT_GAP_S dropouts.
    A good run of fewer than ffttest.MIN_GOOD_RUN_SAMPLES samples is no part
    of an event (Graham, 2026-10-09), as in the manifests.
  * FFTs are planned as ffttest.prewarm plans them (one job per file) and
    binned in the pool workers (prepare_export; no FFT pickles since
    2026-11-23). Binning, panel drawing and CSV shaping are the pieces
    plot_good_chunks uses (short_interval._FftGrid, ShortIntervalPlotter.
    _draw_onto, fft_excel_export._probe_dataframe), assembled here for one
    probe and one event. Only the figure frame (title, colorbar, x-limits) is
    repeated here.
  * PAD, NUM_BINS, EVENT_GAP_S, TRUST_GOOD_PICKLES and the probe list are read
    from plot_biggest_spikes, so the UI and the driver cannot drift apart.

All times are UTC: MMS Epoch is TT2000, and pycdf converts it to UTC.
"""
import os
import re
import threading
import time as _time
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

import plot_biggest_spikes as driver
from acquisition import acquire
from data_paths import DATA_DIR, PICKLE_ROOT
from fce import collect_fce
from fft_excel_export import _probe_dataframe
from ffttest import (
	SAMPLE_RATE, cdf_check, cdf_time_bounds, field_burst, find_data_pickles, find_local_cdfs, prewarm,
	_bounds_key, _edp_good_segments, _fft_pickle_path, _parse_cdf_time,
	_plan_fft_jobs, _scan_cdf_files, _scm_good_segments, _segments_from_pickle,
	_segments_via_sidecar, good_run_rule,
)
from short_interval import CHUNK_FIG_W, ShortIntervalPlotter
from spike_biggest_burst_fft import field_spec
try:
	from spectro.data_intervals import cdf_data_intervals, merge_bursts
	from spectro.specs import BinSpec, PowerScale, empty_fill_for
except ImportError:
	from src.spectro.data_intervals import cdf_data_intervals, merge_bursts
	from src.spectro.specs import BinSpec, PowerScale, empty_fill_for

PAD                = driver.PAD
NUM_BINS           = driver.NUM_BINS
EVENT_GAP_S        = driver.EVENT_GAP_S
TRUST_GOOD_PICKLES = driver.TRUST_GOOD_PICKLES
PROBES             = tuple(driver.probes)
FIELDS             = ('B', 'E')
# Window-centre stride. Every call the driver makes leaves ffttest's default of
# 0.05 s in place, so this is the value the FFT cache on disk was built with.
STEP               = 0.05
DEFAULT_LPF_HZ     = 3000
MIN_LPF_HZ         = 100
CSV_NAME           = 'ffts.csv'
PNG_NAME           = 'graph.png'
LAYOUTS            = ('tall', 'wide')

# (instrument, mode) behind each data letter. B's mode follows its burst selection
# (scb or schb, instrument_spec.B_BURST); E is always dce. F is FGM -- needed only for f_ce.
INSTRUMENTS = {'B': field_burst('B')[:2], 'E': field_burst('E')[:2], 'F': ('fgm', '')}

# What export_event returns -- exactly one outcome per event.
WRITTEN        = 'written'
WRITTEN_NO_FCE = 'written without the f_ce line: no FGM samples in the event'
NO_FFT_ROWS    = f'skipped: no FFT rows fell inside the event (e.g. shorter than one {2 * PAD:g} s window)'
DISCARDED      = 'cancelled while it was being written: its new files were deleted'   # only with `cancelled`

# Output written whole or not at all (Graham, 2026-11-24: on a cancel "the only files that should be deleted are
# ones being currently downloaded and output artifacts currently being generated, the rest should persist"). Each
# file is written as <name>.partial and moved into place only once its event (or event image) is complete; a cancel
# before then deletes the partials, so the event's old files, if any, stay as they were. Partials open now are listed
# so a force quit can delete them (discard_partial_outputs); prepare_export sweeps any left by a dead process.
PARTIAL = '.partial'
_writing = set()
_writing_lock = threading.Lock()
_commit_lock = threading.Lock()       # held while an event's partials are moved into place; never cancelled midway


def _partial(path):
	p = path + PARTIAL
	with _writing_lock:
		_writing.add(p)
	return p


def _drop_partials(paths):
	for p in paths:
		try:
			os.remove(p)
		except OSError:
			pass
		with _writing_lock:
			_writing.discard(p)


def _commit_partials(pairs):
	"""Move each (partial, final) into place, all of them, with no cancel check in between."""
	with _commit_lock:
		for part, final in pairs:
			for attempt in range(5):           # a virus scanner or indexer can hold a fresh file for a moment
				try:
					os.replace(part, final)
					break
				except PermissionError:
					if attempt == 4:
						raise
					_time.sleep(0.2 * (attempt + 1))
			with _writing_lock:
				_writing.discard(part)


def discard_partial_outputs(wait_s=5.0):
	"""Force quit: wait (bounded) for a commit under way, then delete every partial still listed. Returns the paths
	not deleted (a file still held open by its writer on Windows)."""
	got = _commit_lock.acquire(timeout=wait_s)
	try:
		with _writing_lock:
			paths = list(_writing)
		left = []
		for p in paths:
			try:
				os.remove(p)
			except FileNotFoundError:
				pass
			except OSError:
				left.append(p)
		return left
	finally:
		if got:
			_commit_lock.release()


def sweep_partials(top_dir):
	"""Delete *.partial files under a request folder that no write of this process holds (left by a crash or a
	force quit). Returns how many were deleted."""
	with _writing_lock:
		ours = set(_writing)
	n = 0
	for d, _sub, names in os.walk(top_dir):
		for name in names:
			p = os.path.join(d, name)
			if name.endswith(PARTIAL) and p not in ours:
				try:
					os.remove(p)
					n += 1
				except OSError:
					pass
	return n


class Cancelled(Exception):
	"""Raised between units of work once the user has pressed Cancel."""


# --------------------------------------------------------------------------
# Request range and names
# --------------------------------------------------------------------------

_TIME_RE = re.compile(r'^\s*(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,2}))?)?\s*$')


def parse_time_text(text):
	"""Optional typed time -> datetime.time, or None when blank (= whole day).

	Accepts H:MM, HH:MM, HH:MM:SS, HH:MM:SS.f and HH:MM:SS.ff -- seconds and
	fractions are optional and fractions stop at hundredths, the finest a
	request can be named at.
	"""
	if text is None or not text.strip():
		return None
	m = _TIME_RE.match(text)
	if not m:
		raise ValueError(f'"{text.strip()}" is not a time -- use HH:MM, HH:MM:SS or HH:MM:SS.ff')
	hh, mm = int(m.group(1)), int(m.group(2))
	ss = int(m.group(3)) if m.group(3) else 0
	frac = m.group(4) or ''
	if hh > 23 or mm > 59 or ss > 59:
		raise ValueError(f'"{text.strip()}" is out of range (00:00 to 23:59:59.99)')
	return time(hh, mm, ss, int(frac.ljust(2, '0')) * 10_000 if frac else 0)


@dataclass(frozen=True)
class RequestRange:
	"""[start, end) in UTC: start is included, end is the first instant NOT
	included.

	A blank end time stores 00:00:00 of the day AFTER the chosen end date, so
	the chosen date is included in full: Feb 23 -> Feb 24 stores an end of
	Feb 25 00:00:00, and its folder still reads 2021-02-23_2021-02-24.
	"""
	start: datetime
	end: datetime

	@property
	def whole_days(self):
		"""Starts at 00:00:00 and ends at 00:00:00 of the day after its last
		included day -- i.e. covers whole calendar days."""
		return self.start.time() == time(0) and self.end.time() == time(0)

	@property
	def last_day(self):
		"""Last calendar day with any time inside the request."""
		return (self.end - timedelta(microseconds=1)).date()


def resolve_range(start_date, start_time, end_date, end_time):
	"""Turn the pickers into a RequestRange.

	A blank start time starts at 00:00:00 of the start date. A blank end time
	includes the whole end date: the stored end is 00:00:00 of the NEXT day,
	the first instant after it (the last instant inside the day would need a
	never-ending fraction). A typed end time is taken literally, so a typed
	00:00 on the end date stops where that date begins.
	"""
	start = datetime.combine(start_date, start_time or time(0))
	if end_time is None:
		end = datetime.combine(end_date + timedelta(days=1), time(0))
	else:
		end = datetime.combine(end_date, end_time)
	if end <= start:
		raise ValueError('The end must be after the start.')
	return RequestRange(start, end)


def name_ts(dt):
	"""Filename timestamp: 2021-02-23-21H-24M-55.05S. Every place zero-padded;
	the fraction is truncated (not rounded) to hundredths."""
	return f'{dt:%Y-%m-%d-%H}H-{dt:%M}M-{dt:%S}.{dt.microsecond // 10_000:02d}S'


def label_ts(dt):
	"""Human timestamp for plot labels and CSV headers: 2021-02-23 21:24:55.05
	(truncated to hundredths, like name_ts)."""
	return f'{dt:%Y-%m-%d %H:%M:%S}.{dt.microsecond // 10_000:02d}'


def fields_suffix(fields):
	"""'B', 'E' or 'BE' -- always in that order, whatever order `fields` is in."""
	suffix = ''.join(f for f in FIELDS if f in fields)
	if not suffix:
		raise ValueError('Select at least one field.')
	return suffix


def top_folder_name(rng, fields, lpf_hz, fce, pad=PAD, num_bins=NUM_BINS):
	"""Folder for one request.

	Whole-day ranges are named by date only -- first and last day INCLUDED
	(the stored end is 00:00:00 of the day after the last one):
	2021-02-23_2021-02-24_BE_pad0.25s_bins25_lpf3000Hz. If either end carries
	a time, both ends are full timestamps and the second is the stored end:
	2021-02-23-10H-00M-00.00S_2021-02-25-00H-00M-00.00S_B_....
	Settings that hold for every event and probe follow; _fce is last.
	"""
	if rng.whole_days:
		span = f'{rng.start:%Y-%m-%d}_{rng.last_day:%Y-%m-%d}'
	else:
		span = f'{name_ts(rng.start)}_{name_ts(rng.end)}'
	name = f'{span}_{fields_suffix(fields)}_pad{pad:g}s_bins{num_bins}_lpf{int(lpf_hz)}Hz'
	return f'{name}_fce' if fce else name


def event_folder_name(start, end):
	"""<start>_<end> of the (clipped) event, e.g.
	2021-02-23-21H-24M-55.05S_2021-02-23-21H-53M-00.99S."""
	return f'{name_ts(start)}_{name_ts(end)}'


def lpf_bounds(pad=PAD, fs=SAMPLE_RATE):
	"""(lowest, highest) frequency ceiling the UI allows, in Hz.

	The highest is the Nyquist row itself. _compute_file_ffts snaps every rfft
	bin onto the Df = 1/(2*pad) grid, which puts an even window's Nyquist bin at
	round((fs/2)/Df)*Df -- 4096 Hz at BurstRate.SCB = 8192 and 8192 Hz at BurstRate.SCHB =
	16384, both at pad = 0.25.

	This stopped one grid step short (4094) until 2026-09-13, which was right
	only while _compute_file_ffts still doubled an even window's Nyquist bin:
	that bin was overstated 2x, so withholding it hid a real error. The doubling
	is gone -- DC and Nyquist are both left undoubled now -- so the row is
	ordinary, correctly-normalised data, attenuated by the instrument's
	anti-alias filter exactly like the rows just beneath it (measured: 4094 Hz
	already sits at 0.67x the flat 4082-4092 plateau, and 4094 is not a special
	bin). Verified against the CDF's own FREQUENCY_FILTER attribute --
	`Fsamp>8192Hz ... Fmax>4096.0000Hz (=Fsamp/2=Fnyquist)` -- and a measured
	122072 ns sample step (8191.887 Hz, the clock running 13.8 ppm slow).
	"""
	df = 1.0 / (2.0 * pad)
	return MIN_LPF_HZ, int(round((fs / 2.0) / df) * df)


# --------------------------------------------------------------------------
# Local data coverage and downloads
# --------------------------------------------------------------------------

def count_local_files(letter, probe, start, end):
	"""How many local CDFs hold data inside [start, end) for one probe and one
	of B / E / F.

	Files stamped inside the range count directly. The last file stamped
	before `start` counts only when its own recorded data reaches `start`.
	(find_cdf_files with download=False takes that file on its name alone, so
	a stale file from any earlier date would look like coverage -- which is
	why this check does not use it.)
	"""
	inst, mode = INSTRUMENTS[letter]
	files = _scan_cdf_files(data_dir=DATA_DIR, inst=inst, mode=mode, probe=probe)
	n = sum(1 for ft, _p in files if start <= ft < end)
	before = [p for ft, p in files if ft < start]
	if before:
		# Validated before its bounds are read (ffttest.cdf_check): a bad one is
		# deleted and counted, and holds no data here.
		check = cdf_check(_bounds_key(inst=inst), probe, pickle_root=PICKLE_ROOT)
		try:
			last = (cdf_time_bounds(before[-1], key=_bounds_key(inst=inst), probe=probe)[1]
			        if check([before[-1]]) else None)
		except Exception:
			last = None
		if last is not None and last >= start:
			n += 1
	return n


class Coverage(dict):
	"""check_coverage's {(probe, letter): local file count}. `refetch` holds the
	(probe, letter) pairs whose validation deleted a corrupt file: the window offers
	to download those too, even when other files still cover the range."""
	refetch = frozenset()


def check_coverage(rng, letters, probes, cancelled=None):
	"""{(probe, letter): local file count} for every requested combination, as a
	Coverage.

	The acquisition phase's validation runs first (acquisition.acquire with
	download=False): every CDF the range can open is checked once, and a corrupt
	one is deleted and counted before anything here counts or reads it. Nothing
	after it validates or deletes (Graham, 2026-09-27)."""
	refetch = set()
	for letter in letters:
		got = acquire(letter, probes, rng.start, rng.end, data_dir=DATA_DIR, pickle_root=PICKLE_ROOT,
		              download=False, cancelled=cancelled)
		for name in got.unresolved:                     # CDF names: mms<probe>_...
			m = re.match(r'mms(\d)_', name)
			if m:
				refetch.add((int(m.group(1)), letter))
	coverage = Coverage({(p, l): count_local_files(l, p, rng.start, rng.end) for p in probes for l in letters})
	coverage.refetch = frozenset(refetch)
	return coverage


def download_missing(rng, missing, log=print, cancelled=None):
	"""Fetch what the SDC has for each (probe, letter) in `missing` over the range,
	through the acquisition phase: one acquisition.acquire per instrument, which
	requests each probe in its own paced SDC call (Graham, 2026-09-28), validates
	everything it can open, and fetches a corrupt file again, up to 5 downloads.
	This is the only place the SDC is ever contacted. `cancelled` is checked between
	calls and while waiting out the gap between them."""
	by_letter = {}
	for probe, letter in missing:
		by_letter.setdefault(letter, set()).add(probe)
	for letter in sorted(by_letter):
		acquire(letter, sorted(by_letter[letter]), rng.start, rng.end, data_dir=DATA_DIR,
		        pickle_root=PICKLE_ROOT, cancelled=cancelled, log=log)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
	probe: int
	field: str             # 'B' or 'E'
	start: datetime        # clipped to the request range
	end: datetime
	clipped_start: bool    # the physical event began before the range
	clipped_end: bool      # ... or ran past it

	@property
	def duration_s(self):
		return (self.end - self.start).total_seconds()


def _py(t):
	return pd.Timestamp(t).to_pydatetime(warn=False)


def clip_bursts(bursts, rng, probe, field):
	"""Clip merged bursts to the range -> Events. A burst wholly outside is
	dropped; one the range cuts is kept up to the edge and flagged."""
	out = []
	for b in bursts:
		s, e = _py(b.start), _py(b.end)
		cs, ce = max(s, rng.start), min(e, rng.end)
		if cs < ce:
			out.append(Event(probe, field, cs, ce, s < rng.start, e > rng.end))
	return out


def find_events(rng, fields, probes, cancelled=lambda: False, log=print):
	"""Every quality-good event in the range for each (probe, field), sorted by
	start. Checks the CDFs it finds (an unreadable one is deleted and logged, as
	the driver does) and reads their good runs from the per-CDF sidecars;
	computes no FFTs."""
	specs = [(field_spec(f)['sig_key'], p, PAD, DEFAULT_LPF_HZ, [(rng.start, rng.end, False)])
	         for f in fields for p in probes]
	if not specs:
		return []
	log(f'Finding events: {len(fields)} field(s) x {len(probes)} probe(s) ...')
	fft_jobs, fft_owner, manifest_specs = _plan_fft_jobs(
		data_dir=DATA_DIR, pickle_root=PICKLE_ROOT, specs=specs, download=False)
	if cancelled():
		raise Cancelled
	# (file_time, probe, field) -> CDF, for every file that survived discovery's
	# stale-anchor check. The field letter is part of the key for the reason
	# prewarm documents: SCM and EDP files share filename stamps.
	cdf_by = {(job[0][2], fft_owner[pkl][1], fft_owner[pkl][0][0]): job[0][0]
	          for pkl, job in fft_jobs.items()}

	events = []
	for key, inst, _mode, probe, _pad, fs, window_files in manifest_specs:
		fld = key[0]
		cdfs = [c for _st, _ts, _stream, files in window_files for c in files
		        if (_parse_cdf_time(path=c), probe, fld) in cdf_by]

		def segments(cdf, probe=probe, fs=fs, fld=fld, inst=inst):
			if cancelled():
				raise Cancelled
			if TRUST_GOOD_PICKLES:
				return _segments_via_sidecar(
					cdf_path=cdf_by[(_parse_cdf_time(path=cdf), probe, fld)], key=fld, fs=fs,
					probe=probe, pickle_root=PICKLE_ROOT)
			finder = _scm_good_segments if inst == 'scm' else _edp_good_segments
			return finder(cdf, probe)

		bursts = merge_bursts(cdf_data_intervals(cdfs, segments_resolver=segments, **good_run_rule(fs)),
		                      tol=timedelta(seconds=EVENT_GAP_S))
		found = clip_bursts(bursts, rng, probe, fld)
		log(f'  MMS{probe} {fld}: {len(cdfs)} file(s), {len(found)} event(s)')
		events.extend(found)
	events.sort(key=lambda e: (e.start, e.probe, e.field))
	return events


# --------------------------------------------------------------------------
# FGM coverage of an event
# --------------------------------------------------------------------------

FGM_FULL, FGM_PARTIAL, FGM_NONE = 'full', 'partial', 'none'


@dataclass(frozen=True)
class FgmCoverage:
	"""How much of one event the probe's FGM data covers (FGM feeds f_ce)."""
	state: str                      # FGM_FULL, FGM_PARTIAL or FGM_NONE
	fraction: float                 # 0..1 of the event; exactly 1.0 when full, 0.0 when none
	gaps: Tuple[Tuple[datetime, datetime], ...] = ()    # the uncovered stretches that count

	@property
	def percent(self):
		"""Whole percent for display. Partial never reads 0 or 100, which would
		look like the other two states."""
		if self.state != FGM_PARTIAL:
			return 100 if self.state == FGM_FULL else 0
		return min(99, max(1, int(self.fraction * 100)))


def fgm_runs(probe, rng, tol_s=EVENT_GAP_S, log=print):
	"""Contiguous FGM runs [(start, end), ...] for one probe over the range,
	from the local FGM burst CDFs' own sample times, merged across gaps shorter
	than tol_s -- the same rule that merges segments into events.

	Reads every local FGM file stamped inside the range, plus the last one
	stamped before it (its data may run into the range). FGM files are small
	(1-4 MB), and _epoch_datetime64 reads their times without a datetime object
	per sample, so this needs no cache. An unreadable file is logged and skipped."""
	from spacepy import pycdf
	from ffttest import _epoch_datetime64
	inst, mode = INSTRUMENTS['F']
	files = _scan_cdf_files(data_dir=DATA_DIR, inst=inst, mode=mode, probe=probe)
	before = [i for i, (ft, _p) in enumerate(files) if ft < rng.start]
	wanted = [path for ft, path in files[before[-1] if before else 0:] if ft < rng.end]
	# Validated before any is opened (ffttest.cdf_check): a bad one is deleted and
	# counted toward the 5 strikes instead of being opened, logged and left there.
	usable = cdf_check('F', probe, pickle_root=PICKLE_ROOT)(wanted)
	chunks = []
	for path in usable:
		try:
			with pycdf.CDF(path) as cdf:
				t = _epoch_datetime64(cdf, 'Epoch').as_unit('ns').asi8
		except Exception as exc:
			log(f'  MMS{probe} FGM: cannot read {os.path.basename(path)} ({type(exc).__name__}: {exc}) -- skipped')
			continue
		if len(t):
			chunks.append(t)
	if not chunks:
		return []
	t = np.sort(np.concatenate(chunks))
	breaks = np.flatnonzero(np.diff(t) >= tol_s * 1e9)
	starts = np.concatenate(([0], breaks + 1))
	ends = np.concatenate((breaks, [len(t) - 1]))
	return [(_py(t[s]), _py(t[e])) for s, e in zip(starts, ends)]


def coverage_of(start, end, runs, tol_s=EVENT_GAP_S):
	"""FgmCoverage of [start, end) by FGM `runs`.

	An uncovered stretch shorter than tol_s does not count against the event --
	at either edge or inside -- because runs are merged over gaps that short
	anyway. Full: no uncovered stretch of tol_s or more. None: no overlap at all.
	Partial: anything between, with `fraction` the share left after the
	uncovered stretches that count."""
	span = (end - start).total_seconds()
	overlapping = sorted((max(s, start), min(e, end)) for s, e in runs if s < end and e > start)
	if span <= 0 or not overlapping:
		return FgmCoverage(FGM_NONE, 0.0)
	gaps, cursor = [], start
	for s, e in overlapping:
		if s > cursor:
			gaps.append((cursor, s))
		cursor = max(cursor, e)
	if cursor < end:
		gaps.append((cursor, end))
	counted = tuple(g for g in gaps if (g[1] - g[0]).total_seconds() >= tol_s)
	if not counted:
		return FgmCoverage(FGM_FULL, 1.0)
	missing = sum((b - a).total_seconds() for a, b in counted)
	return FgmCoverage(FGM_PARTIAL, max(0.0, 1.0 - missing / span), counted)


def fgm_coverage(events, rng, runs_for=None, log=print):
	"""{event: FgmCoverage} -- FGM runs are read once per probe, not per event."""
	runs_for = runs_for or (lambda probe: fgm_runs(probe, rng, log=log))
	runs = {p: runs_for(p) for p in sorted({e.probe for e in events})}
	return {ev: coverage_of(ev.start, ev.end, runs[ev.probe]) for ev in events}


# --------------------------------------------------------------------------
# Grouping probes across events, and leaving probes out
# --------------------------------------------------------------------------

GROUP_START_S = 2.0      # events on different probes starting this close form one grouped event


@dataclass(frozen=True)
class EventGroup:
	"""Events of one field whose starts lie within GROUP_START_S of the first
	one's. The group runs from that first start to the latest end."""
	field: str
	members: Tuple[Event, ...]          # in start order

	@property
	def probes(self):
		return tuple(sorted({m.probe for m in self.members}))

	@property
	def start(self):
		return self.members[0].start

	@property
	def end(self):
		return max(m.end for m in self.members)

	@property
	def duration_s(self):
		return (self.end - self.start).total_seconds()

	def as_events(self):
		"""One Event per probe over the whole group window -- what Run writes for
		a grouped row, so every probe's files line up in time.

		The probe whose event starts the group comes first. event_images names an
		image after the first event at its start time, and every probe here shares
		that start, so this order is what keeps a grouped run naming its images
		after the same probe an ungrouped run does."""
		clipped_start = any(m.clipped_start for m in self.members)
		clipped_end = any(m.clipped_end for m in self.members)
		first = self.members[0].probe
		order = [first] + [p for p in self.probes if p != first]
		return [Event(p, self.field, self.start, self.end, clipped_start, clipped_end) for p in order]


def without_probes(events, excluded):
	"""The events, minus every one on an excluded probe."""
	excluded = set(excluded)
	return [e for e in events if e.probe not in excluded]


def group_events(events, tol_s=GROUP_START_S):
	"""EventGroups, per field. Events are taken in start order, and one joins
	the current group when it starts within tol_s of that group's FIRST start --
	not of its latest member, so a group cannot creep along a chain of starts.
	Leaving a probe out and regrouping recalculates each group's start and end
	from the members that remain."""
	groups = []
	for fld in FIELDS:
		current = []
		for ev in sorted((e for e in events if e.field == fld), key=lambda e: (e.start, e.probe)):
			if current and (ev.start - current[0].start).total_seconds() <= tol_s:
				current.append(ev)
				continue
			if current:
				groups.append(EventGroup(fld, tuple(current)))
			current = [ev]
		if current:
			groups.append(EventGroup(fld, tuple(current)))
	groups.sort(key=lambda g: (g.start, FIELDS.index(g.field)))
	return groups


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportSettings:
	root: str                  # folder the user chose
	rng: RequestRange
	fields: Tuple[str, ...]
	lpf_hz: int
	fce: bool
	layout: str = 'tall'       # 'tall' or 'wide' -- see _probe_dataframe
	# Colour-scale limits, one PowerScale per field. Purely visual: they change
	# graph.png and nothing else -- ffts.csv is byte-identical whatever they are.
	# That is why they are deliberately NOT part of top_dir: re-running a request
	# at a different scale overwrites the previous PNGs in place, by choice.
	scales: Optional[Mapping[str, 'PowerScale']] = None
	# Which of the two files each event folder gets (Graham, 2026-10-07: "If there
	# is no option to export only CSVs or only graphs in the app, it should be
	# created"). Both by default, as always; at least one. They choose files
	# only: what is in a file written either way is the same, byte for byte.
	csv: bool = True
	graph: bool = True

	@property
	def top_dir(self):
		return os.path.join(self.root, top_folder_name(self.rng, self.fields, self.lpf_hz, self.fce))

	def scale_for(self, field):
		"""The colour scale for one field: the caller's, or that field's default."""
		if self.scales and field in self.scales:
			return self.scales[field]
		return PowerScale.for_field(field)

	def validate(self):
		lo, hi = lpf_bounds()
		if not lo <= self.lpf_hz <= hi:
			raise ValueError(f'The frequency ceiling must be between {lo} and {hi} Hz.')
		if self.layout not in LAYOUTS:
			raise ValueError(f'The CSV layout must be one of {LAYOUTS}.')
		if not (self.csv or self.graph):
			raise ValueError('Choose the CSV, the graph or both.')
		for fld in self.fields:
			self.scale_for(fld)        # PowerScale validates itself on construction
		fields_suffix(self.fields)


def event_dir(top_dir, ev):
	return os.path.join(top_dir, f'MMS{ev.probe}', ev.field, event_folder_name(ev.start, ev.end))


def write_event_csv(path, start, time_bins, mean_g, field, n_chan, layout):
	"""One CSV for one event and probe: _probe_dataframe's exact shape, but with
	time headers to hundredths. _probe_dataframe prints tenths while cells are
	0.05 s apart, so every header would appear twice.

	Emptiness is decided against the fixed no-data marker, NOT against the
	colour-scale floor. Those were the same number until the scale became a user
	option; comparing to a user-chosen floor would silently change which rows the
	CSV drops every time someone re-scaled a plot."""
	p_cnt = (mean_g != empty_fill_for(field)).astype(int)
	df = _probe_dataframe(t_bins=time_bins, mean=mean_g, p_cnt=p_cnt, st=start, n_chan=n_chan,
	                      with_date=True, layout=layout)
	keep = (p_cnt > 0).any(axis=1)                  # the drop_empty mask _probe_dataframe applies
	left = np.asarray(time_bins[:mean_g.shape[0]], dtype=float)[keep]
	labels = [label_ts(start + timedelta(seconds=float(t))) for t in left]
	if layout == 'wide':
		if df.shape[1] - 2 != len(labels):          # trailing max_power / min_power columns
			raise RuntimeError('CSV time columns do not match the grid -- _probe_dataframe changed shape')
		df.columns = labels + list(df.columns[-2:])
	else:
		if len(df) - 2 != len(labels):             # trailing max_power / min_power rows
			raise RuntimeError('CSV time rows do not match the grid -- _probe_dataframe changed shape')
		df.iloc[:len(labels), df.columns.get_loc('time_utc')] = labels
	df.to_csv(path, float_format='%.6e', index=(layout == 'wide'))
	return path


def _figure_title(fld):
	units = (r'\frac{\text{nT}^2}{\text{Hz}}' if fld == 'B'
	         else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}')
	return (fr'$|{fld}|$ vs. Frequency vs. Time  $\left[{units}\right]$   '
	        fr'({PAD * 2:.3g} s window, $\Delta f = {int(1 / (PAD * 2))}$ Hz)')


def draw_event_png(path, plotter, ev, binned, fce=None):
	"""One probe, one event. The panel is _draw_onto -- the look every other
	spectrogram here has; only the frame around it lives here. Built on a bare
	Figure (no pyplot), so it is safe off the main thread."""
	mean_g, time_bins, ye, bins, n_chan, vmi, vma, dd_used = binned
	fig = Figure(figsize=(CHUNK_FIG_W, 4.0), layout='constrained')
	FigureCanvasAgg(fig)
	ax = fig.subplots()
	mappable = plotter._draw_onto(
		ax=ax, mean_g=mean_g, time_bins=time_bins, ye=ye, bins=bins, n_chan=n_chan,
		vmi=vmi, vma=vma, key=ev.field, pad=PAD, start=ev.start, display_dt=dd_used,
		is_top_row=False, is_bottom_row=True, is_left_col=True, probe=ev.probe, fce=fce,
	)
	if ev.duration_s > 0:
		ax.set_xlim(0, ev.duration_s)
	ax.set_xlabel(f'Seconds since {label_ts(ev.start)} UTC')
	end_txt = label_ts(ev.end) if ev.end.date() != ev.start.date() else label_ts(ev.end)[11:]
	fig.suptitle(f'{_figure_title(ev.field)}\n{label_ts(ev.start)} – {end_txt} UTC')
	cbar = fig.colorbar(mappable, ax=ax, location='right', shrink=0.9)
	cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')
	fig.savefig(path, dpi=150, bbox_inches='tight', format='png')     # format: path may be a .partial
	return path


class BinnedFfts:
	"""Every window's binned FFT grid for one request, as prepare_export leaves it: the 8-tuple
	ShortIntervalPlotter._bin_fft_files returns (or None: no FFT rows in the window), keyed by
	(field, probe, start, end). `complete` is False when the computation was cancelled part way."""

	def __init__(self):
		self._grids = {}
		self.complete = True

	def put(self, field, probe, start, end, binned):
		self._grids[(field, probe, start, end)] = binned

	def has(self, field, probe, start, end):
		return (field, probe, start, end) in self._grids

	def get(self, field, probe, start, end):
		try:
			return self._grids[(field, probe, start, end)]
		except KeyError:
			raise RuntimeError(f'no FFTs were prepared for MMS{probe} {field} {start} - {end}: '
			                   f'prepare_export must be given this window') from None

	def __len__(self):
		return len(self._grids)


_prepared = None      # the last prepare_export's BinnedFfts; export_event and event_images default to it


def forget_prepared():
	"""Drop the last request's grids (the event window's Run calls this when it ends)."""
	global _prepared
	_prepared = None


def prepared_ffts(binned=None):
	"""`binned`, or the last prepare_export's grids."""
	store = binned if binned is not None else _prepared
	if store is None:
		raise RuntimeError('no FFTs prepared: call prepare_export first')
	return store


def prepare_export(settings, events, log=print, extra_windows=None, cancelled=None, progress=None):
	"""Everything the events share, done once before any of them is written:
	check the settings, create the request folder and its bins READMEs, and
	compute every event's FFTs in ONE worker pool -- files shared between
	events are computed once and never raced (the file-level dedup prewarm
	exists for; its job planning is the one used here).

	The FFTs are binned in the workers (2026-11-23, Graham: "Get rid of the
	binning scratch"): each file's spectra stay in its worker, which returns only
	the cells they fall in for each window the file feeds (_bin_fft_job), and
	those are folded here, in each window's file order, into the same grid
	_bin_fft_files used to build from the file's pickle. Nothing is written to
	disk. Returns the BinnedFfts (also kept for export_event / event_images).

	extra_windows: optional {(field, probe): [(start, end, False), ...]} computed
	in that same call -- the event images' windows (event_images.fft_windows),
	which reach probes and times the selected events alone do not. One call
	means one worker pool, which matters on Windows where every pool start
	re-imports the pipeline in each worker.

	cancelled: optional callable; once true, no further file is started, the
	files the workers hold are waited for and their results dropped, and this
	returns with BinnedFfts.complete False. progress: optional
	progress(files_done, files_total)."""
	global _prepared
	_prepared = None
	settings.validate()
	os.makedirs(settings.top_dir, exist_ok=True)
	swept = sweep_partials(settings.top_dir)
	if swept:
		log(f'Deleted {swept} partial file(s) a stopped earlier Run left in {settings.top_dir}.')
	plotter = ShortIntervalPlotter(data_dir=DATA_DIR, pickle_root=PICKLE_ROOT)
	for fld in fields_suffix(settings.fields):
		plotter._write_bins_readme_for(bins_readme_dir=settings.top_dir, key=fld, pad=PAD,
		                               lpf_lim=settings.lpf_hz, num_bins=NUM_BINS, step=STEP)
	windows = {}
	for ev in events:
		windows.setdefault((ev.field, ev.probe), []).append((ev.start, ev.end, False))
	for pair, extra in (extra_windows or {}).items():
		windows.setdefault(pair, []).extend(extra)
	store = BinnedFfts()
	if windows:
		log(f'Computing FFTs for {len(events)} event(s) ...')
		_bin_windows(settings, windows, store, cancelled=cancelled, progress=progress)
	_prepared = store
	return store


def _bin_windows(settings, windows, store, cancelled=None, progress=None):
	"""prepare_export's FFT step: plan the file jobs (prewarm's planning), give each job the windows its file
	feeds, run them on a stoppable pool and fold the partials into one _FftGrid per window."""
	from ffttest import MAX_POOL_WORKERS, _pool_imap
	from short_interval import _FftGrid, _bin_fft_job
	fft_jobs, _owner, _specs = _plan_fft_jobs(
		DATA_DIR, PICKLE_ROOT,
		[(field_spec(f)['sig_key'], p, PAD, settings.lpf_hz, w) for (f, p), w in windows.items()],
		step=STEP, download=False)
	binning = BinSpec(lpf_lim=settings.lpf_hz, num_bins=NUM_BINS, display_dt=None)
	bins, n_chan = ShortIntervalPlotter._analytic_log_bins(PAD, binning.lpf_lim, binning.num_bins)
	grids, order, at, waiting, feeds = {}, {}, {}, {}, {}
	targets = []                       # [(field, probe, start, end)], the target ids are indexes into it
	for (f, p), wins in windows.items():
		for st, ts, _stream in wins:
			key = (f, p, st, ts)
			if key in feeds:
				continue
			feeds[key] = tid = len(targets)
			targets.append(key)
			# The files _bin_fft_files was given: the window's own discovery (_event_fft_paths), in its order,
			# each one a job of this request
			order[tid] = [fp for fp in _window_fft_paths(f, p, st, ts) if fp in fft_jobs]
			grids[tid] = _FftGrid(f, PAD, st, ts, binning, STEP, bins, n_chan, scale=settings.scale_for(f))
			at[tid], waiting[tid] = 0, {}
	job_targets = {}
	for tid, paths in order.items():
		f, p, st, ts = targets[tid]
		for fp in paths:
			job_targets.setdefault(fp, []).append((tid, (f, PAD, st, ts, binning, STEP)))

	def finish(tid):
		store.put(*targets[tid], grids.pop(tid).result())

	for tid in list(order):
		if not order[tid]:
			finish(tid)
	keys = [fp for fp in fft_jobs if fp in job_targets]
	total, done = len(keys), 0
	if progress is not None:
		progress(0, total)
	results = _pool_imap(_bin_fft_job, ((fft_jobs[fp], job_targets[fp]) for fp in keys), MAX_POOL_WORKERS,
	                     'FFT computation', cancelled=cancelled)
	for fp, parts in zip(keys, results):
		for tid, _geom in job_targets[fp]:
			waiting[tid][fp] = parts.get(tid)
			paths = order[tid]
			while at[tid] < len(paths) and paths[at[tid]] in waiting[tid]:
				grids[tid].add_partial(waiting[tid].pop(paths[at[tid]]))
				at[tid] += 1
			if at[tid] == len(paths):
				finish(tid)
		done += 1
		if progress is not None:
			progress(done, total)
	store.complete = done == total
	return store


def _window_fft_paths(field, probe, start, end):
	"""The FFT pickle path of every file covering [start, end] for (field, probe), in discovery order -- what
	identifies a file's job in prewarm's planning (_plan_fft_jobs keys its jobs by it); none need exist."""
	inst, mode = INSTRUMENTS[field]
	return [_fft_pickle_path(pickle_root=PICKLE_ROOT, inst=inst, mode=mode, pad=PAD, file_time=ftime, probe=probe,
	                         step=STEP)
	        for ftime, _cdf in find_local_cdfs(DATA_DIR, inst, mode, start, end, probe=probe,
	                                           pickle_root=PICKLE_ROOT,
	                                           check=cdf_check(field, probe, pickle_root=PICKLE_ROOT))]


def _event_fft_paths(ev):
	"""The FFT pickles covering this event.

	One cache serves every ceiling now: the pickles hold the full spectrum and
	the ceiling is applied where the rows are binned, so changing the ceiling in
	the UI no longer builds a second copy of the cache.

	The export no longer reads pickles (prepare_export bins in the workers since
	2026-11-23); kept for the scripts that still do.
	"""
	return [fp for fp in _window_fft_paths(ev.field, ev.probe, ev.start, ev.end) if os.path.exists(fp)]


def export_event(settings, ev, log=print, binned=None, cancelled=None):
	"""Write and draw one event whose FFTs prepare_export has computed and
	binned: ffts.csv when settings.csv, graph.png when settings.graph (f_ce is
	read only for the graph). binned: that call's BinnedFfts (default: the last
	one). Returns the event's outcome: WRITTEN, WRITTEN_NO_FCE or NO_FFT_ROWS
	(in which case nothing is written), or DISCARDED. Files of an event run
	again are overwritten; nothing else is touched.

	Each file is written as a .partial and moved into place once the event is
	complete. cancelled: optional callable checked after each file; once true,
	the event's partials are deleted (its old files, if any, stay) and this
	returns DISCARDED. Without it (the census) an event is always finished."""
	plotter = ShortIntervalPlotter(data_dir=DATA_DIR, pickle_root=PICKLE_ROOT)
	tag = f'MMS{ev.probe} {ev.field} {label_ts(ev.start)}'
	binned = prepared_ffts(binned).get(ev.field, ev.probe, ev.start, ev.end)
	if binned is None:
		log(f'  {tag}: {NO_FFT_ROWS}')
		return NO_FFT_ROWS
	out = event_dir(settings.top_dir, ev)
	os.makedirs(out, exist_ok=True)
	# vmi/vma are the DISPLAY limits now, so the CSV cannot use them to tell an
	# empty cell from a real one -- it is handed the field instead.
	mean_g, time_bins, _ye, _bins, n_chan, _vmi, _vma, _dd = binned
	stop = cancelled or (lambda: False)
	staged = []
	try:
		if settings.csv:
			final = os.path.join(out, CSV_NAME)
			staged.append((_partial(final), final))
			write_event_csv(staged[-1][0], ev.start, time_bins, mean_g, ev.field, n_chan, settings.layout)
		fce = None
		if settings.graph and not stop():
			if settings.fce:
				fce = collect_fce(DATA_DIR, PICKLE_ROOT, ev.start, ev.end, probe=ev.probe, download=False)
			final = os.path.join(out, PNG_NAME)
			staged.append((_partial(final), final))
			draw_event_png(staged[-1][0], plotter, ev, binned, fce)
		if stop():
			_drop_partials([p for p, _f in staged])
			log(f'  {tag}: {DISCARDED}')
			return DISCARDED
		_commit_partials(staged)
	except BaseException:
		_drop_partials([p for p, _f in staged])
		raise
	outcome = WRITTEN_NO_FCE if settings.graph and settings.fce and fce[0].size == 0 else WRITTEN
	log(f'  {tag}: {outcome} -> {out}')
	return outcome
