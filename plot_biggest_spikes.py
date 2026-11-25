"""
Every request in REQUESTS always gets its ENTIRE [start, end] calculated into
FFTs — every field, every probe, via gen_fft (the parallel/cached long-interval
path, see ffttest.py). That happens regardless of mode; a request can span
anything from a few seconds to a week+.

The mode only controls whether the request ALSO produces a graph column:

  ('spike', start, end)     On top of the full-interval calculation above,
                             also find the whistler-band spike within
                             [start, end]: a sliding WINDOW_S-second window is
                             walked across the span, scored by total FFT power
                             inside WHISTLER_BAND, and the window at the argmax
                             becomes one column of the saved spectrogram grid.
                             Run ONCE PER FIELD, over that field's OWN data: B's
                             column shows B's biggest whistler-band spike, E's
                             shows E's, and the two need not be anywhere near
                             each other in time. The spike search itself is only
                             ever used to pick what to show in the graph — it
                             never limits what gets calculated. This column is
                             drawn from data streamed back in Phase 2 below
                             (cheap — narrow, WINDOW_S-second span). A field
                             with no burst data in [start, end] contributes no
                             column at all, rather than an empty one.

  ('full', start, end)      Also becomes a graph column, but instead of
                             narrowing to a spike, plots the ENTIRE [start,
                             end] span. Streaming the whole span back through
                             Phase 2 the way 'spike' does would reproduce the
                             documented MemoryError for anything wider than a
                             few minutes, so 'full' columns are drawn via
                             ShortIntervalPlotter._stream_binned instead —
                             it reads straight from the on-disk pickle cache
                             Phase 1 already warmed, binning one file at a
                             time, so memory stays flat regardless of span
                             width.

  ('calculate', start, end) Just the full-interval calculation — no spike
                             search, no graph column. Use this when you want
                             the FFT cache (and manifest) populated for a
                             range but don't want a graph out of it.

  ('chunks', start, end)     One image PER qf-GOOD DATA CHUNK in [start, end],
                             instead of a single grid column. A "chunk" is a
                             contiguous quality-good run as defined by the CDF's
                             own quality flag (SCM qf=='GGG' / EDP quality>=2),
                             NOT a window of our own invention; files whose good
                             spans overlap in time are the same burst even when
                             their start stamps differ across probes. Chunks are
                             found PER FIELD, from that field's own quality flag,
                             so B's chunks and E's chunks are independent sets --
                             an E burst with no B counterpart still gets its own
                             images and CSVs, and neither field is ever plotted
                             over the other's intervals. Each chunk
                             becomes one PNG with the probes stacked as rows,
                             x-axis anchored at t=0 = the earliest good start in
                             the chunk (a probe that starts later sits further
                             right). Drawn by STREAMING: the FFT pool keeps
                             computing later files while the current chunk is
                             being drawn (see ffttest.stream_good_chunk_ffts +
                             ShortIntervalPlotter.plot_good_chunks). 'chunks'
                             requests get their own streaming pass and never
                             join the combined grid or the Phase 1 prewarm.
                             Output: <batch>/images/<field>/<YYYY-MM-DD>/
                             <field>_<start>_<end>.png  (this run's batch
                             folder, one subfolder per calendar day, alongside
                             csv/, intervals/, and the bins README -- not ./out)
                             Its manifests are per PROBE like every other
                             request's (see WRITE_MANIFEST), not per cluster:
                             a cluster spans probes, so it cannot report when
                             an individual probe started or stopped.

Every entry above also accepts an optional trailing show_fce bool (default
True when omitted), e.g. ('spike', start, end, False) — controls only
whether THIS request's column gets the f_ce overlay (see fce.py) drawn on
it. FGM is still downloaded for every request regardless (see
_download_request) since the download pass has no per-request visibility
into what's about to be drawn; the toggle just skips that request's
(cheap) collect_fce call and the draw itself.

Run: python plot_biggest_spikes.py
Output: ./out/spectrograms/B_..._nN.png and E_..._nN.png (N = number of
'spike' + 'full' requests; 'calculate' requests contribute no plot column).
"""
import os
import queue
import threading
from datetime import datetime, timedelta

import numpy as np

from short_interval import ShortIntervalPlotter
from spike_biggest_burst_fft import find_whistler_spike, field_spec
from ffttest import prewarm, find_cdf_files, stream_good_chunk_ffts, field_burst, SEGMENT_GAP_SECONDS
from acquisition import acquire
from fce import collect_fce
from data_paths import DATA_DIR, PICKLE_ROOT, INTERVAL_MANIFEST, BATCH_DIR
try:
	from spectro.data_intervals import (
		write_burst_manifest, write_burst_manifest_readable, merge_bursts, DataInterval)
except ImportError:
	from src.spectro.data_intervals import (
		write_burst_manifest, write_burst_manifest_readable, merge_bursts, DataInterval)

REF_FIELD     = 'B'              # UNUSED since spike windows went per-field -- each field now searches its own data (see main). Left in place, not deleted, but nothing reads it.
WHISTLER_BAND = (150, 500)       # Hz — where whistler-mode tones typically show up
PAD           = 0.25             # half-window (s); matches spike_biggest_burst_fft's burst PAD
WINDOW_S      = 10.0             # length (s) of the sliding spike-search window
NUM_BINS      = 25               # fewer log-frequency channels than the 40 default -> readable y-axis
SAVE_DIR      = './out/spectrograms'
# Interval manifests. What lands in THIS run's bundle:
#   <batch>/intervals/<start>-<end>/<field>_mms<probe>.txt
# One folder per request (including 'chunks' requests), at most 4 probes x 2
# fields = 8 files inside it -- the probes are the only thing whose data
# start/stop times differ within a request. Rows are
# start / end / duration_s / n_files. No version in these names: a batch is one
# run's snapshot, so there is nothing for it to be stale against.
#
# Grid ('spike'/'full'/'calculate') manifests are computed once into the
# cross-run cache at data_paths.INTERVAL_MANIFEST and copied here. THAT is
# where _v<MANIFEST_VERSION> lives, and it is the only reason it exists: the
# cache is keyed on a stable filename, so a rerun whose entry is newer than
# every source CDF skips recomputation, and a version bump changes the name so
# an old-format entry can never be mistaken for current. 'chunks' manifests are
# written straight here with no cache entry -- their segments fall out of the
# streaming pass that has to run anyway, so caching them would save nothing.
WRITE_MANIFEST = True
# Gap (seconds) below which two recorded intervals are treated as ONE physical
# event and merged into a single manifest row. Measured, not guessed: across
# this batch's manifests the gap distribution is empty between 1 s and 3 s --
# 99.7% of E's gaps are under 1 s (EDP's quality flag chattering inside a
# burst, thousands of millisecond-scale holes) and every real burst boundary is
# 10 s or wider, most of them minutes. 2.0 sits in that empty band. Raising it
# past ~3 s starts merging across two genuine 3-5 s B gaps; lowering it below
# ~1 s starts splitting bursts back into dropout confetti (E's Feb 23 file goes
# from 13 rows to 4865 -- and 13 is what B independently reports for the same
# day, which is the check that this threshold recovers real structure rather
# than smoothing it away). The value is written into each manifest's header so
# a reader can tell a real boundary from a sub-threshold one.
#
# 2026-10-01, Graham: one event gap for every path -- "1.5 s for both" -- so this
# now IS ffttest.SEGMENT_GAP_SECONDS, which main.py's and gen_fft's manifests
# already used. The archive census of 2026-09-30 (13,185 burst CDFs,
# _wsl_scratch/recorded_gap_census.out) puts 1.5 s in the same empty band: scb
# and dce good data has no gap between 0.65 s and 4.0 s, inside files or between
# them, so 1.5 s and 2.0 s group scb and dce identically. Only schb, which
# records in snippets, has good gaps between 1.5 and 2 s (216), and those now
# split. The "genuine 3-5 s B gaps" above are SCM's zeroed convolution edges
# (flag 'ZZZ', 2 s each side of a processing boundary), which are never good data.
EVENT_GAP_S = SEGMENT_GAP_SECONDS
# Also emit the legacy duration-only companion (<field>_mms<probe>.duration.txt)
# beside each manifest: start + duration, no end column. Off by default -- the
# manifest above already carries start, end AND duration, so this was pure
# file-count doubling. Flip it here before a run if something still wants the
# old shape.
MANIFEST_DURATION_COPY = False
SEPARATE_INTERVAL_FILES = True # True -> one file per interval instead of one combined grid per field
EXPORT_CSV    = True  # write this run's power CSVs into its batch folder: one folder per request under csv/<field>/, one CSV per calendar day per probe inside it -- see make_batch_run_dir
CSV_LAYOUT    = 'wide'           # 'wide' (default, timestamp columns) or 'tall' (~3x faster to_csv on a long window) -- see fft_excel_export._probe_dataframe
EXPORT_BINNING = True            # write the bin-index -> frequency-range README alongside the plot -- see plot_from_fft_data's bins_readme_dir

# Where qf-good data boundaries come from (SCM qf=='GGG', EDP quality>=2 -- the
# same test ffttest._cdf_to_df filters the FFT input with). The name predates
# reading CDFs directly; there are no data pickles any more.
#   False -- re-read each CDF's own quality variable every run, through
#            _scm_good_segments / _edp_good_segments. Authoritative by
#            construction; the setting to use if you suspect the generating
#            CODE is wrong (as opposed to the data having changed).
#   True  -- read them from each CDF's good-run sidecar
#            (ffttest._segments_via_sidecar): run boundaries only, computed from
#            the CDF's time and quality variables the first time and remembered
#            after. Skips a full Epoch + quality-array read per file per probe on
#            every later run -- the dominant cost of a wide 'chunks' request.
#
# True is safe even when data HAS changed. ffttest._invalidate_if_stale checks
# every sidecar against its CDF's mtime before anything reads it, so a
# re-downloaded CDF gets its runs recomputed -- and the FFT pickles computed
# from that CDF are checked against the same mtime -- regardless of this flag.
# The stat-per-file that proves a sidecar current is cheap; the quality-array
# read it saves is not.
TRUST_GOOD_PICKLES = True

# Per-field frequency ceiling — EDP has a lower useful ceiling than SCM (see main.py).
#FIELD_LPF_LIM = {'B': 1000, 'E': 500}
FIELD_LPF_LIM = {'B': 3000, 'E': 3000}

probes = [1, 2, 3, 4]
ref_probe = probes[0]
ref_spec  = field_spec(REF_FIELD)   # UNUSED -- see REF_FIELD. ref_probe above IS still used: it is which PROBE each field's own spike search scores on.

# spike for plotting biggest spike in interval, full for plotting the entire
# interval, calculate for the whole thing with no graph column. Optional
# trailing show_fce bool per entry (default True) -- see module docstring.
# spike, full, calculate, chunks
REQUESTS = [
	('spike', datetime(2021, 2, 23), datetime(2021, 2, 24)),
	('spike', datetime(2021, 3, 2),  datetime(2021, 3, 3)),
	('chunks', datetime(2021, 2, 23, 11, 7, 33), datetime(2021, 3, 2, 11, 7, 33))
]


def _missing_instruments(start, end):
	"""Cheap, local-only check (no network) -- which of this request's
	instruments (scm/edp/fgm) are missing CDF coverage for at least one
	probe. Returns a list of (inst, mode) pairs still needed; empty means
	every instrument already has at least some local coverage for every
	probe.

	Per-instrument, NOT a single bool for the whole request: a request
	missing ONLY fgm (e.g. probe 1's FGM never downloaded) must not also
	re-download scm/edp -- those already-local instruments each cost a
	real SDC round-trip (subprocess spawn+import, then pyspedas
	re-verifying/relisting every file it already has) for zero new data.
	Includes FGM alongside B/E: it's a required input now, not optional --
	collect_fce (see fce.py) needs it for the f_ce overlay every request
	draws, so missing FGM coverage must route through the download worker
	same as a missing B/E field would, but ONLY fgm, not its siblings."""
	import time as _time_mod  # TEMP diagnostic (see prints below) -- remove once the B->E gap is root-caused
	missing = []
	for inst, mode in (field_burst('B')[:2], field_burst('E')[:2], ('fgm', '')):   # B: scb or schb
		_t0 = _time_mod.monotonic()
		inst_missing = any(
			not find_cdf_files(DATA_DIR, inst, mode, start, end, probe=probe, download=False)
			for probe in probes
		)
		if inst_missing:
			missing.append((inst, mode))
			print(f'[_missing_instruments] {inst} missing after {_time_mod.monotonic() - _t0:.2f}s', flush=True)  # TEMP diagnostic
		else:
			print(f'[_missing_instruments] {inst} existence-check ({len(probes)} probes) took {_time_mod.monotonic() - _t0:.2f}s', flush=True)  # TEMP diagnostic
	return missing


# The acquisition phase's field letter for each instrument.
_LETTER = {'scm': 'B', 'edp': 'E', 'fgm': 'F'}


def _validate_present(start, end, missing):
	"""The acquisition phase's validation for the instruments this request already
	has on disk (acquisition.acquire, download=False): every CDF the request can
	open is checked once, before anything reads it. One that fails is deleted and
	counted, and its instrument joins `missing`, so the download worker fetches it
	again -- up to the 5 downloads acquire allows. Returns `missing`."""
	for inst, mode in (field_burst('B')[:2], field_burst('E')[:2], ('fgm', '')):   # B: scb or schb
		if (inst, mode) in missing:
			continue
		got = acquire(_LETTER[inst], probes, start, end, data_dir=DATA_DIR, pickle_root=PICKLE_ROOT,
		              download=False)
		if got.unresolved:
			missing.append((inst, mode))
	return missing


def _download_request(start, end, missing):
	"""Acquire ONLY the instruments _missing_instruments (or _validate_present)
	flagged as missing for this request -- see its docstring for why this must not
	unconditionally hit scm/edp/fgm regardless of what's actually absent.
	acquisition.acquire per instrument: one paced SDC call per probe (Graham,
	2026-09-28), then every CDF it can open validated, and a corrupt one fetched
	again, before the request reaches the compute queue."""
	import time as _time_mod  # TEMP diagnostic (see prints below) -- remove once the B->E gap is root-caused
	for inst, mode in missing:
		_t0 = _time_mod.monotonic()
		acquire(_LETTER[inst], probes, start, end, data_dir=DATA_DIR, pickle_root=PICKLE_ROOT)
		print(f'[_download_request] {inst} acquire took {_time_mod.monotonic() - _t0:.2f}s', flush=True)  # TEMP diagnostic


def _download_worker(download_queue, compute_queue):
	"""The ONLY thread in the program that ever triggers a download. Pulls
	jobs one at a time off download_queue -- downloads never overlap, and
	nothing else ever calls acquire(download=True) for this script, so there's no
	lock contention to reason about at all. On finishing a request's
	download, hands it off to compute_queue so the main thread can compute
	it whenever it gets there."""
	while True:
		job = download_queue.get()
		if job is None:
			return
		index, mode, start, end, show_fce, missing = job
		_download_request(start, end, missing)
		compute_queue.put((index, mode, start, end, show_fce))


def _tee_chunk_bounds(stream, sink):
	"""UNUSED since the chunks manifest went per-probe -- kept, not deleted.

	Passes a stream_good_chunk_ffts stream through to the plotter while
	recording each CLUSTER's (start, end, n_files) into `sink`. That is a
	cross-probe union (one cluster = one PNG with the probes stacked as rows),
	which is exactly why it no longer backs the manifest: the manifest is
	per-probe, and a union cannot say when probe 3 started. The chunks path now
	reads stream_good_chunk_ffts' segments_sink instead. Still the right tool if
	you ever want to log what was DRAWN rather than what was RECORDED.
	"""
	for chunk_start, chunk_end, probe_df, n_files in stream:
		sink.append((chunk_start, chunk_end, n_files))
		yield chunk_start, chunk_end, probe_df, n_files


def make_batch_run_dir():
	"""Create and return THIS RUN's batch folder: E:/PlasmaCoop/batches/<wall-
	clock run stamp>/. Named by launch time (not the data span) so re-running
	the same REQUESTS always makes a FRESH folder and never overwrites a prior
	batch. All of this run's outputs land together under it:
	  <batch>/csv/<field>/<request-span>/fft_<field>_<YYYYMMDD>_MMS<probe>.csv
	                                    (one folder per request, one CSV per
	                                     calendar day per probe inside it; a
	                                     'chunks' request's folder also holds
	                                     its own <field>_bins_README.md)
	  <batch>/intervals/<start>-<end>/<field>_mms<probe>.txt
	                                    (one folder per request, one manifest
	                                     per probe per field inside it, names
	                                     unversioned -- grid requests' copies
	                                     come from the cross-run cache at
	                                     data_paths.INTERVAL_MANIFEST, which is
	                                     the only place a version appears)
	  <batch>/<field>_<span>_bins_README.md               (grid requests)"""
	batch_dir = os.path.join(BATCH_DIR, f'{datetime.now():%Y%m%dT%H%M%S}')
	os.makedirs(batch_dir, exist_ok=True)
	return batch_dir


def main():
	plotter = ShortIntervalPlotter(data_dir=DATA_DIR, pickle_root=PICKLE_ROOT)

	# One batch folder for this whole run -- created up front so the manifest
	# mirror (prewarm, below) and the CSV/README export (plot_from_fft_data,
	# further down) all land in the same place.
	batch_dir = make_batch_run_dir()
	manifest_batch_dir = os.path.join(batch_dir, 'intervals')
	print(f'[batch] this run -> {batch_dir}', flush=True)

	download_queue = queue.Queue()
	compute_queue = queue.Queue()
	worker = threading.Thread(target=_download_worker, args=(download_queue, compute_queue), daemon=True)
	worker.start()

	# Pass 1: one fast sweep over every request, in order. Each request lands
	# on exactly one queue -- straight to compute if every instrument it
	# needs is already local, or to the download worker (carrying exactly
	# which instruments are missing) otherwise. Never blocks either way.
	# show_fce defaults True when a REQUESTS entry omits the trailing bool.
	for index, req in enumerate(REQUESTS):
		mode, start, end = req[0], req[1], req[2]
		show_fce = req[3] if len(req) > 3 else True
		missing = _validate_present(start, end, _missing_instruments(start, end))
		if not missing:
			compute_queue.put((index, mode, start, end, show_fce))
		else:
			download_queue.put((index, mode, start, end, show_fce, missing))

	# Pass 2: drain compute jobs as they become ready -- completion order, not
	# necessarily REQUESTS order (a later request needing no download can
	# finish before an earlier one still downloading). Every request's data is
	# local by the time this loop finishes; nothing is computed here yet.
	pending = [compute_queue.get() for _ in range(len(REQUESTS))]

	download_queue.put(None)  # stop signal
	worker.join()

	# 'chunks' requests are handled by their own streaming pass (see end of
	# main) and never join the combined grid or the Phase 1 prewarm below --
	# they stream their own FFT compute so drawing overlaps computation.
	chunk_pending = [p for p in pending if p[1] == 'chunks']
	grid_pending  = [p for p in pending if p[1] != 'chunks']

	# Phase 1 -- cache-warm EVERY request's window, for EVERY field/probe, via
	# exactly two shared job pools (see ffttest.prewarm). REQUESTS' date ranges
	# overlap today (the 'calculate' request spans across both 'spike'
	# requests' days) -- deduplicating at the file-job level here, rather than
	# parallelizing per-request or per-(field,probe), means the same
	# underlying CDF/FFT pickle is never submitted twice regardless of how
	# many overlapping requests need it.
	#
	# ALL windows are stream=False here, including 'spike' windows: a 'spike'
	# request's window is a full calendar day just to give find_whistler_spike
	# (below) enough range to search, but that search does its own independent
	# per-file cache read (see spike_biggest_burst_fft.py) -- it never touches
	# prewarm's returned data at all -- and only a WINDOW_S-second slice of
	# that day is ever actually plotted. Streaming a whole day back over IPC
	# for all 4 probes x 2 fields x 2 'spike' requests reproduced the exact
	# same MemoryError as the wide 'calculate' window did, just from a
	# different request mode -- fixing 'calculate' alone wasn't the whole fix.
	# This pass computes and caches every file to disk and returns nothing.
	warm_windows = [(start, end, False) for _, _, start, end, _ in grid_pending]
	warm_specs = [
		(field_spec(field)['sig_key'], probe, PAD, FIELD_LPF_LIM[field], warm_windows)
		for field in ['B', 'E']
		for probe in probes
	]
	prewarm(DATA_DIR, PICKLE_ROOT, warm_specs, manifest_dir=INTERVAL_MANIFEST,
	        manifest_batch_dir=manifest_batch_dir if WRITE_MANIFEST else None,
	        write_manifest=WRITE_MANIFEST, manifest_duration_copy=MANIFEST_DURATION_COPY,
	        event_gap_s=EVENT_GAP_S, trust_good_pickles=TRUST_GOOD_PICKLES)

	for _index, mode, _start, _end, _show_fce in grid_pending:
		if mode not in ('spike', 'full', 'calculate'):
			raise ValueError(f"REQUESTS entry has unknown mode {mode!r} (expected 'spike', 'full', or 'calculate')")

	# Every request's data is now cached; the spike searches below hit cache
	# and return instantly, regardless of which order requests are visited in.
	#
	# EVERYTHING FROM HERE IS PER FIELD. B and E are physically separate
	# instruments whose bursts do not have to coincide: an E burst can happen
	# minutes from the nearest B burst, or on a day with no B burst at all. So
	# each field runs its OWN whistler-band spike search over its OWN data and
	# gets its OWN intervals, f_ce slices, plots, and CSVs. Previously one
	# search ran on REF_FIELD ('B') and BOTH fields were drawn over that single
	# B-chosen window, which meant every E panel was showing whatever E
	# happened to be doing during a B event -- usually nothing, which is why E
	# produced empty plots and no CSVs at all. A field's intervals must never
	# be derived from the other's; the only thing they still share is the
	# Phase 1 cache warm above (which covers both fields over the full request
	# spans, so neither search pays for the other).
	paths = []
	for field in ['B', 'E']:
		spec    = field_spec(field)
		sig_key = spec['sig_key']

		results = [None] * len(REQUESTS)
		for index, mode, start, end, show_fce in grid_pending:
			if mode == 'spike':
				# Searched in THIS field's data, scored on THIS field's power.
				try:
					# lpf_lim must match FIELD_LPF_LIM, not find_whistler_spike's
					# own 1000 Hz default: the FFT cache is keyed on lpf_lim
					# (see ffttest._fft_pickle_path), so leaving it at the
					# default makes the spike search build a SECOND full set of
					# FFT pickles under pad0.250_lpf1000 that nothing else reads.
					# Scoring is unaffected -- WHISTLER_BAND is 150-500 Hz, so
					# both ceilings select the identical rows -- this only keeps
					# the search and the plot on one shared cache.
					st, ts, _power = find_whistler_spike(
						ref_probe, sig_key, start, end, spec,
						band=WHISTLER_BAND, pad=PAD, window_s=WINDOW_S, download=False,
						lpf_lim=FIELD_LPF_LIM[field],
					)
				except FileNotFoundError as exc:
					# No burst pickles, or no band power, for this field in this
					# request's span. That is a normal outcome for a field that
					# simply wasn't recording here -- it drops the interval
					# instead of drawing an empty panel over a window borrowed
					# from the other field.
					print(f'[{field}] no spike window in '
					      f'{start:%Y-%m-%d %H:%M:%S}-{end:%Y-%m-%d %H:%M:%S}: {exc}', flush=True)
					continue
				results[index] = (st, ts, 'spike', show_fce)
			elif mode == 'full':
				results[index] = (start, end, 'full', show_fce)

		intervals = [r for r in results if r is not None]
		if not intervals:
			print(f'[{field}] no intervals with data -- nothing drawn', flush=True)
			continue

		# Phase 2 -- stream back ONLY the narrow, already-picked 'spike'
		# windows that are about to be plotted from fft_by_probe. 'full'
		# columns are NOT included here: streaming a whole [start, end] span
		# back through this same in-memory path would reproduce the
		# documented MemoryError for anything wider than a few minutes --
		# ShortIntervalPlotter.plot_from_fft_data instead draws 'full'
		# columns via _stream_binned, which reads the phase-1-warmed pickle
		# cache directly, one file at a time. Every underlying file for the
		# 'spike' windows is already cache-warm from phase 1, so this call is
		# cheap disk-cache reads (get_or_compute_fft's os.path.exists
		# branch), not recomputation -- "use whatever artifacts we already
		# have" -- and small enough (WINDOW_S=10s x len(intervals)) that
		# streaming it back in full is never a memory concern regardless of
		# lpf_lim/step. Still writes its own manifest (WRITE_MANIFEST,
		# same toggle as phase 1) even though phase 1 already recorded this
		# span at the wide (whole-day) request granularity: this pass
		# writes the narrower, actually-plotted spike window as its own
		# entry, which phase 1's wider one doesn't replace -- manifests
		# should reflect every computation made, not just be deduplicated
		# down to whichever call happened first.
		plot_windows = [(st, ts, True) for st, ts, kind, _show_fce in intervals if kind == 'spike']
		fft_data = {}
		if plot_windows:
			# THIS field only -- these windows were found in this field's data
			# and mean nothing for the other one.
			plot_specs = [
				(sig_key, probe, PAD, FIELD_LPF_LIM[field], plot_windows)
				for probe in probes
			]
			# write_manifest=False on purpose. This pass re-reads the SAME files
			# phase 1 already recorded, just clipped to the narrow window about
			# to be drawn -- verified across every probe and field: each narrow
			# manifest's rows sat entirely inside a row of its phase-1 sibling,
			# and its CDF set was a subset. So it carried no coverage fact the
			# request's own manifest lacked; what it actually recorded was which
			# window got PLOTTED, which is already in the PNG and CSV names. It
			# doubled the manifests in every batch for that.
			fft_data = prewarm(DATA_DIR, PICKLE_ROOT, plot_specs, write_manifest=False,
			                   trust_good_pickles=TRUST_GOOD_PICKLES)

		# f_ce (electron cyclotron frequency) overlay data. The SOURCE is
		# field-independent (FGM, not B/E), but the WINDOWS are not -- these
		# are this field's own intervals, so the slices are taken per field
		# rather than computed once and shared.
		#
		# ONE collect_fce call PER PROBE, covering the UNION of every
		# show_fce-enabled interval's [start, end] -- not one independent
		# call per (interval, probe) cross product. A probe's FGM CDFs/
		# pickles are the same underlying files regardless of which column
		# needs them, so re-querying per interval was redundant I/O, and
		# worse, gave the narrow 'spike' column and the wide 'full' column
		# for the SAME probe two independently-resolved anchors -- if one
		# query's cheap (download=False, no bounds-verification/repair --
		# see _pick_start_anchor) anchor pick landed on a stale file right
		# at that column's own edge, that column alone would come back
		# empty even though a wider query for the same probe, moments
		# earlier or later, found the data fine. Querying the union once
		# and slicing in memory per column (cheap -- no extra CDF/pickle
		# I/O) removes that inconsistency: every column for a given probe
		# now sees the exact same underlying read.
		#
		# Columns whose request set show_fce=False don't extend the union
		# and get no dict entry -- plot_from_fft_data's
		# fce_by_col_probe.get((col, probe)) lookup naturally misses and
		# draws that cell with no overlay (see ShortIntervalPlotter._draw_onto).
		fce_cols = [
			(col, start, end)
			for col, (start, end, _kind, show_fce) in enumerate(intervals)
			if show_fce
		]
		fce_by_col_probe = {}
		if fce_cols:
			union_start = min(s for _, s, _ in fce_cols)
			union_end   = max(e for _, _, e in fce_cols)
			for probe in probes:
				times_ns, fce_hz = collect_fce(DATA_DIR, PICKLE_ROOT, union_start, union_end,
				                               probe=probe, download=False)
				for col, start, end in fce_cols:
					m = (times_ns >= np.datetime64(start, 'ns')) & (times_ns <= np.datetime64(end, 'ns'))
					fce_by_col_probe[(col, probe)] = (times_ns[m], fce_hz[m])

		fft_by_probe = {probe: fft_data.get((sig_key, probe)) for probe in probes}
		result = plotter.plot_from_fft_data(
			fft_by_probe=fft_by_probe, intervals=intervals, key=field, probes=probes,
			pad=PAD, lpf_lim=FIELD_LPF_LIM[field], num_bins=NUM_BINS,
			save_dir=SAVE_DIR, fce_by_col_probe=fce_by_col_probe,
			separate_files=SEPARATE_INTERVAL_FILES,
			export_csv_dir=batch_dir if EXPORT_CSV else None,
			csv_layout=CSV_LAYOUT,
			bins_readme_dir=batch_dir if EXPORT_BINNING else None,
		)
		# SEPARATE_INTERVAL_FILES=True makes plot_from_fft_data return a
		# LIST of paths (one per interval) instead of a single path --
		# see its docstring. Flatten into `paths`/prints either way so
		# both modes report consistently. An interval whose probes all came
		# back empty is not drawn at all, so this list can be shorter than
		# `intervals` (or empty).
		saved = [p for p in (result if SEPARATE_INTERVAL_FILES else [result]) if p is not None]
		paths.extend(saved)
		for p in saved:
				print(f'Saved: {p}')

	# Per-qf-chunk streamed plots -- one PNG per qf-good data chunk, probes as
	# rows, drawn as each burst's FFTs stream back from the pool so drawing
	# overlaps computation (the whole reason this path is separate from the
	# compute-all-then-draw-all grid above). Each 'chunks' request streams its
	# OWN FFT compute via stream_good_chunk_ffts -- that's why chunk_pending was
	# left out of the Phase 1 prewarm. Downloads for these requests already ran
	# through the same download worker as every other request, so discovery here
	# is download=False (local cache only).
	for index, mode, start, end, show_fce in chunk_pending:
		for field in ['B', 'E']:
			sig_key   = field_spec(field)['sig_key']
			chunk_dir = os.path.join(batch_dir, 'images', field)
			# segments_sink is filled with THIS field's per-probe good spans
			# before the first chunk is yielded -- see stream_good_chunk_ffts.
			# The manifest is built from it rather than from the streamed chunk
			# bounds, because a chunk is a cross-probe cluster (one PNG, probes
			# stacked as rows) and its bounds are the union across probes. The
			# whole point of a per-probe manifest is that the probes are what
			# differ; a union answers a different question.
			probe_segments = {}
			probe_recorded = {}      # the same files' recorded runs, for the manifest's recorded table
			stream = stream_good_chunk_ffts(
				DATA_DIR, PICKLE_ROOT, sig_key, probes, PAD, start, end,
				lpf_lim=FIELD_LPF_LIM[field], download=False,
				trust_good_pickles=TRUST_GOOD_PICKLES,
				segments_sink=probe_segments, recorded_sink=probe_recorded,
			)
			# This request's own CSV folder, named for the request span, with
			# one CSV per calendar day inside it (see plot_good_chunks). The
			# bin README lands in the same folder so the CSVs' bin_N columns
			# are interpretable without hunting for it.
			request_csv_dir = os.path.join(
				batch_dir, 'csv', field,
				f'chunks_{start:%Y%m%dT%H%M%S}-{end:%Y%m%dT%H%M%S}')
			saved = plotter.plot_good_chunks(
				stream,
				key=field, probes=probes, pad=PAD,
				lpf_lim=FIELD_LPF_LIM[field], num_bins=NUM_BINS,
				save_dir=chunk_dir,
				export_csv_dir=request_csv_dir if EXPORT_CSV else None,
				csv_layout=CSV_LAYOUT,
				bins_readme_dir=request_csv_dir if EXPORT_BINNING else None,
			)
			for p in saved:
				print(f'Saved (chunk): {p}')
			# Same shape as every other request's manifests: one folder named
			# for the request span, one file per (probe, field) inside it. A
			# 'chunks' request is not special here -- it is still "for this
			# request, what did probe N's field F record".
			if WRITE_MANIFEST and probe_segments:
				req_dir = os.path.join(
					manifest_batch_dir,
					f'{start:%Y%m%dT%H%M%S}-{end:%Y%m%dT%H%M%S}')
				os.makedirs(req_dir, exist_ok=True)
				for probe in sorted(probe_segments):
					# Clip to the request window and merge that probe's own
					# segments into contiguous bursts. n_files falls out of
					# merge_bursts counting distinct source stamps, so it means
					# the same thing it does in a grid request's manifest -- this
					# probe's files, not the cluster's cross-probe total.
					def _clipped(runs):
						ivs = []
						for cs, ce, ft in runs:
							cs = max(cs, start)
							ce = min(ce, end)
							if cs < ce:
								ivs.append(DataInterval(cs, ce, str(ft)))
						return ivs
					ivs = _clipped(probe_segments[probe])
					if not ivs:
						continue
					gap = timedelta(seconds=EVENT_GAP_S)
					bursts = merge_bursts(ivs, tol=gap)
					# Second table: where the instrument recorded, whatever the
					# quality, grouped the same way (Graham, 2026-09-30).
					recorded = merge_bursts(_clipped(probe_recorded.get(probe, [])), tol=gap)
					mpath = os.path.join(req_dir, f'{field}_mms{probe}.txt')
					write_burst_manifest(mpath, bursts, event_gap=gap, recorded=recorded)
					if MANIFEST_DURATION_COPY:
						write_burst_manifest_readable(
							os.path.join(req_dir, f'{field}_mms{probe}.duration.txt'),
							bursts)
					print(f'Saved (chunk manifest): {mpath}')


# Windows' multiprocessing 'spawn' start method (used by gen_fft's
# ProcessPoolExecutor) re-imports this script from scratch in every worker
# process. Without this guard, each worker would re-run the whole request
# loop -- including spawning its own nested process pools -- instead of just
# executing the one task function it was actually given.
if __name__ == '__main__':
	main()
