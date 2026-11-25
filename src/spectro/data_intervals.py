"""
Request-scoped CDF data-interval manifest.

When an FFT request is made over [st, ts], the data it actually draws from is a
set of burst CDF files, each covering a contiguous interval. This module lists
those intervals CHEAPLY: start/end come from the filename timestamps only, no
CDF contents are read. So the manifest is available the instant a request is
issued (before any FFT work) and can later drive a pick/filter step — choosing
which intervals to actually process.

Pure: stdlib only, no DataFrames, no CDF I/O. The expensive read of an
interval's samples happens later, downstream, and only for intervals the caller
keeps — i.e. we don't touch the heavy files until the last second, and nothing
here grows worse than linear in the number of files in the request.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

# Burst CDF filenames embed a 14-digit YYYYMMDDhhmmss start stamp between
# underscores, e.g. mms1_scm_brst_l2_scb_20190816095323_v2.2.1.cdf
_FNAME_RE = re.compile(r'_(\d{14})_')


class DataInterval(NamedTuple):
    start: datetime
    end: Optional[datetime]   # None when the last file's end is unknown
    path: str


class Burst(NamedTuple):
    """One physical event — the merge of intervals with no real gap between.

    ``n_files`` is how many source files the burst was assembled from, which is
    the only thing about the sources the manifest reports. The file names/paths
    themselves are deliberately not carried: they made every manifest row
    enormous (a week-long request's row ran to thousands of characters of
    absolute Windows paths) while answering a question nobody asked of this
    file — the manifest exists to say WHEN data was recorded and HOW MUCH went
    into each span, not to re-list a directory.

    ``recorded`` is how much of ``end - start`` actually holds the samples the
    row was built from. Once ``merge_bursts`` joins across a sub-threshold gap
    (see its ``tol``), the row spans time that holds none of them — EDP's
    quality flag drops ~2% of a burst in thousands of millisecond-scale holes —
    and ``end - start`` alone would silently claim it. In a manifest's used
    table those samples are the quality-good ones, so ``recorded`` is the good
    time and the file labels that column ``good``; in its recorded table they
    are every sample the instrument recorded, whatever the quality (see
    ``write_burst_manifest``). None means "not tracked", and readers should
    fall back to ``end - start``; use ``recorded_or_span`` rather than testing
    for None at each call site.
    """
    start: datetime
    end: datetime
    n_files: int
    recorded: Optional[timedelta] = None

    @property
    def recorded_or_span(self) -> timedelta:
        return self.recorded if self.recorded is not None else self.end - self.start


def _file_time(path: str) -> Optional[datetime]:
    m = _FNAME_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), '%Y%m%d%H%M%S')
    except ValueError:
        return None


def cdf_data_intervals(
    cdf_paths: Iterable[str],
    fallback_end: Optional[datetime] = None,
    bounds_resolver: Optional[Callable[[str], Tuple[Optional[datetime], Optional[datetime]]]] = None,
    segments_resolver: Optional[Callable[[str], List[Tuple[Optional[datetime], Optional[datetime]]]]] = None,
    clip_start: Optional[datetime] = None,
    clip_end: Optional[datetime] = None,
    min_run: Optional[timedelta] = None,
    run_gap: timedelta = timedelta(0),
) -> list[DataInterval]:
    """Build (start, end, path) intervals for a request's CDF list.

    Each interval spans only recorded, used data:

    * ``start`` is the true first recorded sample E_start(file) (resolver mode)
      or the filename timestamp T(file) (cheap mode).
    * ``end`` is ``min(E(file), start(next))`` — the true data end E(file), cut
      over at the next file's TRUE start (resolver mode). Contiguous burst files
      then tile with no dropped data; an overlap (E > start(next)) goes to the
      next file (B wins, as ``_compute_file_ffts`` intends); a real inter-burst
      gap (E < start(next)) ends at the data end E(file), so no part of an
      interval ever covers a no-data span. (Using the whole-second filename
      stamp here would wrongly discard up to ~1 s of real data per seam.)
    * ``clip_start`` / ``clip_end`` trim every interval to the requested
      calculation window (e.g. the request's [st, ts]); intervals left empty by
      the trim are dropped. This removes any leading/trailing span that is
      recorded-but-unused (e.g. the anchor file's data before ``st``) so the
      manifest's coverage equals the requested-and-recorded data, with internal
      no-data gaps appearing only *between* intervals, never inside one.

    Accuracy modes:

    * ``bounds_resolver=None`` (cheap — no files opened): ``end`` falls back to
      T(next), which is only an UPPER BOUND on E(file). This does NOT guarantee
      gap-free intervals; use it for an instant first pass only.
    * ``bounds_resolver=fn``: ``fn(path) -> (start, end)`` reads the file's own
      Epoch array (e.g. ``ffttest.cdf_time_bounds`` — first/last record only) to
      give the TRUE recorded bounds. Required for the gap-free guarantee above;
      only two records are read per file.
    * ``segments_resolver=fn``: ``fn(path) -> [(start, end), ...]``, one pair
      per contiguous span of data actually USED within that file (e.g.
      ``ffttest._scm_good_segments`` — quality-good sample runs). Takes
      precedence over ``bounds_resolver`` when given. Each returned pair
      becomes its own row, so a gap INSIDE a single file (data physically
      recorded but not used, e.g. quality-flagged bad) shows up as a real gap
      in the manifest too, not just gaps *between* files. A path whose
      resolved segment list is empty contributes nothing (the whole file is
      unused). A segment with ``start is None`` is dropped (unresolvable).
      Segments are NOT cut at the next file's start, the way whole-file
      bounds are: every row is a file's own good data, so two overlapping
      files' rows overlap and ``merge_bursts`` takes their union, counting
      shared time once. Cutting at the next row's start (whichever file
      that row came from) lost good data at overlapping seams: where the
      newer file's first segment began before the older file's last one,
      the newer segment was cut to milliseconds and the older one ended
      with its file, so the newer file's good data after that end was in
      no interval (2017-03-15 13:11:23.006-26.918, one event split in
      two; found by post-process verification, fixed 2026-11-07, Graham:
      "Fix the bug").
    * ``min_run`` (segments mode only): segments that belong to a run spanning
      less than ``min_run`` are dropped before anything else -- see
      ``drop_short_runs``, which ``run_gap`` is passed to. None keeps every
      segment.
    """
    if segments_resolver is not None:
        rows = []  # (start, end_or_None, path)
        for p in cdf_paths:
            for rstart, rend in segments_resolver(p):
                if rstart is None:
                    continue
                rows.append((rstart, rend, p))
        rows.sort(key=lambda r: r[0])
        if min_run is not None:
            rows = drop_short_runs(rows, min_run, run_gap)
    else:
        parsed = []
        for p in cdf_paths:
            t = _file_time(p)
            if t is not None:
                parsed.append((t, p))
        parsed.sort()

        # Resolve each file's bounds first so the cutover can use the next
        # file's *true* start. Cheap mode keeps the filename stamp (start) and
        # leaves the end unknown (None) -> filled below from the next start.
        rows = []  # (start, end_or_None, path)
        for t, p in parsed:
            if bounds_resolver is not None:
                rstart, rend = bounds_resolver(p)
                rows.append((rstart if rstart is not None else t, rend, p))
            else:
                rows.append((t, None, p))
        rows.sort(key=lambda r: r[0])

    intervals = []
    for i, (start, rend, p) in enumerate(rows):
        next_start = rows[i + 1][0] if i + 1 < len(rows) else None
        if rend is None:
            # cheap mode: end is the next file's start (an upper bound), or fallback.
            end = next_start if next_start is not None else fallback_end
        elif segments_resolver is not None:
            # A segment is its file's own good data, kept whole: an overlap
            # with another file's segment is joined, not cut (see above).
            end = rend
        elif next_start is not None:
            # Cut over at the next file's TRUE start: contiguous files tile with
            # no dropped data; an overlap (rend > next_start) goes to the next
            # file (B wins); a real gap (rend < next_start) ends at the data end.
            end = min(rend, next_start)
        else:
            end = rend

        # Trim to the requested window: no leading/trailing recorded-but-unused
        # span. Intervals that fall entirely outside the window collapse and drop.
        if clip_start is not None and start is not None and start < clip_start:
            start = clip_start
        if clip_end is not None and (end is None or end > clip_end):
            end = clip_end
        if start is not None and end is not None and start >= end:
            continue

        intervals.append(DataInterval(start, end, p))
    return intervals


def drop_short_runs(rows: List[tuple], min_run: timedelta, run_gap: timedelta) -> List[tuple]:
    """Keep the rows whose run spans at least ``min_run``; drop the rest.

    ``rows`` are ``(start, end, ...)`` tuples sorted by start, each one stretch
    of contiguous good samples (one per sidecar run). A run is a chain of rows,
    each starting no later than ``run_gap`` after the chain's running end, so a
    run that a file boundary cuts in two, or that two overlapping files both
    hold, is measured whole -- the same threshold inside a file and across a
    seam. A chain holding a row with no end is kept: it cannot be measured.

    Callers pass the sample-count rule as a span: a run of n samples spans
    n - 1 sample periods (``ffttest.good_run_rule``). Rows come back in their
    order, unchanged.
    """
    kept: List[tuple] = []
    chain: List[tuple] = []
    chain_end = None

    def close():
        if chain and (chain_end is None or chain_end - chain[0][0] >= min_run):
            kept.extend(chain)

    for row in rows:
        start, end = row[0], row[1]
        if chain and (chain_end is None or start <= chain_end + run_gap):
            chain.append(row)
            chain_end = None if (chain_end is None or end is None) else max(chain_end, end)
        else:
            close()
            chain, chain_end = [row], end
    close()
    return kept


def merge_bursts(intervals: Iterable[DataInterval],
                 tol: timedelta = timedelta(0)) -> List[Burst]:
    """Merge per-file intervals into contiguous bursts — one Burst per recording.

    Time-sorted intervals join into the same burst when the next one starts at
    or before the running end plus ``tol``, i.e. they overlap OR are within
    ``tol`` of touching. Set ``tol`` to a couple of sample periods so the normal
    seam between burst files (the next file starts exactly one sample after the
    previous ends) merges, while a real inter-burst gap (seconds to minutes)
    starts a new burst. With ``tol=0`` only true overlap/touch merges.

    Requires true bounds (build the intervals with a ``bounds_resolver``); cheap
    filename-only ends are upper bounds and will not merge correctly. Intervals
    with a missing start/end are skipped.
    """
    ivs = sorted((iv for iv in intervals if iv.start is not None and iv.end is not None),
                 key=lambda iv: iv.start)
    bursts: List[Burst] = []
    seen: List[set] = []          # distinct source paths per burst, for the count
    for iv in ivs:
        if bursts and iv.start <= bursts[-1].end + tol:
            b = bursts[-1]
            seen[-1].add(iv.path)
            # `recorded` accumulates each merged interval's own length, so it
            # counts only real samples; the gap that was just bridged is in
            # (end - start) but not here. Overlapping intervals are clamped to
            # the part past the running end so shared time is not counted twice.
            new_end = max(b.end, iv.end)
            gained = new_end - b.end if iv.start < b.end else iv.end - iv.start
            bursts[-1] = Burst(b.start, new_end, len(seen[-1]),
                               b.recorded_or_span + max(gained, timedelta(0)))
        else:
            seen.append({iv.path})
            bursts.append(Burst(iv.start, iv.end, 1, iv.end - iv.start))
    return bursts


# The two tables a manifest holds (Graham, 2026-09-30: "There should be two tables,
# one for used aka good segments and one for recorded segments"): table name ->
# (its seconds column, what its rows are). Each table opens with a
# '# table<TAB>name<TAB>what' line so a reader can tell them apart; like every
# header line it starts with '#', so a reader that skips comments skips it too.
USED_TABLE = 'used'
RECORDED_TABLE = 'recorded'
_TABLES = {
    USED_TABLE: ('good', 'quality-good segments: the data the FFTs are computed from'),
    RECORDED_TABLE: ('recorded', 'where the instrument recorded, whatever the quality'),
}


def _write_table(f, table: str, bursts: Iterable[Burst]) -> None:
    column, what = _TABLES[table]
    f.write(f'# table\t{table}\t{what}\n')
    f.write(f'# start\tend\tevent\t{column}\tn_files\n')
    for b in bursts:
        f.write(f'{b.start.isoformat(sep=" ")}\t{b.end.isoformat(sep=" ")}\t'
                f'{(b.end - b.start).total_seconds():.3f}\t'
                f'{b.recorded_or_span.total_seconds():.3f}\t{b.n_files}\n')


def write_burst_manifest(out_path: str, bursts: Iterable[Burst],
                         event_gap: Optional[timedelta] = None,
                         corrupted: Optional[Iterable[str]] = None,
                         recorded: Optional[Iterable[Burst]] = None) -> str:
    """Write a manifest: the used table (``bursts``), then the recorded table
    (``recorded``) when given, each row
    ``start<TAB>end<TAB>event<TAB>seconds<TAB>n_files``.

    Two tables because two different things are worth seeing (Graham,
    2026-09-30): where good-quality data was used, and where the instrument
    recorded at all. Measured on real days, the difference is at the edges and
    inside the bursts: SCM flags the first and last 2 s of every burst, EDP flags
    thousands of millisecond dropouts, and one EDP burst had its first 80 s
    flagged. The used table's rows are built from good runs and its seconds
    column is ``good``; the recorded table's rows are built from every recorded
    sample and its column is ``recorded``. Both group at the same ``event_gap``.

    ``event`` is ``end - start`` — how long the physical event lasted. The
    seconds column is how much of that actually holds the table's samples (see
    ``Burst``). In the used table they are equal for SCM, whose quality flag does
    not chatter inside a burst; for EDP ``good`` runs ~2% lower and the difference
    is the dropouts the row was merged across.

    ``event_gap``, when given, is written into a leading comment naming the
    threshold ``merge_bursts`` used. Without it a reader cannot tell whether
    the gap between two rows is a real boundary or merely wider than the
    threshold, which makes every row ambiguous.

    One row per contiguous burst. THE manifest format — there is no second
    machine-vs-human split any more. It carries start AND end (so the gap
    between consecutive rows is readable straight off the page, which is the
    whole reason anyone opens one of these) AND the duration (so "how long was
    this burst" needs no arithmetic). ``write_burst_manifest_readable`` below
    is the legacy duration-only view, kept only for callers that opt into it.

    Timestamps use a space separator rather than ISO's ``T``: these files are
    read by a person first, and ``read_burst_manifest`` parses either.

    ``n_files`` is the count of distinct source files that went into that span,
    and is the only source information written — see ``Burst`` for why the
    paths themselves are not. Plain text (not a pickle), safe to write from any
    platform. Returns out_path.

    ``corrupted``, when given, is written as a trailing ``# corrupted`` section:
    the files that failed to download intact enough times to be given up on. A
    row in this manifest says data IS here; without this section a file that
    never arrived is indistinguishable from data that was never taken, and the
    record of what is missing dies with the deleted file. Written as comments so
    every existing reader — ``read_burst_manifest`` included — skips it
    untouched.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        if event_gap is not None:
            f.write(f'# merged at event_gap = {event_gap.total_seconds():g} s; '
                    f'durations in seconds\n')
        else:
            f.write('# durations in seconds\n')
        _write_table(f, USED_TABLE, bursts)
        if recorded is not None:
            _write_table(f, RECORDED_TABLE, recorded)
        names = sorted(set(corrupted or ()))
        if names:
            f.write(f'# corrupted: {len(names)} file(s) never downloaded intact, '
                    f'so their data is absent from the rows above\n')
            for name in names:
                f.write(f'# corrupted\t{name}\n')
    return out_path


def write_burst_manifest_readable(out_path: str, bursts: Iterable[Burst]) -> str:
    """Write bursts human-readably: one row per contiguous burst as
    ``start<TAB>duration<TAB>duration_seconds``.

    Companion to ``write_burst_manifest`` for eyeballing a run's coverage —
    each interval as its start time plus how long it lasted, rather than the
    machine start/end triple. ``duration`` is ``H:MM:SS(.ffffff)`` (the
    ``timedelta`` str form); ``duration_seconds`` repeats it as a plain float
    for quick summing/sorting, and ``n_files`` carries the same count the
    machine manifest reports. Same one-row-per-burst contract as the machine
    writer. Returns out_path.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('# start\tduration\tduration_seconds\tn_files\n')
        for b in bursts:
            dur = b.end - b.start
            f.write(f'{b.start.isoformat(sep=" ")}\t{dur}\t'
                    f'{dur.total_seconds():.3f}\t{b.n_files}\n')
    return out_path


def read_corrupted_names(path: str) -> List[str]:
    """The basenames in a manifest's trailing ``# corrupted`` section, in file
    order, or ``[]`` when it has none (every manifest written before the section
    existed, and every run with nothing given up on).

    The counterpart to ``write_burst_manifest``'s ``corrupted`` argument. Kept
    separate from ``read_burst_manifest`` so a caller that only wants coverage
    is unaffected.
    """
    names = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 2 and parts[0] == '# corrupted':
                    names.append(parts[1])
    except OSError:
        return []
    return names


def read_burst_manifest(path: str, table: str = USED_TABLE) -> List[Burst]:
    """Parse one table of a machine burst manifest (as written by
    ``write_burst_manifest``) back into ``Burst`` objects — the inverse of that
    writer. ``table`` is ``'used'`` (the default: the good segments every
    earlier version wrote) or ``'recorded'``; a file without that table gives [].

    Used to derive the legacy duration-only view
    (``write_burst_manifest_readable``) from an already-written manifest
    without recomputing intervals. Blank lines and ``#`` header/comment lines
    are skipped; a ``# table`` line switches which table the rows below it
    belong to.

    Reads every manifest shape, so files left on disk by earlier versions parse
    alongside current ones:

    * v5 (current): two tables, each ``start end event seconds n_files``.
    * v4: ``start end event recorded n_files`` — count is the 5th column;
      ``recorded`` (then the good time) is kept, ``event`` is derived and
      ignored. Its rows are the used table: they were built from good runs.
    * v3: ``start end duration_s n_files`` — count is the 4th column;
      ``recorded`` comes back None (that format did not track it).
    * v1/v2: ``start end n_files`` — count is the 3rd column.

    Discriminated on column count, not on the header, so a headerless fragment
    still round-trips. A row missing its count entirely (or carrying a
    non-numeric one) yields a burst with a count of 0.
    """
    bursts: List[Burst] = []
    current = USED_TABLE          # rows before any '# table' line: every pre-v5 file
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('# table\t'):
                current = line.split('\t')[1]
                continue
            if not line or line.startswith('#') or current != table:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            start = datetime.fromisoformat(parts[0])
            end = datetime.fromisoformat(parts[1])
            # 5+ -> v4 (event[2], recorded[3], count[4]); 4 -> v3 (count[3]);
            # 3 -> v1/v2 (count[2]).
            count_col = 4 if len(parts) > 4 else (3 if len(parts) > 3 else 2)
            try:
                n_files = int(parts[count_col]) if len(parts) > count_col else 0
            except ValueError:
                n_files = 0
            recorded = None
            if len(parts) > 4:
                try:
                    recorded = timedelta(seconds=float(parts[3]))
                except ValueError:
                    recorded = None
            bursts.append(Burst(start, end, n_files, recorded))
    return bursts


# Bump whenever write_burst_manifest's output shape/content changes in a way
# that makes an existing file's format stale (new column, different
# delimiter, changed rounding, a fix to cdf_data_intervals/merge_bursts
# logic, etc). Embedded in the filename (not a sidecar like ffttest.py's
# DATA_PICKLE_VERSION/FFT_PICKLE_VERSION) so a version bump produces a NEW
# path instead of silently overwriting an old-format file in place, and
# never gets mistaken for "up to date" by the mtime check below (a
# different version is a different filename, so os.path.exists on the
# current version's path is simply False until it's (re)computed once).
MANIFEST_VERSION = 7   # 1->2: dropped the cdf_paths column; n_files is now distinct-file count
                       # 2->3: added the duration_s column (start/end/duration_s/n_files) and
                       #       switched timestamps to a space separator. The bump also retires
                       #       every v2 file written before the B/E pickle-collision fix in
                       #       ffttest.prewarm -- those reported EDP's good spans under SCM's
                       #       name, and because write_request_manifest below skips any manifest
                       #       newer than its source CDFs, a corrected rerun would otherwise have
                       #       been a no-op that kept the bad file. A new version is a new
                       #       filename, so v3 recomputed from scratch and the v2 files stayed
                       #       put on disk for comparison rather than being deleted.
                       # 3->4: rows are now physical EVENTS, merged across sub-threshold gaps
                       #       (see the caller's event gap), not raw contiguous-sample runs.
                       #       Columns renamed and extended to
                       #       start/end/event/recorded/n_files so the merged-over dropout
                       #       time stays visible instead of being absorbed into the span.
                       # 4->5: two tables per file (Graham, 2026-09-30: "There should be two
                       #       tables, one for used aka good segments and one for recorded
                       #       segments"). The used table is the v4 rows unchanged, its seconds
                       #       column renamed good; the recorded table groups every recorded
                       #       sample, whatever its quality, at the same event gap. The bump
                       #       also retires v4 files written before 2026-09-29's switch to the
                       #       fine sidecar, whose content changed without a bump.
                       # 5->6: the used table drops every good run of fewer than
                       #       ffttest.MIN_GOOD_RUN_SAMPLES samples before grouping (Graham,
                       #       2026-10-09: "Single digit numbers of samples do not count as
                       #       events"), so an event edge no longer sits on one. A v5 table
                       #       was cut under the old rule; the new filename keeps it from
                       #       being read as current. The recorded table is unchanged.
                       # 6->7: segments mode no longer cuts a segment at the next segment's
                       #       start (2026-11-07, Graham: "Fix the bug"). At an overlapping
                       #       CDF seam that cut lost the newer file's good data after the older
                       #       file's end and split the event; both tables were affected (the
                       #       recorded table uses segments mode too). A v6 file can hold a split
                       #       event, so the new filename retires it.


def write_request_manifest(
    manifest_root: str,
    cdf_paths: Iterable[str],
    key: str,
    inst: str,
    mode: str,
    probe: int,
    pad: float,
    st: datetime,
    ts: datetime,
    fallback_end: Optional[datetime] = None,
    bounds_resolver: Optional[Callable[[str], Tuple[Optional[datetime], Optional[datetime]]]] = None,
    segments_resolver: Optional[Callable[[str], List[Tuple[Optional[datetime], Optional[datetime]]]]] = None,
    tol: timedelta = timedelta(0),
    corrupted: Optional[Iterable[str]] = None,
    recorded_resolver: Optional[Callable[[str], List[Tuple[Optional[datetime], Optional[datetime]]]]] = None,
    min_run: Optional[timedelta] = None,
    run_gap: timedelta = timedelta(0),
) -> str:
    """Compute + write the burst manifest for one FFT request -- UNLESS an
    up-to-date one already exists, in which case this is a no-op that just
    returns the existing path. Canonical entry point — every FFT computation
    (long-interval ``gen_fft`` and short-interval ``ShortIntervalPlotter``)
    should call this exactly once per request so a manifest always exists,
    in the same place, in the same shape.

    Nested by probe and field (``inst``/``mode``) so requests for different
    fields or probes never collide on disk. The filename is STABLE for a
    given (key, st, ts, pad, MANIFEST_VERSION) — no per-call timestamp — so
    re-running the same request is a cheap no-op unless one of:

    * MANIFEST_VERSION changed -- see its own comment. A version bump is a
      new filename, so this never confuses an old-format file for current.
    * new data landed on disk for this range since the existing manifest
      was written. Detected cheaply via mtime: if any path in `cdf_paths`
      (whatever THIS call's caller just discovered) is newer than the
      existing manifest file, something was added/changed since, so this
      recomputes. A freshly downloaded CDF always has a fresh mtime, so
      this needs no content diffing or stored file-list comparison.

    ``segments_resolver``, when given, takes precedence over
    ``bounds_resolver`` — see ``cdf_data_intervals`` — so the manifest can
    report sub-file gaps (e.g. quality-bad spans) instead of only gaps
    between files.

    ``recorded_resolver`` is the same kind of callable for the recorded table:
    every run of recorded samples in a file, whatever their quality. Its rows
    are clipped and grouped exactly as the used table's are. Without it the
    recorded table is written only when no quality filter was applied
    (``segments_resolver`` None: whole-file bounds, where what was used IS what
    was recorded); otherwise the file has the used table alone.

    ``min_run`` / ``run_gap`` go to the used table's ``cdf_data_intervals``
    only: a run too short to count is no part of an event, but the recorded
    table reports where the instrument recorded, whatever came of it.

    This is the CACHE writer, and the version in the filename is what makes the
    cache safe -- see MANIFEST_VERSION. Callers that also want a per-run copy
    (e.g. a batch bundle filed by request) should copy the returned path, not
    write here twice: a batch snapshot has nothing to be stale against, so it
    carries no version.

    Path: ``{manifest_root}/mms/{probe}/{inst}/{mode}/intervals/
    {key}_{st}_{ts}_pad{pad:g}_v{MANIFEST_VERSION}.txt``
    """
    cdf_paths = list(cdf_paths)
    out_dir = os.path.join(manifest_root, 'mms', str(probe), inst, mode, 'intervals')
    fname = (f'{key}_{st:%Y%m%dT%H%M%S}_{ts:%Y%m%dT%H%M%S}_pad{pad:g}'
             f'_v{MANIFEST_VERSION}.txt')
    out_path = os.path.join(out_dir, fname)

    if os.path.exists(out_path):
        manifest_mtime = os.path.getmtime(out_path)
        newest_cdf_mtime = max(
            (os.path.getmtime(p) for p in cdf_paths if os.path.exists(p)),
            default=0.0,
        )
        if newest_cdf_mtime <= manifest_mtime:
            return out_path

    intervals = cdf_data_intervals(cdf_paths, fallback_end=fallback_end,
                                   bounds_resolver=bounds_resolver, segments_resolver=segments_resolver,
                                   clip_start=st, clip_end=ts, min_run=min_run, run_gap=run_gap)
    bursts = merge_bursts(intervals, tol=tol)
    if recorded_resolver is not None:
        recorded = merge_bursts(cdf_data_intervals(cdf_paths, fallback_end=fallback_end,
                                                   segments_resolver=recorded_resolver,
                                                   clip_start=st, clip_end=ts), tol=tol)
    elif segments_resolver is None:
        recorded = bursts
    else:
        recorded = None
    # tol IS the event gap -- pass it on so the file names the threshold it was
    # merged at rather than leaving the reader to guess.
    return write_burst_manifest(out_path, bursts, event_gap=tol, corrupted=corrupted, recorded=recorded)
