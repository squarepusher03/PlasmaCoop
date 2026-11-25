"""
Tests for spectro.data_intervals (request-scoped CDF interval manifest).

Runs as a plain script — no pytest dependency:

    python tests/test_data_intervals.py

Each test_* function asserts one independently-verifiable thing. All pure:
no CDF files are opened — intervals come from filenames only.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta

# Make `import spectro...` work whether run from project root or tests/.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

from spectro.data_intervals import (
	DataInterval,
	Burst,
	cdf_data_intervals,
	merge_bursts,
	write_burst_manifest,
	write_burst_manifest_readable,
	read_burst_manifest,
)

_PATHS = [
	'p/mms1_scm_brst_l2_scb_20190816095323_v2.2.1.cdf',
	'p/mms1_scm_brst_l2_scb_20190816095533_v2.2.1.cdf',  # +130s, contiguous
	'p/mms1_scm_brst_l2_scb_20190816100000_v2.2.1.cdf',  # later (gap)
]
_FALLBACK = datetime(2019, 8, 16, 10, 2, 0)


def test_end_is_next_start():
	ivs = cdf_data_intervals(_PATHS)
	assert ivs[0].end == ivs[1].start
	assert ivs[1].end == ivs[2].start


def test_last_end_uses_fallback():
	ivs = cdf_data_intervals(_PATHS, fallback_end=_FALLBACK)
	assert ivs[-1].end == _FALLBACK
	# Without a fallback the last end is unknown.
	assert cdf_data_intervals(_PATHS)[-1].end is None


def test_skips_unparseable_and_sorts():
	shuffled = [_PATHS[2], 'p/no_timestamp_here.cdf', _PATHS[0], _PATHS[1]]
	ivs = cdf_data_intervals(shuffled)
	assert len(ivs) == 3  # junk dropped
	starts = [iv.start for iv in ivs]
	assert starts == sorted(starts)  # output is time-ordered


def test_bounds_resolver_overrides_with_true_ends():
	# A fake resolver standing in for ffttest.cdf_time_bounds: returns true
	# (start, end) read from file contents. End must NOT be the next file's start.
	true_bounds = {
		_PATHS[0]: (datetime(2019, 8, 16, 9, 53, 23), datetime(2019, 8, 16, 9, 55, 23)),
		_PATHS[1]: (datetime(2019, 8, 16, 9, 55, 33), datetime(2019, 8, 16, 9, 57, 33)),
		_PATHS[2]: (datetime(2019, 8, 16, 10, 0, 0), datetime(2019, 8, 16, 10, 2, 0)),
	}
	ivs = cdf_data_intervals(_PATHS, bounds_resolver=lambda p: true_bounds[p])
	# E(file_0) is the resolver's end, strictly before the next file's start.
	assert ivs[0].end == datetime(2019, 8, 16, 9, 55, 23)
	assert ivs[0].end != ivs[1].start         # E(A) != T(B) — the whole point
	assert ivs[0].end < ivs[1].start          # the inter-burst gap is excluded


def test_overlap_clips_end_to_next_start():
	# E(file_0) runs PAST file_1's start (overlap). The FFT pipeline cuts over at
	# T(next), so the next file wins the shared region: end clips to T(B).
	overlap = {
		_PATHS[0]: (datetime(2019, 8, 16, 9, 53, 23), datetime(2019, 8, 16, 9, 56, 0)),  # ends after _PATHS[1] start
		_PATHS[1]: (datetime(2019, 8, 16, 9, 55, 33), datetime(2019, 8, 16, 9, 57, 33)),
		_PATHS[2]: (datetime(2019, 8, 16, 10, 0, 0), datetime(2019, 8, 16, 10, 2, 0)),
	}
	ivs = cdf_data_intervals(_PATHS, bounds_resolver=lambda p: overlap[p])
	assert ivs[0].end == ivs[1].start                          # cutover at T(B)
	assert ivs[0].end == datetime(2019, 8, 16, 9, 55, 33)      # = next file's filename start
	# Last file has no successor, so it keeps its true end E(file).
	assert ivs[-1].end == datetime(2019, 8, 16, 10, 2, 0)


def test_resolver_start_falls_back_to_filename():
	# Resolver may return start=None; we keep the filename stamp then.
	ivs = cdf_data_intervals(_PATHS, bounds_resolver=lambda p: (None, datetime(2019, 8, 16, 9, 55, 53)))
	assert ivs[0].start == datetime(2019, 8, 16, 9, 53, 23)


def test_clip_trims_leading_and_trailing_to_window():
	# Anchor file starts well before the request; last file ends after it.
	bounds = {
		_PATHS[0]: (datetime(2019, 8, 16, 9, 53, 23), datetime(2019, 8, 16, 9, 56, 0)),
		_PATHS[1]: (datetime(2019, 8, 16, 9, 55, 33), datetime(2019, 8, 16, 9, 57, 33)),
		_PATHS[2]: (datetime(2019, 8, 16, 10, 0, 0), datetime(2019, 8, 16, 10, 5, 0)),
	}
	st = datetime(2019, 8, 16, 9, 54, 0)   # inside file 0
	ts = datetime(2019, 8, 16, 10, 2, 0)   # inside file 2
	ivs = cdf_data_intervals(_PATHS, bounds_resolver=lambda p: bounds[p],
	                         clip_start=st, clip_end=ts)
	assert ivs[0].start == st               # no leading recorded-but-unused span
	assert ivs[-1].end == ts                # no trailing span past the request
	# Every interval lies strictly within [st, ts].
	assert all(st <= iv.start < iv.end <= ts for iv in ivs)


def test_no_interval_covers_a_no_data_gap():
	# File 0 has a real gap before file 1 (E(0) < T(1)); the interval must end at
	# the true data end E(0), never stretching across the empty gap to T(1).
	bounds = {
		_PATHS[0]: (datetime(2019, 8, 16, 9, 53, 23), datetime(2019, 8, 16, 9, 54, 0)),  # ends early -> gap
		_PATHS[1]: (datetime(2019, 8, 16, 9, 55, 33), datetime(2019, 8, 16, 9, 57, 33)),
		_PATHS[2]: (datetime(2019, 8, 16, 10, 0, 0), datetime(2019, 8, 16, 10, 2, 0)),
	}
	ivs = cdf_data_intervals(_PATHS, bounds_resolver=lambda p: bounds[p])
	assert ivs[0].end == datetime(2019, 8, 16, 9, 54, 0)   # true E(0), not T(1)
	assert ivs[0].end < ivs[1].start                       # gap sits BETWEEN intervals


def test_interval_outside_window_is_dropped():
	bounds = {
		_PATHS[0]: (datetime(2019, 8, 16, 9, 53, 23), datetime(2019, 8, 16, 9, 54, 0)),
		_PATHS[1]: (datetime(2019, 8, 16, 9, 55, 33), datetime(2019, 8, 16, 9, 57, 33)),
		_PATHS[2]: (datetime(2019, 8, 16, 10, 0, 0), datetime(2019, 8, 16, 10, 2, 0)),
	}
	# Window covers only the last file.
	ivs = cdf_data_intervals(_PATHS, bounds_resolver=lambda p: bounds[p],
	                         clip_start=datetime(2019, 8, 16, 10, 0, 0),
	                         clip_end=datetime(2019, 8, 16, 10, 2, 0))
	assert len(ivs) == 1
	assert ivs[0].path == _PATHS[2]


_SAMPLE = timedelta(seconds=1 / 8192)        # one sample period @ SCB
_TOL = timedelta(seconds=2 / 8192)           # merge within ~2 samples


def test_merge_joins_sample_adjacent_files_into_one_burst():
	# Three files that tile with a one-sample seam = a single continuous burst.
	t0 = datetime(2019, 8, 16, 9, 0, 0)
	ivs = [
		DataInterval(t0, t0 + timedelta(seconds=130), _PATHS[0]),
		DataInterval(t0 + timedelta(seconds=130) + _SAMPLE, t0 + timedelta(seconds=260), _PATHS[1]),
		DataInterval(t0 + timedelta(seconds=260) + _SAMPLE, t0 + timedelta(seconds=390), _PATHS[2]),
	]
	bursts = merge_bursts(ivs, tol=_TOL)
	assert len(bursts) == 1
	assert bursts[0].start == ivs[0].start
	assert bursts[0].end == ivs[-1].end
	assert bursts[0].n_files == len(_PATHS)   # counts every source file


def test_merge_splits_on_real_gap():
	t0 = datetime(2019, 8, 16, 9, 0, 0)
	ivs = [
		DataInterval(t0, t0 + timedelta(seconds=130), _PATHS[0]),
		# real inter-burst gap (minutes) -> separate burst
		DataInterval(t0 + timedelta(seconds=1800), t0 + timedelta(seconds=1930), _PATHS[1]),
	]
	bursts = merge_bursts(ivs, tol=_TOL)
	assert len(bursts) == 2
	assert bursts[0].n_files == 1
	assert bursts[1].n_files == 1


def test_merge_joins_overlapping_files():
	t0 = datetime(2019, 8, 16, 9, 0, 0)
	ivs = [
		DataInterval(t0, t0 + timedelta(seconds=130), _PATHS[0]),
		DataInterval(t0 + timedelta(seconds=100), t0 + timedelta(seconds=230), _PATHS[1]),  # overlaps
	]
	bursts = merge_bursts(ivs, tol=timedelta(0))   # overlap merges even at tol=0
	assert len(bursts) == 1
	assert bursts[0].end == ivs[1].end


def test_merge_does_not_join_zero_overlap_beyond_tol():
	# A sub-second gap larger than the sample tolerance is NOT a continuous burst.
	t0 = datetime(2019, 8, 16, 9, 0, 0)
	ivs = [
		DataInterval(t0, t0 + timedelta(seconds=130), _PATHS[0]),
		DataInterval(t0 + timedelta(seconds=130) + timedelta(milliseconds=50), t0 + timedelta(seconds=260), _PATHS[1]),
	]
	bursts = merge_bursts(ivs, tol=_TOL)
	assert len(bursts) == 2


def test_burst_manifest_written_one_row_per_burst():
	t0 = datetime(2019, 8, 16, 9, 0, 0)
	bursts = [Burst(t0, t0 + timedelta(seconds=130), 2),
	          Burst(t0 + timedelta(seconds=1800), t0 + timedelta(seconds=1930), 1)]
	with tempfile.TemporaryDirectory() as d:
		out = write_burst_manifest(os.path.join(d, 'nested', 'b.txt'), bursts)
		assert os.path.exists(out)               # nested dir auto-created
		lines = open(out).read().splitlines()
	assert lines[0].startswith('#')              # units/threshold line
	assert len(lines) == 3 + len(bursts)         # units, table and column lines, one row per burst
	# Verify the written fields: start end event good n_files -- no paths
	# column. 'event' sits alongside end rather than replacing it (a reader
	# needs end to see the gap to the next row, and event to see how long this
	# one ran, without arithmetic for either); 'good' defaults to the full
	# span when the burst carries no tracked value. With no recorded table
	# given, the used table is the only one.
	assert lines[0].startswith('# ') and 'seconds' in lines[0]   # units line
	assert lines[1].split('\t')[:2] == ['# table', 'used']
	assert lines[2].split('\t') == ['# start', 'end', 'event', 'good', 'n_files']
	first = lines[3].split('\t')
	assert len(first) == 5, first                # paths column is gone for good
	assert first[0] == bursts[0].start.isoformat(sep=' ')
	assert first[1] == bursts[0].end.isoformat(sep=' ')
	assert first[2] == f'{(bursts[0].end - bursts[0].start).total_seconds():.3f}'
	assert first[3] == first[2]                  # untracked -> falls back to span
	assert first[4] == str(bursts[0].n_files)


def test_readable_manifest_start_and_duration_per_burst():
	t0 = datetime(2019, 8, 16, 9, 0, 0)
	bursts = [Burst(t0, t0 + timedelta(seconds=130), 2),
	          Burst(t0 + timedelta(seconds=1800), t0 + timedelta(seconds=1930, milliseconds=500), 1)]
	with tempfile.TemporaryDirectory() as d:
		out = write_burst_manifest_readable(os.path.join(d, 'nested', 'r.txt'), bursts)
		assert os.path.exists(out)                   # nested dir auto-created
		lines = open(out).read().splitlines()
	assert lines[0].startswith('#')                  # header
	assert len(lines) == 1 + len(bursts)             # one row per burst
	# Row = start<TAB>duration<TAB>duration_seconds<TAB>n_files; start is the
	# burst's start, duration is end - start (not the end time), seconds matches.
	first = lines[1].split('\t')
	assert first[0] == bursts[0].start.isoformat(sep=' ')
	assert first[1] == str(bursts[0].end - bursts[0].start)   # '0:02:10'
	assert first[2] == f'{(bursts[0].end - bursts[0].start).total_seconds():.3f}'
	assert first[3] == str(bursts[0].n_files)
	# Fractional-second duration survives to the seconds column.
	assert lines[2].split('\t')[2] == '130.500'


def test_readable_manifest_handles_zero_file_burst():
	# A burst reporting no source files must still write a complete row.
	t0 = datetime(2019, 8, 16, 9, 0, 0)
	bursts = [Burst(t0, t0 + timedelta(seconds=42), 0)]
	with tempfile.TemporaryDirectory() as d:
		out = write_burst_manifest_readable(os.path.join(d, 'r.txt'), bursts)
		row = open(out).read().splitlines()[1].split('\t')
	assert row[0] == t0.isoformat(sep=' ')
	assert row[2] == '42.000'


def test_machine_manifest_streamed_chunk_counts_roundtrip():
	# Streamed good-data chunks carry their cluster's own source-file count --
	# the whole point of the manifest's n_files column (they used to write 0).
	t0 = datetime(2019, 8, 16, 9, 0, 0)
	bursts = [Burst(t0, t0 + timedelta(seconds=42), 4),
	          Burst(t0 + timedelta(seconds=100), t0 + timedelta(seconds=230), 7)]
	with tempfile.TemporaryDirectory() as d:
		machine = write_burst_manifest(os.path.join(d, 'm.txt'), bursts)
		row = open(machine).read().splitlines()[3].split('\t')
		assert row[4] == '4'                         # real count, not 0
		# Round-trip: written 'recorded' comes back populated, so compare the
		# fields the writer actually carries rather than the bare tuples.
		back = read_burst_manifest(machine)
		assert [(b.start, b.end, b.n_files) for b in back] == \
		       [(b.start, b.end, b.n_files) for b in bursts]
		assert all(b.recorded == b.end - b.start for b in back)


def test_read_burst_manifest_parses_legacy_three_column_rows():
	# v1/v2 files (start/end/n_files, ISO 'T') predate the duration column and
	# still sit on disk next to v3 ones. read_burst_manifest discriminates on
	# column count, so both shapes parse and n_files never silently reads 0.
	t0 = datetime(2019, 8, 16, 9, 0, 0)
	bursts = [Burst(t0, t0 + timedelta(seconds=130), 2),
	          Burst(t0 + timedelta(seconds=1800), t0 + timedelta(seconds=1930), 5)]
	with tempfile.TemporaryDirectory() as d:
		legacy = os.path.join(d, 'v2.txt')
		with open(legacy, 'w') as f:
			f.write('# start\tend\tn_files\n')
			for b in bursts:
				f.write(f'{b.start.isoformat()}\t{b.end.isoformat()}\t{b.n_files}\n')
		assert read_burst_manifest(legacy) == bursts


def test_read_burst_manifest_roundtrips_machine_format():
	t0 = datetime(2019, 8, 16, 9, 0, 0)
	bursts = [Burst(t0, t0 + timedelta(seconds=130), 2),
	          Burst(t0 + timedelta(seconds=1800), t0 + timedelta(seconds=1930), 1)]
	with tempfile.TemporaryDirectory() as d:
		machine = write_burst_manifest(os.path.join(d, 'm.txt'), bursts)
		parsed = read_burst_manifest(machine)
	# Exact round-trip incl. counts; 'recorded' is filled in from the file, so
	# normalise the originals through the same fallback the writer used.
	assert parsed == [b._replace(recorded=b.recorded_or_span) for b in bursts]


def test_overlapping_seam_keeps_both_files_good_data():
	# The 2017-03-15 13:11 EDP seam, in shape (fixed 2026-11-07): the older file's last good segment starts
	# after the newer file's first one begins, and ends with its file. Segments are joined, never cut at the
	# next row's start, so the newer file's good data past the older file's end stays in the one event.
	t0 = datetime(2017, 3, 15, 13, 11, 0)
	s = lambda a, b: (t0 + timedelta(seconds=a), t0 + timedelta(seconds=b))
	segments = {'old.cdf': [s(0, 22.1), s(22.12, 23.0)],    # its file ends at 23.0
	            'new.cdf': [s(22.11, 26.9), s(27.0, 40)]}
	ivs = cdf_data_intervals(['old.cdf', 'new.cdf'], segments_resolver=segments.get)
	bursts = merge_bursts(ivs, tol=timedelta(seconds=1.5))
	assert [(b.start, b.end, b.n_files) for b in bursts] == [(t0, t0 + timedelta(seconds=40), 2)]
	# good time is the union: only the 0.01 s hole at 22.10-22.11 and the 0.1 s one at 26.9-27.0 are out
	assert bursts[0].recorded == timedelta(seconds=40 - 0.01 - 0.1)
	covered = lambda t: any(iv.start <= t <= iv.end for iv in ivs)
	assert covered(t0 + timedelta(seconds=25))                # the span the old cut lost


def test_overlapping_seam_short_run_rule_counts_once():
	# Two files holding the same 6 samples are one run of 6: dropped, as before the fix.
	t0 = datetime(2017, 3, 15, 13, 11, 0)
	run = (t0, t0 + timedelta(microseconds=5 * 122))
	ivs = cdf_data_intervals(['a.cdf', 'b.cdf'], segments_resolver={'a.cdf': [run], 'b.cdf': [run]}.get,
	                         min_run=timedelta(microseconds=9 * 122), run_gap=timedelta(microseconds=183))
	assert ivs == []


def _run():
	tests =[v for k, v in sorted(globals().items()) if k.startswith('test_')]
	for t in tests:
		t()
		print(f'  ok  {t.__name__}')
	print(f'{len(tests)} passed')


if __name__ == '__main__':
	_run()
