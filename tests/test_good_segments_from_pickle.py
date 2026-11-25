"""
Equivalence test for ffttest._segments_from_pickle.

TRUST_GOOD_PICKLES=True replaces the CDF quality read (_scm_good_segments /
_edp_good_segments) with boundaries derived from the good-only data pickle's own
time column. That substitution is only safe if the derived runs are IDENTICAL to
the runs the quality flag would have produced. This test proves it the same way
the real pipeline works: build a full-length sample series plus a quality mask,
compute the reference segments the way the CDF finders do (contiguous runs of
the mask), then drop the bad samples exactly as _cdf_to_df does, pickle the
survivors, and check _segments_from_pickle recovers the reference runs.

The bad samples are DROPPED, never stored as gaps or zero-fill, so a run
boundary is simply a timestamp spacing wider than one sample period at the
instrument's specified rate.

Covers both instruments' rates (they are the same 8192 Hz, passed separately so
the parameter is proven to be honoured rather than inherited), a leading
convolution-edge cut, interior bad stretches down to a single sample, trailing
cuts, all-good files, a long clean run that would shatter under a zero-margin
threshold, all-bad files, and single-sample files.

Run (needs numpy/pandas -- the `plasma` conda env):

    python tests/test_good_segments_from_pickle.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

import ffttest
from ffttest import _segments_from_pickle, SAMPLE_RATE


def _reference_segments(time, good):
	"""Contiguous runs of `good`, as (start, end) Timestamps -- the exact
	run-boundary logic _scm_good_segments/_edp_good_segments use (ffttest.py:
	padded diff, run_starts/run_ends), applied to the FULL series."""
	if not good.any():
		return []
	padded = np.concatenate(([False], good, [False]))
	diffs = np.diff(padded.astype(np.int8))
	run_starts = np.flatnonzero(diffs == 1)
	run_ends = np.flatnonzero(diffs == -1) - 1
	return [(time[s], time[e]) for s, e in zip(run_starts, run_ends)]


def _make_series(n, fs, t0='2021-02-23T11:07:33'):
	"""Full-length timestamp series at `fs` Hz, as _cdf_to_df would see it."""
	step_ns = int(round(1e9 / fs))
	base = pd.Timestamp(t0).value
	return pd.to_datetime(np.arange(n, dtype=np.int64) * step_ns + base)


def _roundtrip(tmpdir, name, time, good, fs):
	"""Drop bad samples (as _cdf_to_df does), pickle, derive segments back.

	Note what is NOT written: the dropped samples leave no row, no zero-fill,
	no gap marker of any kind. The pickle holds only surviving samples, which
	is precisely why the runs are recoverable from their timestamps alone.
	"""
	path = os.path.join(tmpdir, f'{name}.pkl')
	kept = time[good]
	df = pd.DataFrame({
		'time': kept,
		'Bx': np.zeros(len(kept)),
		'By': np.zeros(len(kept)),
		'Bz': np.zeros(len(kept)),
	})
	df.to_pickle(path)
	return _segments_from_pickle(path, fs)


def _check(label, tmpdir, n, fs, bad_slices):
	time = _make_series(n, fs)
	good = np.ones(n, dtype=bool)
	for sl in bad_slices:
		good[sl] = False

	expected = _reference_segments(time, good)
	actual = _roundtrip(tmpdir, label.replace(' ', '_'), time, good, fs)

	assert len(actual) == len(expected), (
		f'{label}: got {len(actual)} segments, expected {len(expected)}\n'
		f'  expected={expected}\n  actual={actual}'
	)
	for i, (exp, act) in enumerate(zip(expected, actual)):
		assert act[0] == exp[0], f'{label}: segment {i} start {act[0]} != {exp[0]}'
		assert act[1] == exp[1], f'{label}: segment {i} end {act[1]} != {exp[1]}'
	print(f'  ok  {label}: {len(actual)} segment(s)')


def main():
	print(f'ffttest.SAMPLE_RATE = {SAMPLE_RATE}')
	with tempfile.TemporaryDirectory() as tmpdir:
		# SCM/B at the live SAMPLE_RATE, not a literal: if that constant ever moves
		# again, the threshold arithmetic moves with it and these runs re-prove
		# that segment boundaries still land where they should.
		print('SCM/B cadence (SAMPLE_RATE):')
		_check('all good', tmpdir, 5000, SAMPLE_RATE, [])
		_check('leading conv edge', tmpdir, 5000, SAMPLE_RATE, [slice(0, 1600)])
		_check('trailing cut', tmpdir, 5000, SAMPLE_RATE, [slice(4200, 5000)])
		_check('one interior stretch', tmpdir, 5000, SAMPLE_RATE, [slice(2000, 2500)])
		_check('single bad sample', tmpdir, 5000, SAMPLE_RATE, [slice(2500, 2501)])
		_check('three stretches', tmpdir, 20000, SAMPLE_RATE,
		       [slice(0, 1200), slice(5000, 5003), slice(11000, 14000)])
		_check('adjacent singles', tmpdir, 3000, SAMPLE_RATE,
		       [slice(1000, 1001), slice(1002, 1003), slice(1004, 1005)])

		# EDP/E cadence -- passed as its own literal, so the code path is proven
		# to use the rate it is handed rather than reaching for B's. EDP burst
		# happens to run at that same 8192 Hz (NASA's cadence metadata and the
		# files themselves agree -- see SAMPLE_RATE in ffttest.py).
		print('EDP/E cadence (rate passed explicitly):')
		_check('edp all good', tmpdir, 4096, 8192.0, [])
		_check('edp two stretches', tmpdir, 8192, 8192.0,
		       [slice(0, 500), slice(4000, 4400)])
		_check('edp single bad', tmpdir, 4096, 8192.0, [slice(2048, 2049)])

		# A long all-good file must stay ONE segment. The timestamps step by
		# int(1e9 / fs) ns -- 122070 at 8192 Hz, against a true period of
		# 122070.3125 ns -- so every spacing sits a fraction under 1/fs and that
		# shortfall accumulates across the file. The threshold has to clear the
		# rounding while still catching a single dropped sample (2/fs); this is
		# the case that proves the margin does both.
		print('Threshold margin (rounded sample period):')
		_check('long clean run', tmpdir, 200000, SAMPLE_RATE, [])

		# Degenerate files.
		print('Degenerate:')
		_check('all bad', tmpdir, 1000, SAMPLE_RATE, [slice(0, 1000)])

		single = _make_series(1, SAMPLE_RATE)
		segs = _roundtrip(tmpdir, 'single_sample', single, np.ones(1, dtype=bool), SAMPLE_RATE)
		assert len(segs) == 1 and segs[0][0] == segs[0][1] == single[0], segs
		print(f'  ok  single sample: {len(segs)} segment(s)')

		missing = _segments_from_pickle(os.path.join(tmpdir, 'does_not_exist.pkl'), SAMPLE_RATE)
		assert missing == [], missing
		print('  ok  missing pickle -> [] (fail-open, matches CDF finders)')

		empty_path = os.path.join(tmpdir, 'empty.pkl')
		pd.DataFrame({'time': pd.to_datetime([]), 'Bx': []}).to_pickle(empty_path)
		assert _segments_from_pickle(empty_path, SAMPLE_RATE) == []
		print('  ok  empty pickle -> []')

	print('\nAll good-segment derivation tests passed.')


if __name__ == '__main__':
	main()
