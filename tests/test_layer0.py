"""
Round-trip and parity tests for spectro Layer 0 modules.

Runs as a plain script — no pytest dependency:

    python tests/test_layer0.py

Each test_* function asserts one independently-verifiable thing.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np

# Make `import spectro...` work whether the script is run from project root
# or from inside tests/. We prepend the project's src/ to sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

from spectro.instrument_spec import SPECS, BurstRate, SAMPLE_RATE, get_spec, InstrumentSpec
from spectro.cache_layout import (
	cdf_glob_pattern, parse_cdf_ts,
	pickle_path, pickle_glob_pattern, parse_pickle_ts,
	fft_path, fft_glob_pattern,
)
from spectro.freq_grid import df, n_window, make_grid, snap


# ---------------------------------------------------------------------------
# instrument_spec
# ---------------------------------------------------------------------------

def test_specs_have_expected_dispatch():
	assert SPECS['B'].inst == 'scm' and SPECS['B'].mode == 'scb'
	assert SPECS['E'].inst == 'edp' and SPECS['E'].mode == 'dce'
	assert SPECS['B'].fs == SAMPLE_RATE
	assert SPECS['B'].field == 'B'
	assert SPECS['E'].field == 'E'


def test_get_spec_resolves_short_and_long_keys():
	assert get_spec('B') is SPECS['B']
	assert get_spec('Bz') is SPECS['B']
	assert get_spec('E') is SPECS['E']
	assert get_spec('Ex') is SPECS['E']


def test_instrument_spec_is_frozen():
	# dataclass(frozen=True) — must raise on attribute assignment
	spec = SPECS['B']
	try:
		spec.fs = 1
	except Exception as e:
		assert 'frozen' in str(e).lower() or 'cannot assign' in str(e).lower()
	else:
		raise AssertionError('InstrumentSpec must be frozen')


def test_var_template_formats_with_probe():
	# ffttest._cdf_to_df uses these literal strings — must match
	assert SPECS['B'].var_template.format(probe=1) == 'mms1_scm_acb_gse_scb_brst_l2'
	assert SPECS['E'].var_template.format(probe=2) == 'mms2_edp_dce_gse_brst_l2'
	assert SPECS['B'].epoch_template.format(probe=3) == 'Epoch'
	assert SPECS['E'].epoch_template.format(probe=4) == 'mms4_edp_epoch_brst_l2'


# ---------------------------------------------------------------------------
# cache_layout — CDF source
# ---------------------------------------------------------------------------

def test_cdf_glob_matches_ffttest_pattern():
	# ffttest.find_cdf_files L38: f'mms{probe}/{inst}/brst/l2/{mode}/**/*.cdf'
	pat = cdf_glob_pattern('./pydata', SPECS['B'], probe=1).replace('\\', '/')
	assert pat == './pydata/mms1/scm/brst/l2/scb/**/*.cdf'

	pat_e = cdf_glob_pattern('E:/PlasmaCoop/pydata', SPECS['E'], probe=2).replace('\\', '/')
	assert pat_e.endswith('mms2/edp/brst/l2/dce/**/*.cdf')


def test_parse_cdf_ts_matches_ffttest_regex():
	# ffttest.find_cdf_files L41 regex: r'_(\d{14})_'
	path = 'mms1_scm_brst_l2_scb_20190816093103_v2.2.1.cdf'
	assert parse_cdf_ts(path) == datetime(2019, 8, 16, 9, 31, 3)


def test_parse_cdf_ts_raises_on_missing_timestamp():
	try:
		parse_cdf_ts('not_a_cdf_filename.cdf')
	except ValueError:
		pass
	else:
		raise AssertionError('parse_cdf_ts must raise ValueError')


# ---------------------------------------------------------------------------
# cache_layout — data pickle round-trip (the core invariant)
# ---------------------------------------------------------------------------

def test_pickle_path_format_matches_ffttest():
	# ffttest.ensure_data_pickle L121-122 writes
	# {pickle_root}/mms/{probe}/{inst}/{mode}/{YYYY}/{MM}/{DD}/{HHMMSS}.pkl
	ts = datetime(2019, 8, 16, 9, 31, 3)
	p = pickle_path('/cache', SPECS['B'], probe=1, file_ts=ts).replace('\\', '/')
	assert p == '/cache/mms/1/scm/scb/2019/08/16/093103.pkl'


def test_pickle_path_roundtrips():
	# The invariant: writer and reader agree on schema.
	for ts in [
		datetime(2019, 8, 16, 9, 31, 3),
		datetime(2020, 1, 1, 0, 0, 0),
		datetime(2026, 12, 31, 23, 59, 59),
	]:
		for probe in (1, 2, 3, 4):
			for spec in (SPECS['B'], SPECS['E']):
				p = pickle_path('/cache', spec, probe=probe, file_ts=ts)
				assert parse_pickle_ts(p) == ts


def test_pickle_glob_matches_ffttest_pattern():
	# ffttest.find_data_pickles L65: f'mms/{probe}/{inst}/{mode}/*/*/*/*.pkl'
	pat = pickle_glob_pattern('/cache', SPECS['B'], probe=1).replace('\\', '/')
	assert pat == '/cache/mms/1/scm/scb/*/*/*/*.pkl'


def test_parse_pickle_ts_rejects_non_pickle():
	try:
		parse_pickle_ts('/cache/mms/1/scm/scb/something_else.txt')
	except ValueError:
		pass
	else:
		raise AssertionError('parse_pickle_ts must raise ValueError on non-pickle paths')


# ---------------------------------------------------------------------------
# cache_layout — FFT pickle path
# ---------------------------------------------------------------------------

def test_fft_path_includes_pad_dir():
	# ffttest._fft_pickle_path L153-156: pad{pad:.3f} subdir between {mode} and {YYYY}
	ts = datetime(2019, 8, 16, 9, 31, 3)
	p = fft_path('/cache', SPECS['B'], probe=1, pad=0.1, file_ts=ts).replace('\\', '/')
	assert p == '/cache/mms/1/scm/scb/fft/pad0.100/2019/08/16/093103.pkl'


def test_fft_path_distinct_per_pad():
	# Two pad values must land in different directories so they don't collide.
	ts = datetime(2019, 8, 16, 9, 31, 3)
	p1 = fft_path('/cache', SPECS['B'], probe=1, pad=0.1, file_ts=ts)
	p2 = fft_path('/cache', SPECS['B'], probe=1, pad=0.5, file_ts=ts)
	assert p1 != p2
	assert 'pad0.100' in p1.replace('\\', '/')
	assert 'pad0.500' in p2.replace('\\', '/')


def test_fft_glob_matches_pattern():
	pat = fft_glob_pattern('/cache', SPECS['E'], probe=2, pad=0.25).replace('\\', '/')
	assert pat == '/cache/mms/2/edp/dce/fft/pad0.250/*/*/*/*.pkl'


# ---------------------------------------------------------------------------
# freq_grid — exact numeric parity with ffttest._compute_file_ffts inline math
# ---------------------------------------------------------------------------

def test_df_matches_inline_formula():
	# ffttest.gen_fft L358: Df = 1 / (pad * 2)
	for pad in (0.1, 0.25, 0.5, 1.0):
		assert df(pad) == 1.0 / (pad * 2)


def test_n_window_matches_inline_formula():
	# ffttest.gen_fft L357: n_expected = int(fs * (pad * 2))
	for pad in (0.1, 0.25, 0.5):
		for fs in (BurstRate.SCB, BurstRate.SCHB):
			assert n_window(pad, fs) == int(fs * (pad * 2))


def test_make_grid_matches_ffttest_inline():
	# ffttest._compute_file_ffts L197:
	# freq_grid = (np.round(np.arange(0, lpf_lim / Df + 1)) * Df).astype(int)
	for pad, lpf in [(0.1, 1000), (0.25, 1000), (0.5, 500)]:
		d_ref    = 1.0 / (pad * 2)
		expected = (np.round(np.arange(0, lpf / d_ref + 1)) * d_ref).astype(np.int32)
		got      = make_grid(pad, lpf)
		assert np.array_equal(got, expected), f'mismatch at pad={pad}, lpf={lpf}'


def test_snap_matches_ffttest_inline():
	# ffttest._compute_file_ffts L272-273:
	# signal_fft_freq = np.fft.rfftfreq(n, d=1/fs)
	# signal_fft_freq = (np.round(signal_fft_freq / Df) * Df).astype(int)
	for pad in (0.1, 0.25):
		fs       = SAMPLE_RATE
		n        = n_window(pad, fs)
		d_ref    = 1.0 / (pad * 2)
		freqs    = np.fft.rfftfreq(n, d=1.0 / fs)
		expected = (np.round(freqs / d_ref) * d_ref).astype(np.int32)
		got      = snap(freqs, pad)
		assert np.array_equal(got, expected), f'mismatch at pad={pad}'


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
	tests = sorted(
		((name, fn) for name, fn in globals().items()
		 if name.startswith('test_') and callable(fn)),
		key=lambda kv: kv[0],
	)
	failed = 0
	for name, fn in tests:
		try:
			fn()
			print(f'  PASS  {name}')
		except Exception as e:
			failed += 1
			print(f'  FAIL  {name}: {type(e).__name__}: {e}')
	total = len(tests)
	print(f'\n{total - failed}/{total} passed' + (', 0 failed.' if not failed else f', {failed} FAILED.'))
	return 0 if failed == 0 else 1


if __name__ == '__main__':
	sys.exit(main())
