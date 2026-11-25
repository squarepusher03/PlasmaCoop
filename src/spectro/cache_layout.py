"""
Single owner of the on-disk path schema for cached pickles and FFT results.

Writers (data_cache, fft_cache) and readers (file_index) both route through
this module so the schema is defined in exactly one place. parse_pickle_ts
is the inverse of pickle_path; round-trip equality is the verified invariant.

No filesystem I/O happens here — only path strings and timestamp parsing.
"""
from __future__ import annotations

import os
import re
from datetime import datetime

from .instrument_spec import InstrumentSpec


# ---------------------------------------------------------------------------
# CDF source files (mirrors ffttest.find_cdf_files glob + regex)
# ---------------------------------------------------------------------------

def cdf_glob_pattern(data_dir: str, spec: InstrumentSpec, probe: int) -> str:
	"""Recursive glob pattern for CDF files of one instrument under data_dir."""
	return os.path.join(
		data_dir,
		f'mms{probe}/{spec.inst}/brst/l2/{spec.mode}/**/*.cdf',
	)


_CDF_TS_RE = re.compile(r'_(\d{14})_')


def parse_cdf_ts(cdf_path: str) -> datetime:
	"""Extract the 14-digit timestamp embedded in a CDF filename."""
	m = _CDF_TS_RE.search(cdf_path)
	if not m:
		raise ValueError(f'No timestamp in CDF path: {cdf_path}')
	return datetime.strptime(m.group(1), '%Y%m%d%H%M%S')


# ---------------------------------------------------------------------------
# Per-CDF data pickles (mirrors ffttest.ensure_data_pickle write +
# ffttest.find_data_pickles regex)
# ---------------------------------------------------------------------------

def pickle_path(pickle_root: str, spec: InstrumentSpec, probe: int, file_ts: datetime) -> str:
	"""Where the per-CDF DataFrame pickle lives on disk."""
	return os.path.join(
		pickle_root,
		'mms', str(probe), spec.inst, spec.mode,
		file_ts.strftime('%Y'),
		file_ts.strftime('%m'),
		file_ts.strftime('%d'),
		file_ts.strftime('%H%M%S') + '.pkl',
	)


def pickle_glob_pattern(pickle_root: str, spec: InstrumentSpec, probe: int) -> str:
	"""Glob covering every per-CDF data pickle for one (probe, instrument)."""
	return os.path.join(
		pickle_root,
		f'mms/{probe}/{spec.inst}/{spec.mode}/*/*/*/*.pkl',
	)


_PICKLE_TS_RE = re.compile(r'(\d{4})[/\\](\d{2})[/\\](\d{2})[/\\](\d{6})\.pkl$')


def parse_pickle_ts(path: str) -> datetime:
	"""Inverse of pickle_path. Round-trips: parse_pickle_ts(pickle_path(.., ts)) == ts."""
	m = _PICKLE_TS_RE.search(path)
	if not m:
		raise ValueError(f'Not a pickle path: {path}')
	return datetime.strptime(''.join(m.groups()), '%Y%m%d%H%M%S')


# ---------------------------------------------------------------------------
# Per-CDF FFT result pickles (mirrors ffttest._fft_pickle_path)
# Pad lives in the directory so different frequency resolutions don't collide.
# ---------------------------------------------------------------------------

def fft_path(pickle_root: str, spec: InstrumentSpec, probe: int, pad: float, file_ts: datetime,
             lpf_lim: float | None = None) -> str:
	"""Where the per-CDF FFT-result pickle lives. pad picks the subdirectory.

	lpf_lim joins pad in the subdirectory when given, mirroring
	ffttest._fft_pickle_path: it selects which rows the pickle CONTAINS
	(frequency <= lpf_lim), so two ceilings must not share a cache entry.
	Omitting it reproduces the pre-fix pad-only path.
	"""
	sub = f'pad{pad:.3f}' if lpf_lim is None else f'pad{pad:.3f}_lpf{lpf_lim:g}'
	return os.path.join(
		pickle_root,
		'mms', str(probe), spec.inst, spec.mode, 'fft', sub,
		file_ts.strftime('%Y'),
		file_ts.strftime('%m'),
		file_ts.strftime('%d'),
		file_ts.strftime('%H%M%S') + '.pkl',
	)


def fft_glob_pattern(pickle_root: str, spec: InstrumentSpec, probe: int, pad: float) -> str:
	"""Glob covering every per-CDF FFT pickle at one pad value."""
	return os.path.join(
		pickle_root,
		f'mms/{probe}/{spec.inst}/{spec.mode}/fft/pad{pad:.3f}/*/*/*/*.pkl',
	)
