"""
Pure math for the FFT frequency grid.

Owns Df = 1 / (2*pad), the integer-Hz lattice up to lpf_lim, and the snapping
of rfft bins onto that lattice. No I/O, no DataFrames — these primitives
are consumed by fft_kernel and downstream.

Matches ffttest._compute_file_ffts L197 and L272-273 exactly.
"""
from __future__ import annotations

import numpy as np


def df(pad: float) -> float:
	"""Frequency resolution given half-window pad (s).  Df = 1 / (2*pad) Hz."""
	return 1.0 / (2.0 * pad)


def n_window(pad: float, fs: int) -> int:
	"""Expected sample count for a full 2*pad window at sample rate fs."""
	return int(fs * (pad * 2))


def make_grid(pad: float, lpf_lim: float) -> np.ndarray:
	"""Integer-Hz frequency lattice from 0 to lpf_lim in steps of Df."""
	d = df(pad)
	return (np.round(np.arange(0, lpf_lim / d + 1)) * d).astype(np.int32)


def snap(rfft_freqs: np.ndarray, pad: float) -> np.ndarray:
	"""Snap an rfftfreq array onto the integer Df grid."""
	d = df(pad)
	return (np.round(rfft_freqs / d) * d).astype(np.int32)
