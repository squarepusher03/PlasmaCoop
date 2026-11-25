"""
Single source of truth for per-instrument constants.

Replaces the `if key[0] == 'B'` dispatch duplicated across
ffttest._cdf_to_df, ffttest.ensure_data_pickle, and ffttest.gen_fft.

InstrumentSpec also owns the per-instrument plotting metadata (units,
PSD units) and a NoiseFloor abstraction. The NoiseFloor Protocol is the
seam: today every spec carries a ConstantNoiseFloor placeholder; later
implementations (fitted curves, model evaluations) plug in without
changing any consumer of `spec.noise_floor.psd_at(freqs)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import numpy as np


class BurstRate(IntEnum):
	"""MMS burst sample rates, in Hz. Both SCM burst (scb) and EDP burst (dce)
	run at 8192 S/s. Two independent sources agree:
	  * the SCM CDFs say so themselves -- the FREQUENCY_FILTER global attribute
	    reads "Fsamp>8192Hz ... Fmax>4096.0000Hz (=Fsamp/2=Fnyquist)";
	  * NASA's dataset metadata gives the cadence as PT0.0001220703125S (= 1/8192)
	    for EDP burst DCE, PT0.0078125S (= 1/128) for FGM burst, and titles the
	    SCM products "8192 Sample/s" (burst) and "16384 Sample/s" (high burst).
	Measured on real files (probes 1-4, 2017-2023, no gaps), the timestamps step
	122072 ns = 8191.887 Hz for BOTH scm/scb and edp/dce: the spacecraft clock
	running ~14 ppm slow. That is 0.06 Hz at the top of the band -- far inside the
	Df = 2 Hz grid -- so every bin snaps to the same place either way.
	The old 8196.7 came from rounding the sample period to 122000 ns. It was wrong
	by 590 ppm and, worse, put the snapped Nyquist row at 4098 Hz instead of 4096.
	"""
	SCB  = 8_192     # SCM burst; EDP burst (dce) runs at this rate too
	SCHB = 16_384    # SCM high burst; measured 61036 ns steps = 16383.8 Hz

	@property
	def mode(self):
		"""The mode's name in MMS folders, file names, CDF variable names and the SDC
		datatype: 'scb' or 'schb'. Checked against a real schb file on E: (2026-09-30):
		mms1_scm_acb_gse_schb_brst_l2, mms1_scm_qf_scm123_schb_brst_l2, and its
		FREQUENCY_FILTER reads Fsamp>16384Hz."""
		return self.name.lower()


# The burst modes each field can run in (Graham, 2026-09-30: "B field should have
# both SCB and SCHB options while E field only has SCB. The E field SCB can be
# deactivated and permanently selected to indicate the constraint"). EDP has no
# high-burst mode, so E's one option is SCB, and it is not a setting.
FIELD_BURST_OPTIONS = {'B': (BurstRate.SCB, BurstRate.SCHB), 'E': (BurstRate.SCB,)}

# B's selection -- SCB or SCHB. B's rate and its 'scb'/'schb' names (folders, file
# names, CDF variables, the SDC datatype) all follow it; nothing else names either.
B_BURST = BurstRate.SCB
# E's: its only option, permanently selected.
E_BURST = FIELD_BURST_OPTIONS['E'][0]


def field_burst(key, burst=None):
	"""(inst, mode, fs) for a field key ('B', 'Bz', 'E', 'Ex', ...) at that field's
	selected burst mode, or at `burst` when given. fs is a plain int. Raises for a
	mode the field does not have -- E has no SCHB."""
	field = key[0]
	if field not in FIELD_BURST_OPTIONS:
		raise ValueError(f'no burst modes for field {key!r}; B and E have them')
	if burst is None:
		burst = B_BURST if field == 'B' else E_BURST
	if burst not in FIELD_BURST_OPTIONS[field]:
		raise ValueError(f'{field} has no {burst.name} mode; its only options: '
		                 f'{", ".join(b.name for b in FIELD_BURST_OPTIONS[field])}')
	if field == 'B':
		return 'scm', burst.mode, int(burst.value)
	return 'edp', 'dce', int(burst.value)


field_burst('B')                         # fail at import on a B_BURST that is not an option

# Each field's sample rate, a plain int, so every cache path and pickle stays
# byte-identical. E's never follows B's selection. SAMPLE_RATE keeps its old name
# for B -- the SCM rate every caller before the split meant; E code reads E_SAMPLE_RATE.
B_SAMPLE_RATE = field_burst('B')[2]
E_SAMPLE_RATE = field_burst('E')[2]
SAMPLE_RATE = B_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Noise floor — abstraction seam (see ISC-19 spirit: one dispatch point)
# ---------------------------------------------------------------------------

class NoiseFloor(Protocol):
	"""Returns the instrument's noise-floor PSD at the requested frequencies.

	Implementations may be flat constants, fitted curves, or model
	evaluations. Consumers (plotter, diagnostics) only depend on this
	single method; how the floor is computed can change freely.
	"""
	def psd_at(self, freqs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ConstantNoiseFloor:
	"""Placeholder NoiseFloor: same PSD value at every frequency.

	Use NaN to mark "real floor not yet known" — the plotter can branch on
	np.isnan(...) to skip the overlay rather than draw a meaningless line.
	"""
	value: float

	def psd_at(self, freqs: np.ndarray) -> np.ndarray:
		return np.full(freqs.shape, self.value, dtype=np.float64)


# ---------------------------------------------------------------------------
# Instrument descriptor — pure data, no behavior
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentSpec:
	field:          str          # 'B' (magnetic, SCM) or 'E' (electric, EDP)
	inst:           str          # 'scm' / 'edp'
	mode:           str          # 'scb' / 'dce'
	fs:             int          # sample rate (Hz)
	var_template:   str          # CDF variable name; .format(probe=...)
	epoch_template: str          # CDF epoch column;  .format(probe=...)
	units:          str          # signal units: 'nT' for B, 'mV/m' for E
	psd_units:      str          # PSD units / colorbar label
	noise_floor:    NoiseFloor   # see NoiseFloor Protocol above


# EDP burst really does run at the SCB rate -- confirmed 2026-09-11, not just
# inherited: NASA's dataset metadata gives EDP burst DCE a cadence of
# PT0.0001220703125S (= 1/8192), and the timestamps in the files themselves step
# 122072 ns for edp/dce and scm/scb alike. The old `TODO: confirm EDP sample
# rate` in ffttest.gen_fft is answered and gone.
SPECS: dict[str, InstrumentSpec] = {
	'B': InstrumentSpec(
		field='B', inst='scm', mode=B_BURST.mode, fs=B_SAMPLE_RATE,
		var_template='mms{probe}_scm_acb_gse_' + B_BURST.mode + '_brst_l2',
		epoch_template='Epoch',
		units='nT',
		psd_units='nT^2/Hz',
		noise_floor=ConstantNoiseFloor(value=float('nan')),
	),
	'E': InstrumentSpec(
		field='E', inst='edp', mode='dce', fs=E_SAMPLE_RATE,
		var_template='mms{probe}_edp_dce_gse_brst_l2',
		epoch_template='mms{probe}_edp_epoch_brst_l2',
		units='mV/m',
		psd_units='(mV/m)^2/Hz',
		noise_floor=ConstantNoiseFloor(value=float('nan')),
	),
}


def get_spec(key: str) -> InstrumentSpec:
	"""Resolve a key like 'B', 'Bz', 'E', 'Ex' to its InstrumentSpec."""
	return SPECS[key[0]]
