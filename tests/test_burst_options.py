"""Burst modes per field: B has SCB and SCHB, E has SCB only (Graham, 2026-09-30:
"Split E's rate off but B field should have both SCB and SCHB options while E field only
has SCB. The E field SCB can be deactivated and permanently selected to indicate the
constraint").

One setting, instrument_spec.B_BURST, decides B's rate and its scb/schb names; E never
follows it. A file is always read with the names of its own mode (ffttest._scm_mode),
so an SCHB file reads correctly whatever is selected. The end-to-end run with SCHB
selected, on a real SCHB file, is _wsl_scratch/schb_option_check.py.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLBACKEND', 'Agg')

pycdf = pytest.importorskip('spacepy.pycdf')

import acquisition
import event_export as ee
import ffttest
from test_direct_cdf import T

try:
	from spectro import instrument_spec as isp
except ImportError:
	from src.spectro import instrument_spec as isp

SCB, SCHB = isp.BurstRate.SCB, isp.BurstRate.SCHB


def test_the_options_per_field():
	assert isp.FIELD_BURST_OPTIONS == {'B': (SCB, SCHB), 'E': (SCB,)}
	assert isp.E_BURST is SCB                                  # E's one option, not a setting


def test_the_default_selection_is_what_the_code_named_before():
	assert isp.B_BURST is SCB
	assert isp.field_burst('Bz') == ('scm', 'scb', 8192)
	assert isp.field_burst('Ex') == ('edp', 'dce', 8192)
	assert all(type(isp.field_burst(k)[2]) is int for k in 'BE')   # plain ints: paths and pickles unchanged
	assert (isp.SAMPLE_RATE, isp.B_SAMPLE_RATE, isp.E_SAMPLE_RATE) == (8192, 8192, 8192)
	assert acquisition.FIELDS == ee.INSTRUMENTS == {'B': ('scm', 'scb'), 'E': ('edp', 'dce'), 'F': ('fgm', '')}


def test_b_can_be_schb_and_e_cannot():
	assert isp.field_burst('B', SCHB) == ('scm', 'schb', 16384)
	with pytest.raises(ValueError, match='E has no SCHB mode'):
		isp.field_burst('E', SCHB)
	with pytest.raises(ValueError):
		isp.field_burst('F')                                  # FGM has no burst modes here


def test_the_mode_names_come_from_the_enum():
	assert (SCB.mode, SCHB.mode) == ('scb', 'schb')


def test_a_file_is_read_with_the_names_of_its_own_mode(tmp_path):
	"""An SCHB file reads with SCHB names while SCB is selected -- the variable names
	of a real SCHB file on E: (2026-09-30): mms1_scm_acb_gse_schb_brst_l2 and
	mms1_scm_qf_scm123_schb_brst_l2, 61036 ns steps."""
	folder = tmp_path / 'mms1' / 'scm' / 'brst' / 'l2' / 'schb' / '2017' / '01' / '24'
	folder.mkdir(parents=True)
	path = str(folder / 'mms1_scm_brst_l2_schb_20170124235723_v2.2.0.cdf')
	n, step_ns = 4096, 61_036
	raw = int(pycdf.lib.datetime_to_tt2000(T(0))) + np.arange(n, dtype=np.int64) * step_ns
	good = np.ones(n, dtype=bool)
	good[:100] = False                                       # a flagged lead, as real SCM files have
	pycdf.lib.set_backward(False)
	with pycdf.CDF(path, '') as cdf:
		cdf.new('Epoch', type=pycdf.const.CDF_TIME_TT2000)
		cdf.raw_var('Epoch')[:] = raw
		cdf.new('mms1_scm_acb_gse_schb_brst_l2', data=np.zeros((n, 3), dtype=np.float32), type=pycdf.const.CDF_FLOAT)
		cdf.new('mms1_scm_qf_scm123_schb_brst_l2', data=np.where(good, 'GGG', 'ZZZ'), type=pycdf.const.CDF_CHAR)

	assert ffttest._scm_mode(path) == 'schb'
	ffttest._check_cdf(path, 'B', 1)                          # raises if it looked for scb names
	assert len(ffttest._good_sample_times(path, 'B', 1)) == n - 100
	assert len(ffttest._recorded_sample_times(path, 'B', 1)) == n
	assert len(ffttest._cdf_to_df(path, 'Bz', probe=1)) == n - 100
	sidecar = ffttest._segments_sidecar_path('/cache', path, isp.field_burst('B', SCHB)[2])
	assert sidecar.replace(os.sep, '/').endswith('mms/1/scm/schb/segments/gap91552ns/2017/01/24/235723.npy')
