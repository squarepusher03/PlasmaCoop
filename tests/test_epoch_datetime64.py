"""ffttest._epoch_datetime64 must give exactly what pd.to_datetime(cdf[var][:])
gave -- pycdf's datetime objects, rounded to the microsecond -- without
building those objects. Checked against pycdf's own converter on TT2000 values
spread over the mission, including sub-microsecond remainders right at the
rounding boundary, and through the fallback paths."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pycdf = pytest.importorskip('spacepy.pycdf')

from ffttest import _epoch_datetime64


class _FakeVar:
	"""What _epoch_datetime64 touches on a pycdf Var: slicing gives datetimes."""
	def __init__(self, raw):
		self._raw = raw

	def __getitem__(self, sl):
		return pycdf.lib.v_tt2000_to_datetime(self._raw[sl])

	def __len__(self):
		return len(self._raw)


class _FakeCDF:
	def __init__(self, raw):
		self._raw = np.asarray(raw, dtype=np.int64)

	def raw_var(self, name):
		return self._raw

	def __getitem__(self, name):
		return _FakeVar(self._raw)


def _slow(raw):
	return pd.to_datetime(pycdf.lib.v_tt2000_to_datetime(np.asarray(raw, dtype=np.int64)))


def _tt2000(year, month, day):
	return int(pycdf.lib.datetime_to_tt2000(pd.Timestamp(year, month, day).to_pydatetime()))


@pytest.mark.parametrize('start', [(2017, 3, 1), (2019, 8, 16), (2021, 2, 23), (2024, 11, 5)])
def test_matches_pycdf_on_a_burst_like_run(start):
	t0 = _tt2000(*start) + 45_175_945                       # a realistic ns remainder
	raw = t0 + np.arange(200_000, dtype=np.int64) * 122_072
	got = _epoch_datetime64(_FakeCDF(raw), 'Epoch')
	want = _slow(raw)
	assert got.dtype == want.dtype
	assert np.array_equal(got.values, want.values)


def test_rounds_half_microseconds_like_pycdf():
	base = _tt2000(2021, 2, 23)
	raw = base + np.array([0, 499, 500, 501, 999, 1_000, 1_499, 1_500, 1_000_000_499, 1_000_000_500])
	assert np.array_equal(_epoch_datetime64(_FakeCDF(raw), 'Epoch').values, _slow(raw).values)


def test_a_file_spanning_a_leap_second_falls_back():
	# 2016-12-31T23:59:60 is the most recent leap second.
	edge = _tt2000(2017, 1, 1)
	raw = edge + np.arange(-3_000, 3_000, dtype=np.int64) * 1_000_000
	assert np.array_equal(_epoch_datetime64(_FakeCDF(raw), 'Epoch').values, _slow(raw).values)


def test_empty_column():
	got = _epoch_datetime64(_FakeCDF([]), 'Epoch')
	assert len(got) == 0
