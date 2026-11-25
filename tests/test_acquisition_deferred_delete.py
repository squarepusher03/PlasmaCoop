"""A failed copy that cannot be deleted is deleted LAST, after the rest of the
acquisition phase (acquisition.acquire).

Graham, 2026-09-30: "if it can't delete it, make a list or job pool to save for until
after the acquisition and validation is otherwise complete, then lastly delete them
all. Nothing should be accessing those files at this point and it shouldn't be an
issue to delete them."

A locked file is simulated by os.remove refusing it with the PermissionError Windows
raises for a file another process holds open. Every download and every delete is
recorded in one ordered event list, so "last" is checked literally. The SDC is the
scripted fake from test_acquisition; nothing touches the network or E:.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLBACKEND', 'Agg')

pytest.importorskip('spacepy.pycdf')

import acquisition
import cdf_validation
from test_acquisition import FakeSdc, clock, corrupt, good, page, roots, run  # noqa: F401 -- fixtures
from test_direct_cdf import T


@pytest.fixture
def events(monkeypatch):
	"""os.remove that refuses chosen paths. refuse[path] is how many more deletes of it
	fail (None: every one). The event list holds ('remove', path, worked) for every
	delete and ('download', t0, t1) for every SDC call (see recording)."""
	state = {'refuse': {}, 'log': []}
	real_remove = os.remove

	def remove(path, *args, **kwargs):
		left = state['refuse'].get(path, 0)
		if left is None or left > 0:
			if left is not None:
				state['refuse'][path] = left - 1
			state['log'].append(('remove', path, False))
			raise PermissionError(13, 'The process cannot access the file because it is being used '
			                          'by another process', path)
		real_remove(path, *args, **kwargs)
		state['log'].append(('remove', path, True))

	monkeypatch.setattr(os, 'remove', remove)
	return state


MAX_DOWNLOADS = 20   # far past any of these scenarios: more means acquire() is looping


def recording(sdc, state):
	def call(inst, mode, probes, t0, t1, data_dir):
		state['log'].append(('download', t0, t1))
		# Before 2026-09-30 a file that stayed locked was refetched forever, and the
		# fake clock never sleeps, so a regression would hang the suite, not fail it.
		if sum(1 for e in state['log'] if e[0] == 'download') > MAX_DOWNLOADS:
			raise AssertionError(f'more than {MAX_DOWNLOADS} SDC calls: acquire() is looping')
		return sdc(inst, mode, probes, t0, t1, data_dir)
	return call


def test_a_corrupt_file_whose_delete_fails_is_deleted_last_and_not_refetched(roots, clock, events):
	data, cache = roots
	a, c = good(data, 'A'), good(data, 'C')
	b = corrupt(data, 'B')
	events['refuse'][b] = 1                       # locked when judged, free by the end
	sdc = FakeSdc(clock, [])
	got = run(data, cache, T(5), T(7), sdc=recording(sdc, events))

	assert len(sdc.calls) == 1                    # the first download only: B is not fetched again
	assert not os.path.exists(b)
	assert events['log'][-1] == ('remove', b, True)   # the very last thing acquire did
	assert got.unresolved == {os.path.basename(b)} and got.given_up == set()
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 1   # counted once, not per rescan
	assert got.files == {1: [a, c]}


def test_a_file_that_still_cannot_be_deleted_is_reported_and_left_out(roots, clock, events):
	data, cache = roots
	a, c = good(data, 'A'), good(data, 'C')
	b = corrupt(data, 'B')
	events['refuse'][b] = None                    # locked for good
	lines = []
	sdc = FakeSdc(clock, [])
	got = run(data, cache, T(5), T(7), sdc=recording(sdc, events), log=lines.append)

	assert len(sdc.calls) == 1
	assert os.path.exists(b)
	assert got.files == {1: [a, c]}               # never handed to the compute phase as validated
	assert got.unresolved == {os.path.basename(b)}
	assert any('still cannot be deleted' in line for line in lines)
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 1
	# One delete when judged, one at the end -- the rescans in between leave it alone.
	assert [e for e in events['log'] if e[0] == 'remove'] == [('remove', b, False), ('remove', b, False)]


def test_a_rate_limit_page_whose_delete_fails_is_deleted_last_uncounted(roots, clock, events):
	data, cache = roots
	good(data, 'A'), good(data, 'C')
	b = page(data, 'B')
	events['refuse'][b] = 1
	sdc = FakeSdc(clock, [])
	got = run(data, cache, T(5), T(7), sdc=recording(sdc, events))

	assert len(sdc.calls) == 1
	assert not os.path.exists(b)
	assert events['log'][-1] == ('remove', b, True)
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 0   # a 429 page is never an attempt
	assert got.unresolved == {os.path.basename(b)}


def test_validate_only_defers_the_delete_too(roots, clock, events):
	data, cache = roots
	good(data, 'A'), good(data, 'C')
	b = corrupt(data, 'B')
	events['refuse'][b] = 1
	got = run(data, cache, T(5), T(7), download=False)

	assert got.downloads == 0
	assert not os.path.exists(b)
	assert events['log'][-1] == ('remove', b, True)
	assert got.unresolved == {os.path.basename(b)}


def test_a_new_copy_over_a_locked_one_is_judged_not_deleted(roots, clock, events):
	"""The final deletion is for the bytes that failed. If a download replaces them
	(the lock went away and pyspedas wrote over the file), the new copy is validated
	like any arrival, and a good one stays."""
	data, cache = roots
	good(data, 'A'), good(data, 'D')
	b = corrupt(data, 'B')
	c = corrupt(data, 'C')
	events['refuse'][b] = None                    # B's bad copy can never be deleted
	spare = os.path.join(os.path.dirname(data), 'spare')

	def write_good_b_over_the_old_one():
		# A download writes over the file in place; pycdf cannot create over one.
		with open(good(spare, 'B'), 'rb') as src, open(b, 'wb') as dst:
			dst.write(src.read())

	sdc = FakeSdc(clock, [
		(acquisition.OK, []),                             # the first download
		(acquisition.OK, [lambda: good(data, 'C'), write_good_b_over_the_old_one]),   # C's refetch rewrites B too
	])
	got = run(data, cache, T(5), T(9), sdc=recording(sdc, events))

	assert os.path.exists(b) and os.path.exists(c)
	assert cdf_validation.passed(b, 'B', pickle_root=cache)
	assert cdf_validation.passed(c, 'B', pickle_root=cache)
	assert got.unresolved == set() and got.given_up == set()
	assert ('remove', b, True) not in events['log']
