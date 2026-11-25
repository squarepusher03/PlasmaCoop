"""
No FFT cache between requests (Graham, 2026-11-19: "Toss the cache honestly. Producing FFTs is the quickest
processing step that actually constitutes as 'processing', and can be done on the fly per request").

FFT pickles live in <pickle_root>/fft_scratch/<os.name>-<pid>-<stamp>/, the old layout below that, and are deleted
when the last fft_scratch_scope on the root ends (the event window's Run), by drop_fft_scratch (a census day), at
exit, or by the next process's sweep after a crash. Synthetic data in tmp_path only.

Run (the `plasma` conda env):

    python -m pytest -q tests/test_fft_scratch.py
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLBACKEND', 'Agg')

import ffttest
import post_verify

FT = datetime(2021, 2, 23, 10, 0, 0)
OTHER_OS = 'nt' if os.name != 'nt' else 'posix'


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
	"""Each test starts with no roots registered and no scopes open (the tag itself stays this process's)."""
	monkeypatch.setattr(ffttest, '_fft_scratch_roots', set())
	monkeypatch.setattr(ffttest, '_fft_scratch_users', {})


def _fill(path, n=1000):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'wb') as fh:
		fh.write(b'x' * n)
	return n


def _pickles(root):
	return [os.path.join(r, f) for r, _d, files in os.walk(root) for f in files if f.endswith('.pkl')]


def test_path_is_under_this_process_scratch_and_keeps_the_old_layout(tmp_path):
	root = str(tmp_path)
	scratch = os.path.join(root, ffttest.FFT_SCRATCH_DIR, ffttest._scratch_tag())
	p = ffttest._fft_pickle_path(root, 'scm', 'scb', 0.25, FT, probe=1)
	assert p.startswith(scratch + os.sep)
	assert p.replace(os.sep, '/').endswith('mms/1/scm/scb/fft/pad0.250_full/2021/02/23/100000.pkl')
	p10 = ffttest._fft_pickle_path(root, 'scm', 'scb', 0.25, FT, probe=1, step=0.10)
	assert p10.replace(os.sep, '/').endswith('mms/1/scm/scb/fft/pad0.250_step0.100_full/2021/02/23/100000.pkl')


def test_computing_a_path_creates_nothing(tmp_path):
	ffttest._fft_pickle_path(str(tmp_path), 'edp', 'dce', 0.25, FT, probe=2)
	assert os.listdir(tmp_path) == []


def test_tag_names_os_pid_and_stamp():
	tag = ffttest._scratch_tag()
	assert re.fullmatch(rf'{os.name}-{os.getpid()}-\d{{8}}T\d{{6}}', tag)
	assert ffttest._scratch_owner(tag) == (os.name, os.getpid())
	assert ffttest._scratch_owner('not-a-tag') == (None, None)
	assert ffttest._scratch_owner('plain') == (None, None)


def test_drop_deletes_the_scratch_and_returns_its_bytes(tmp_path):
	root = str(tmp_path)
	n = _fill(ffttest._fft_pickle_path(root, 'scm', 'scb', 0.25, FT, probe=1), 1234)
	n += _fill(ffttest._fft_pickle_path(root, 'edp', 'dce', 0.25, FT, probe=1), 766)
	assert ffttest.drop_fft_scratch(root) == n
	assert os.listdir(os.path.join(root, ffttest.FFT_SCRATCH_DIR)) == []        # the empty parent stays
	assert ffttest.drop_fft_scratch(root) == 0


def test_drop_leaves_other_processes_folders_alone(tmp_path):
	root = str(tmp_path)
	_fill(ffttest._fft_pickle_path(root, 'scm', 'scb', 0.25, FT, probe=1))
	other = os.path.join(root, ffttest.FFT_SCRATCH_DIR, f'{OTHER_OS}-1-20260101T000000')
	_fill(os.path.join(other, 'a.pkl'))
	ffttest.drop_fft_scratch(root)
	assert os.listdir(os.path.join(root, ffttest.FFT_SCRATCH_DIR)) == [os.path.basename(other)]


def test_a_process_that_does_not_own_the_tag_deletes_nothing(tmp_path, monkeypatch):
	monkeypatch.setattr(ffttest, '_fft_scratch_tag', f'{os.name}-{os.getpid() + 1}-20260101T000000')
	root = str(tmp_path)
	p = ffttest._fft_pickle_path(root, 'scm', 'scb', 0.25, FT, probe=1)
	_fill(p)
	assert ffttest.drop_fft_scratch(root) == 0
	assert os.path.exists(p)


def test_nested_scopes_delete_only_when_the_last_one_ends(tmp_path):
	root = str(tmp_path)
	p = ffttest._fft_pickle_path(root, 'scm', 'scb', 0.25, FT, probe=1)
	with ffttest.fft_scratch_scope(root) as outer:
		with ffttest.fft_scratch_scope(root) as inner:
			n = _fill(p, 500)
		assert inner['freed'] == 0 and os.path.exists(p)
	assert outer['freed'] == n and not os.path.exists(p)


def test_a_direct_drop_waits_for_an_open_scope(tmp_path):
	root = str(tmp_path)
	p = ffttest._fft_pickle_path(root, 'scm', 'scb', 0.25, FT, probe=1)
	with ffttest.fft_scratch_scope(root) as scope:
		_fill(p)
		assert ffttest.drop_fft_scratch(root) == 0 and os.path.exists(p)      # the request is still reading it
	assert scope['freed'] > 0 and not os.path.exists(p)


def test_exit_drops_even_with_a_scope_open(tmp_path):
	root = str(tmp_path)
	p = ffttest._fft_pickle_path(root, 'scm', 'scb', 0.25, FT, probe=1)
	with ffttest.fft_scratch_scope(root):
		_fill(p)
		ffttest._drop_all_fft_scratch()
		assert not os.path.exists(p)


def test_a_scope_deletes_even_when_its_body_raises(tmp_path):
	root = str(tmp_path)
	p = ffttest._fft_pickle_path(root, 'scm', 'scb', 0.25, FT, probe=1)
	with pytest.raises(RuntimeError):
		with ffttest.fft_scratch_scope(root):
			_fill(p)
			raise RuntimeError('cancelled')
	assert not os.path.exists(p)
	assert ffttest._fft_scratch_users[os.path.normpath(root)] == 0


def test_first_use_of_a_root_sweeps_only_dead_same_os_scratch(tmp_path, monkeypatch):
	root = str(tmp_path)
	base = os.path.join(root, ffttest.FFT_SCRATCH_DIR)
	dead = os.path.join(base, f'{os.name}-111111-20260101T000000')
	live = os.path.join(base, f'{os.name}-222222-20260101T000000')
	foreign = os.path.join(base, f'{OTHER_OS}-111111-20260101T000000')
	mine = os.path.join(base, ffttest._scratch_tag())
	junk = os.path.join(base, 'not_a_tag')
	for d in (dead, live, foreign, mine, junk):
		_fill(os.path.join(d, 'mms', 'x.pkl'))
	asked = []

	def alive(pid):
		asked.append(int(pid))
		return int(pid) == 222222

	monkeypatch.setattr(post_verify, '_alive', alive)
	ffttest.fft_scratch_dir(root)
	assert not os.path.exists(dead)
	assert all(os.path.exists(d) for d in (live, foreign, mine, junk))
	assert sorted(asked) == [111111, 222222]          # the other OS's pid and this process are never asked about
	ffttest.fft_scratch_dir(root)                      # a root is swept once per process, not on every path
	assert sorted(asked) == [111111, 222222]


def test_only_a_multiprocessing_worker_inherits_the_tag(monkeypatch):
	parents = 'nt-424242-20260101T000000'
	monkeypatch.setenv(ffttest.FFT_SCRATCH_ENV, parents)

	monkeypatch.setattr(ffttest, '_fft_scratch_tag', None)
	monkeypatch.setattr(ffttest.multiprocessing, 'parent_process', lambda: None)
	own = ffttest._scratch_tag()
	assert own != parents and ffttest._scratch_owner(own) == (os.name, os.getpid())

	monkeypatch.setattr(ffttest, '_fft_scratch_tag', None)
	monkeypatch.setenv(ffttest.FFT_SCRATCH_ENV, parents)
	monkeypatch.setattr(ffttest.multiprocessing, 'parent_process', lambda: object())
	assert ffttest._scratch_tag() == parents
	assert not ffttest._owns_fft_scratch()


def test_exit_drops_every_root_used(tmp_path):
	roots = [str(tmp_path / 'a'), str(tmp_path / 'b')]
	paths = [ffttest._fft_pickle_path(r, 'scm', 'scb', 0.25, FT, probe=1) for r in roots]
	for p in paths:
		_fill(p)
	ffttest._drop_all_fft_scratch()
	assert not any(os.path.exists(p) for p in paths)


def test_prewarm_writes_only_to_scratch_and_a_drop_leaves_no_pickle(tmp_path):
	pytest.importorskip('spacepy.pycdf')
	from test_direct_cdf import write_cdf, T
	data, cache = str(tmp_path / 'pydata'), str(tmp_path / 'cache')
	write_cdf(data, 'B', '20210223100000', T(0), 32800)
	write_cdf(data, 'B', '20210223100004', T(0), 32736, first=32800)
	ffttest.prewarm(data, cache, [('Bz', 1, 0.25, 3000, [(T(0), T(8), False)])], write_manifest=False, max_workers=2)
	written = _pickles(cache)
	assert len(written) == 2
	scratch = os.path.join(cache, ffttest.FFT_SCRATCH_DIR, ffttest._scratch_tag())
	assert all(p.startswith(scratch + os.sep) for p in written)
	assert not _pickles(os.path.join(cache, 'mms'))     # the old cache folder gets nothing
	assert ffttest.drop_fft_scratch(cache) > 0
	assert _pickles(cache) == []
