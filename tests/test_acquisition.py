"""The acquisition phase (acquisition.py): download, validate, retry -- before any
computation -- so the compute phase never meets a corrupt file.

Every rule Graham set is pinned here (DESIGN_acquisition-phase.md §6):

  * a failed copy is deleted on the spot, every attempt, and counted;
  * a file that never validates is downloaded exactly 5 times, then given up;
  * a 429 page is deleted and NOT counted; the gap between SDC calls is at least
    30 s, a 429 sets it to 300 s, and every call that brings a CDF halves it, never
    below 30 s (Graham, 2026-10-01; the 2026-09-28 rule was 20 s, +15 s per 429,
    capped at 90 s);
  * a given-up name that comes back is validated: intact clears it, corrupt is
    deleted with no further retry;
  * every received file is validated, and every file discovery can open;
  * the CDF library validates during acquisition, and not after it.

The SDC is a scripted fake and the pacer runs on a fake clock: nothing here
touches the network or E:, and nothing sleeps.
"""
import logging
import os
import sys
import types
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLBACKEND', 'Agg')

pycdf = pytest.importorskip('spacepy.pycdf')

import acquisition
import cdf_validation
import ffttest
from test_direct_cdf import write_cdf, T, N_4S

PAGE_429 = b'<html><head><title>429 Too Many Requests</title></head><body></body></html>'


@pytest.fixture(autouse=True)
def clock(monkeypatch):
	"""A fresh pacer on a fake clock, a fresh per-process check cache, and the CDF
	library's validation mode put back afterwards (acquire switches it off)."""
	state = {'t': 0.0, 'sleeps': []}

	def sleep(dt):
		state['sleeps'].append(dt)
		state['t'] += dt

	monkeypatch.setattr(acquisition, '_now', lambda: state['t'])
	monkeypatch.setattr(acquisition, '_sleep', sleep)
	monkeypatch.setattr(acquisition, '_PACER', acquisition._SdcPacer())
	monkeypatch.setattr(ffttest, '_CHECKED_CDFS', set())
	mode = cdf_validation.get_mode()
	yield state
	if mode is not None:
		pycdf.lib._library.CDFsetValidate(pycdf.const.VALIDATEFILEon if mode else pycdf.const.VALIDATEFILEoff)


def path_of(data, stamp, probe=1, version='v1.0.0'):
	return os.path.join(data, f'mms{probe}', 'scm', 'brst', 'l2', 'scb', stamp[:4], stamp[4:6], stamp[6:8],
	                    f'mms{probe}_scm_brst_l2_scb_{stamp}_{version}.cdf')


# Four contiguous 4.004 s files, A B C D, tiling T(0)..T(16).
STAMPS = {'A': '20210223100000', 'B': '20210223100004', 'C': '20210223100008', 'D': '20210223100012'}
FIRST = {'A': 0, 'B': N_4S, 'C': 2 * N_4S, 'D': 3 * N_4S}


def good(data, which, probe=1):
	return write_cdf(data, 'B', STAMPS[which], T(0), N_4S, probe=probe, first=FIRST[which])


def corrupt(data, which, probe=1):
	"""A download cut short: the real file's first half."""
	path = good(data, which, probe)
	with open(path, 'rb') as fh:
		head = fh.read(os.path.getsize(path) // 2)
	with open(path, 'wb') as fh:
		fh.write(head)
	return path


def page(data, which):
	"""The SDC's 429 reply, saved under the CDF's name as pyspedas saves it."""
	path = path_of(data, STAMPS[which])
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'wb') as fh:
		fh.write(PAGE_429)
	return path


class FakeSdc:
	"""The SDC, scripted: call i returns script[i] = (status, [what it delivers]).
	Past the end of the script every call is OK and delivers nothing. Records each
	call's arguments, the fake clock, and whether each watched file was on disk."""

	def __init__(self, clock, script, watch=()):
		self.clock, self.script, self.watch = clock, list(script), list(watch)
		self.calls, self.times, self.present = [], [], []

	def __call__(self, inst, mode, probes, t0, t1, data_dir):
		self.calls.append((inst, mode, tuple(probes), t0, t1))
		self.times.append(self.clock['t'])
		self.present.append([os.path.exists(p) for p in self.watch])
		status, deliver = self.script.pop(0) if self.script else (acquisition.OK, [])
		for make in deliver:
			make()
		return status


def run(data, cache, start, end, sdc=None, download=True, log=None):
	return acquisition.acquire('B', [1], start, end, data_dir=data, pickle_root=cache, download=download,
	                           downloader=sdc, log=log or (lambda *a: None))


@pytest.fixture
def roots(tmp_path):
	return str(tmp_path / 'pydata'), str(tmp_path / 'cache')


# ---------------------------------------------------------------------------
# which files
# ---------------------------------------------------------------------------

def test_a_clean_range_is_validated_and_nothing_else_is_touched(roots):
	data, cache = roots
	a, b, c, d = (good(data, w) for w in 'ABCD')
	got = run(data, cache, T(5), T(7), download=False)
	# B holds the range; A and C are its seam neighbours. D is two files out.
	assert got.files == {1: [a, b, c]}
	assert [cdf_validation.passed(p, 'B', pickle_root=cache) for p in (a, b, c, d)] == [True, True, True, False]
	assert all(os.path.exists(p) for p in (a, b, c, d))
	assert (got.given_up, got.unresolved, got.downloads) == (set(), set(), 0)


@pytest.mark.parametrize('tree, start, end', [
	('contiguous', T(5), T(9)),
	('stale_anchor', T(10), T(35)),     # A's data stops at T(4): discovery falls forward to F
])
def test_the_candidates_cover_every_file_discovery_and_pairing_open(roots, tree, start, end):
	"""The compute phase opens what find_cdf_files, find_local_cdfs (which reads the
	anchor's bounds) and pair_with_next_cdf (which reads the seam neighbours) hand
	it. All of it must have been validated by acquisition."""
	data, cache = roots
	if tree == 'contiguous':
		for w in 'ABCD':
			good(data, w)
		write_cdf(data, 'B', '20210223100016', T(0), N_4S, first=4 * N_4S)
	else:
		good(data, 'A')
		write_cdf(data, 'B', '20210223100030', T(30), N_4S)
		write_cdf(data, 'B', '20210223100040', T(40), N_4S)
	candidates = set(acquisition._candidates(data, 'scm', 'scb', 1, start, end))

	opened = set(ffttest.find_cdf_files(data, 'scm', 'scb', start, end, probe=1, download=False))
	local = ffttest.find_local_cdfs(data, 'scm', 'scb', start, end, probe=1, pickle_root=cache)
	for row in ffttest.pair_with_next_cdf(local, data, 'scm', 'scb', 1):
		opened.update(p for p in (row[1], row[3], row[5]) if p)
	assert local and opened <= candidates


def test_every_version_of_a_candidate_stamp_is_a_candidate(roots):
	data, _cache = roots
	later = path_of(data, STAMPS['A'], version='v1.0.1')
	os.rename(good(data, 'A'), later)
	a = good(data, 'A')
	assert set(acquisition._candidates(data, 'scm', 'scb', 1, T(1), T(2))) >= {a, later}


# ---------------------------------------------------------------------------
# which probes
# ---------------------------------------------------------------------------

def test_only_the_selected_probes_are_requested_each_in_its_own_call(roots, clock):
	"""Graham, 2026-09-28: "acquire should not call using all the probes at once;
	only request the probes the user has selected." """
	data, cache = roots
	sdc = FakeSdc(clock, [])
	acquisition.acquire('B', [3, 1], T(1), T(11), data_dir=data, pickle_root=cache, downloader=sdc,
	                    log=lambda *a: None)
	assert [call[2] for call in sdc.calls] == [(1,), (3,)]      # never 2 or 4, never together
	assert sdc.times == [0, 30]                                  # and paced like any two calls (30 s minimum)


def test_a_retry_is_one_call_per_probe_too(roots, clock):
	data, cache = roots
	for probe in (1, 3):
		good(data, 'A', probe), good(data, 'C', probe)
	sdc = FakeSdc(clock, [
		(acquisition.OK, [lambda: corrupt(data, 'B', 1)]),
		(acquisition.OK, [lambda: corrupt(data, 'B', 3)]),
		(acquisition.OK, [lambda: good(data, 'B', 1)]),
		(acquisition.OK, [lambda: good(data, 'B', 3)]),
	])
	got = acquisition.acquire('B', [1, 3], T(1), T(11), data_dir=data, pickle_root=cache, downloader=sdc,
	                          log=lambda *a: None)
	assert [call[2] for call in sdc.calls] == [(1,), (3,), (1,), (3,)]
	assert all(path_of(data, STAMPS['B'], probe) in got.files[probe] for probe in (1, 3))


# ---------------------------------------------------------------------------
# retries and giving up
# ---------------------------------------------------------------------------

def test_a_corrupt_copy_is_deleted_each_time_and_fetched_again_until_one_passes(roots, clock):
	data, cache = roots
	a, c = good(data, 'A'), good(data, 'C')
	b = path_of(data, STAMPS['B'])
	sdc = FakeSdc(clock, [
		(acquisition.OK, [lambda: corrupt(data, 'B')]),       # the range's download
		(acquisition.OK, [lambda: corrupt(data, 'B')]),       # retry 1
		(acquisition.OK, [lambda: good(data, 'B')]),          # retry 2
	], watch=[b])
	got = run(data, cache, T(1), T(11), sdc)

	assert len(sdc.calls) == 3
	assert sdc.present == [[False], [False], [False]]       # each bad copy was gone before the next call
	# The retries ask for B alone: its stamp, two seconds on.
	assert [(t0, t1) for *_x, t0, t1 in sdc.calls[1:]] == [(T(4), T(6))] * 2
	assert got.files == {1: [a, b, c]}
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 0      # a pass clears the count
	assert got.given_up == set()


def test_a_file_that_never_validates_is_downloaded_five_times_then_given_up(roots, clock):
	data, cache = roots
	a, c = good(data, 'A'), good(data, 'C')
	b = path_of(data, STAMPS['B'])
	sdc = FakeSdc(clock, [(acquisition.OK, [lambda: corrupt(data, 'B')])] * 10)
	got = run(data, cache, T(1), T(11), sdc)

	assert len(sdc.calls) == cdf_validation.GIVE_UP_AFTER == 5
	assert not os.path.exists(b)
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 5
	assert got.given_up == {os.path.basename(b)}
	assert got.files == {1: [a, c]}
	assert got.unresolved == set()


def test_a_refetch_that_brings_nothing_back_counts_as_a_failed_download(roots, clock):
	data, cache = roots
	good(data, 'A'), good(data, 'C')
	b = path_of(data, STAMPS['B'])
	sdc = FakeSdc(clock, [(acquisition.OK, [lambda: corrupt(data, 'B')])])    # then nothing, forever
	got = run(data, cache, T(1), T(11), sdc)
	assert len(sdc.calls) == 5
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 5
	assert got.given_up == {os.path.basename(b)}


def test_without_downloads_a_corrupt_file_is_deleted_counted_and_left_unresolved(roots, clock):
	data, cache = roots
	a, c = good(data, 'A'), good(data, 'C')
	b = corrupt(data, 'B')
	sdc = FakeSdc(clock, [])
	got = run(data, cache, T(1), T(11), sdc, download=False)
	assert sdc.calls == []
	assert not os.path.exists(b)
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 1
	assert got.unresolved == {os.path.basename(b)}
	assert got.files == {1: [a, c]}


def test_a_download_that_fails_midway_still_has_what_it_left_validated(roots, clock):
	"""A download killed at the timeout can leave a half-written file behind.
	Downloading stops, but what it left is validated like anything received."""
	data, cache = roots
	a, c = good(data, 'A'), good(data, 'C')
	b = path_of(data, STAMPS['B'])
	sdc = FakeSdc(clock, [(acquisition.FAILED, [lambda: corrupt(data, 'B')]),
	                      (acquisition.OK, [lambda: good(data, 'B')])])
	got = run(data, cache, T(1), T(11), sdc)
	assert len(sdc.calls) == 1                             # nothing is downloaded after a failure
	assert not os.path.exists(b)
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 1
	assert got.unresolved == {os.path.basename(b)}
	assert got.files == {1: [a, c]}


def test_a_given_up_name_that_comes_back_intact_clears_its_record(roots, clock):
	data, cache = roots
	good(data, 'A'), good(data, 'C')
	b = path_of(data, STAMPS['B'])
	for _ in range(5):
		cdf_validation.record_failure(b, 'B', pickle_root=cache)
	sdc = FakeSdc(clock, [(acquisition.OK, [lambda: good(data, 'B')])])
	got = run(data, cache, T(1), T(11), sdc)
	assert len(sdc.calls) == 1
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 0
	assert b in got.files[1] and got.given_up == set()


def test_a_given_up_name_that_comes_back_corrupt_is_deleted_without_a_retry(roots, clock):
	data, cache = roots
	good(data, 'A'), good(data, 'C')
	b = path_of(data, STAMPS['B'])
	for _ in range(5):
		cdf_validation.record_failure(b, 'B', pickle_root=cache)
	sdc = FakeSdc(clock, [(acquisition.OK, [lambda: corrupt(data, 'B')])] * 3)
	got = run(data, cache, T(1), T(11), sdc)
	assert len(sdc.calls) == 1
	assert not os.path.exists(b)
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 6
	assert got.given_up == {os.path.basename(b)}


def test_every_received_file_is_validated_even_outside_the_range(roots, clock):
	"""pyspedas can deliver more than the range asked for (it always hands back one
	file before it). Whatever arrives is validated, and a bad stray is retried alone."""
	data, cache = roots
	stray = '20210223100040'
	sdc = FakeSdc(clock, [
		(acquisition.OK, [lambda: good(data, 'A'), lambda: good(data, 'B'),
		                  lambda: write_cdf(data, 'B', stray, T(40), N_4S),
		                  lambda: corrupt_at(path_of(data, stray))]),
		(acquisition.OK, [lambda: write_cdf(data, 'B', stray, T(40), N_4S)]),
	])
	got = run(data, cache, T(1), T(3), sdc)
	s = path_of(data, stray)
	assert sdc.calls[1][2:] == ((1,), datetime(2021, 2, 23, 10, 0, 40), datetime(2021, 2, 23, 10, 0, 42))
	assert cdf_validation.passed(s, 'B', pickle_root=cache)
	assert s not in got.files[1]                           # validated, but not the range's


def corrupt_at(path):
	with open(path, 'rb') as fh:
		head = fh.read(os.path.getsize(path) // 2)
	with open(path, 'wb') as fh:
		fh.write(head)


def test_a_failed_copy_that_a_later_version_supersedes_is_not_fetched_again(roots, clock):
	data, cache = roots
	good(data, 'A'), good(data, 'C')
	later = path_of(data, STAMPS['B'], version='v1.0.1')
	os.rename(good(data, 'B'), later)
	old = corrupt(data, 'B')                               # v1.0.0
	sdc = FakeSdc(clock, [])
	got = run(data, cache, T(1), T(11), sdc)
	assert len(sdc.calls) == 1                             # the range's download, no retry
	assert not os.path.exists(old) and os.path.exists(later)
	assert cdf_validation.attempts(old, 'B', pickle_root=cache) == 1
	assert got.given_up == set() and got.unresolved == set()


# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------

def test_a_429_page_is_deleted_uncounted_and_raises_the_gap(roots, clock):
	data, cache = roots
	a, c = good(data, 'A'), good(data, 'C')
	b = path_of(data, STAMPS['B'])
	sdc = FakeSdc(clock, [
		(acquisition.OK, [lambda: page(data, 'B')]),
		(acquisition.OK, [lambda: page(data, 'B')]),
		(acquisition.OK, [lambda: good(data, 'B')]),
	], watch=[b])
	got = run(data, cache, T(1), T(11), sdc)

	assert sdc.present == [[False], [False], [False]]      # each page was deleted
	assert cdf_validation.attempts(b, 'B', pickle_root=cache) == 0
	assert [t1 - t0 for t0, t1 in zip(sdc.times, sdc.times[1:])] == [300, 300]
	assert got.gap_s == 150 and got.rate_limited == 2       # halved: the last call brought a CDF
	assert got.files == {1: [a, b, c]}


def test_a_refused_listing_is_repeated_after_a_longer_gap_until_it_goes_through(roots, clock):
	data, cache = roots
	for w in 'ABC':
		good(data, w)
	sdc = FakeSdc(clock, [(acquisition.RATE_LIMITED, []), (acquisition.RATE_LIMITED, []), (acquisition.OK, [])])
	got = run(data, cache, T(1), T(11), sdc)
	assert len(sdc.calls) == 3 and len(set(sdc.calls)) == 1      # the same call, repeated
	assert [t1 - t0 for t0, t1 in zip(sdc.times, sdc.times[1:])] == [300, 300]
	# The files were all on disk already, so no CDF arrived: nothing halves the gap.
	assert got.gap_s == 300 and got.downloads == 3
	assert len(got.files[1]) == 3


def test_every_refusal_in_a_long_run_waits_the_full_300_s(roots, clock):
	data, cache = roots
	for w in 'ABC':
		good(data, w)
	sdc = FakeSdc(clock, [(acquisition.RATE_LIMITED, [])] * 7 + [(acquisition.OK, [])])
	got = run(data, cache, T(1), T(11), sdc)
	assert [t1 - t0 for t0, t1 in zip(sdc.times, sdc.times[1:])] == [300] * 7
	assert got.downloads == 8 and got.rate_limited == 7


def test_a_429_page_left_by_an_earlier_run_is_refetched_without_raising_the_gap(roots, clock):
	data, cache = roots
	good(data, 'A'), good(data, 'C')
	page(data, 'B')                                        # on disk before this run
	sdc = FakeSdc(clock, [(acquisition.OK, []), (acquisition.OK, [lambda: good(data, 'B')])])
	got = run(data, cache, T(1), T(11), sdc)
	assert len(sdc.calls) == 2
	assert sdc.times[1] - sdc.times[0] == 30
	assert got.gap_s == 30 and got.rate_limited == 0


def test_the_pacer_waits_out_the_gap_since_the_last_call(clock):
	pacer = acquisition._SdcPacer()
	assert pacer.wait(log=lambda *a: None) and clock['sleeps'] == []      # nothing to wait for yet
	pacer.called()
	clock['t'] += 3
	pacer.wait(log=lambda *a: None)
	assert sum(clock['sleeps']) == pytest.approx(27)      # 30 s after the last call ended


def test_a_429_waits_300_s_and_each_cdf_halves_the_gap_down_to_30_s():
	"""Graham, 2026-10-01: "If you hit a 429, wait 300s, then dip down by half every
	time a CDF comes through, with a minimum wait time of 30 seconds" """
	quiet = lambda *a: None
	pacer = acquisition._SdcPacer()
	assert pacer.gap_s == 30
	pacer.settle(rate_limited=True, got_cdf=False, log=quiet)
	assert pacer.gap_s == 300
	gaps = []
	for _ in range(5):
		pacer.settle(rate_limited=False, got_cdf=True, log=quiet)
		gaps.append(pacer.gap_s)
	assert gaps == [150, 75, 37.5, 30, 30]                # halved per CDF, never below 30
	pacer.settle(rate_limited=True, got_cdf=False, log=quiet)
	pacer.settle(rate_limited=True, got_cdf=False, log=quiet)
	assert pacer.gap_s == 300                             # a second 429 waits the same 300 s
	pacer.settle(rate_limited=False, got_cdf=False, log=quiet)
	assert pacer.gap_s == 300                             # a clean call that brought nothing changes nothing
	pacer.settle(rate_limited=False, got_cdf=True, log=quiet)
	pacer.settle(rate_limited=True, got_cdf=True, log=quiet)
	assert pacer.gap_s == 300                             # a 429 in the same call outranks its CDF


@pytest.mark.parametrize('logged, status', [
	(None, acquisition.OK),
	((logging.WARNING, 'Request to MMS SDC returned HTTP status code %d', 429), acquisition.RATE_LIMITED),
	((logging.WARNING, 'Request to MMS SDC returned HTTP status code %d', 503), acquisition.FAILED),
	((logging.ERROR, 'No internet connection!'), acquisition.FAILED),
])
def test_the_download_subprocess_reports_what_the_sdc_said_in_its_exit_code(tmp_path, monkeypatch, logged, status):
	"""pyspedas reports a refused listing only as a log line and writes nothing, so
	the subprocess has to say so. Run in-process here with pyspedas stubbed out."""
	monkeypatch.setitem(sys.modules, 'pyspedas', types.ModuleType('pyspedas'))

	def fake_do_download(*_a):
		if logged:
			logging.log(logged[0], *logged[1:])

	monkeypatch.setattr(ffttest, '_do_download', fake_do_download)
	handlers = list(logging.getLogger().handlers)
	with pytest.raises(SystemExit) as exit_:
		acquisition._download_and_report('scm', 'scb', ['1'], ['2021-02-23/10:00:00', '2021-02-23/10:00:10'],
		                                 str(tmp_path))
	assert acquisition._status_from_exitcode(exit_.value.code) == status
	assert logging.getLogger().handlers == handlers        # the watcher is taken off again


@pytest.mark.parametrize('code, status', [(0, 'ok'), (2, 'rate_limited'), (6, 'rate_limited'),
                                          (4, 'failed'), (1, 'failed'), (-15, 'failed'), (None, 'failed')])
def test_exit_codes_map_to_outcomes(code, status):
	assert acquisition._status_from_exitcode(code) == status


def test_only_a_429_reply_reads_as_a_rate_limit_page():
	assert acquisition._is_rate_limit_page(PAGE_429)
	assert acquisition._is_rate_limit_page(b'Too Many Requests')
	assert not acquisition._is_rate_limit_page(b'<html>503 Service Unavailable</html>')
	assert not acquisition._is_rate_limit_page(b'\xcd\xf3\x00\x01' + b'429' * 10)     # a CDF, however cut
	assert not acquisition._is_rate_limit_page(bytes(range(256)) + b'429')           # binary garbage


# ---------------------------------------------------------------------------
# the library's validation mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('before', ['on', 'off'])
def test_the_library_validates_during_acquisition_and_not_after(roots, monkeypatch, before):
	"""'on' is a fresh process (the library default); 'off' is a process an earlier
	acquire already switched off -- acquisition's own checks must still validate."""
	data, cache = roots
	for w in 'ABC':
		good(data, w)
	modes = []
	real = ffttest._check_cdf
	monkeypatch.setattr(ffttest, '_check_cdf', lambda *a: modes.append(cdf_validation.get_mode()) or real(*a))
	pycdf.lib._library.CDFsetValidate(pycdf.const.VALIDATEFILEon if before == 'on' else pycdf.const.VALIDATEFILEoff)
	run(data, cache, T(5), T(7), download=False)
	assert modes == [1, 1, 1]
	assert cdf_validation.get_mode() == 0


def test_acquire_refuses_to_run_without_somewhere_to_count(roots):
	data, _cache = roots
	with pytest.raises(ValueError):
		acquisition.acquire('B', [1], T(0), T(1), data_dir=data, pickle_root=None, download=False)
