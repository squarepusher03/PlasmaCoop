"""cdf_validation: the shared record of CDF versions that passed validation.

What has to hold for it to replace ffttest._CHECKED_CDFS safely:

  * a recorded version reads back as passed, and an unrecorded one does not;
  * the key is the filename and the field letter, nothing else -- the 60 ms
    validation is the check, this table is only where its answer lives, so
    nothing here ever stats or opens the CDF;
  * the key is the BASENAME, so the same file counts as passed whether it was
    recorded under the Windows view of the data drive or the WSL one;
  * the field letter is part of the key, because _check_cdf reads different
    variables per field;
  * recording twice leaves one row;
  * the two facts it carries -- validated, and sidecar-computed -- are
    independent while the version holds, and BOTH reset when it moves, which is
    what makes "recalculate a re-downloaded CDF's sidecar" fall out for free;
  * a reader never creates the database (pool workers must not each make one);
  * the failed-redownload counter climbs on each failure, resets on a pass, and
    survives the file being deleted;
  * a file that goes fail -> pass is different bytes, so its sidecar is cleared;
  * every failure degrades to "not validated" rather than raising;
  * a second connection can read while a write is in flight.

Nothing here touches E: -- every path is pytest's tmp_path, and the CDFs are
plain files, since none of these functions opens a CDF. The validation-mode
guard is covered separately and skips when spacepy is not importable.
"""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cdf_validation


NAME = 'mms1_scm_brst_l2_scb_20190816010003_v2.2.1.cdf'


def _cdf(tmp_path, name=NAME, body=b'not a real cdf'):
	"""A stand-in file. These functions only ever stat() it."""
	p = tmp_path / 'pydata' / name
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_bytes(body)
	return str(p)


def _reg(tmp_path):
	return str(tmp_path / 'cache' / cdf_validation.REGISTRY_NAME)


# ---------------------------------------------------------------------------
# the path helper
# ---------------------------------------------------------------------------

def test_registry_path_uses_given_root(tmp_path):
	got = cdf_validation.registry_path(str(tmp_path))
	assert got == os.path.join(str(tmp_path), 'cdf_validation.sqlite')


def test_registry_path_defaults_to_pickle_root():
	"""The readers that need this are handed no pickle_root, so the default has
	to come from data_paths -- that is the whole reason it exists."""
	from data_paths import PICKLE_ROOT
	assert cdf_validation.registry_path() == os.path.join(PICKLE_ROOT, 'cdf_validation.sqlite')


# ---------------------------------------------------------------------------
# record / passed
# ---------------------------------------------------------------------------

def test_unrecorded_file_is_not_passed(tmp_path):
	assert cdf_validation.passed(_cdf(tmp_path), 'B', registry=_reg(tmp_path)) is False


def test_recorded_file_is_passed(tmp_path):
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	assert cdf_validation.record(cdf, 'B', registry=reg) is True
	assert cdf_validation.passed(cdf, 'B', registry=reg) is True


def test_registry_is_created_with_its_schema_on_first_write(tmp_path):
	"""No separate migration step: the first record() makes the directory, the
	file and the table."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	assert not os.path.exists(reg)
	cdf_validation.record(cdf, 'B', registry=reg)
	assert os.path.exists(reg)
	conn = sqlite3.connect(reg)
	try:
		names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
	finally:
		conn.close()
	assert 'touched' in names




def test_key_is_basename_not_absolute_path(tmp_path):
	"""One registry serves the Windows venv's view of the data drive and WSL's,
	so the same file recorded under one path must read as passed under the
	other. Same bytes, same mtime, different directory. (Measured 2026-09-24:
	both views report identical st_mtime_ns and st_size for a real CDF.)"""
	a = _cdf(tmp_path / 'winview')
	b = _cdf(tmp_path / 'wslview')
	st = os.stat(a)
	os.utime(b, ns=(st.st_atime_ns, st.st_mtime_ns))
	reg = _reg(tmp_path)
	cdf_validation.record(a, 'B', registry=reg)
	assert a != b
	assert cdf_validation.passed(b, 'B', registry=reg) is True


def test_field_letter_is_part_of_the_key(tmp_path):
	"""_check_cdf reads different variables per field, so 'passed as B' says
	nothing about E. _CHECKED_CDFS keyed on the letter too."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record(cdf, 'B', registry=reg)
	assert cdf_validation.passed(cdf, 'B', registry=reg) is True
	assert cdf_validation.passed(cdf, 'E', registry=reg) is False


def test_full_key_accepted_as_well_as_letter(tmp_path):
	"""Callers hold keys like 'Bz'/'Ez'; only the first letter is the field, and
	check_cdfs already indexes key[0]. Both forms must agree."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record(cdf, 'Bz', registry=reg)
	assert cdf_validation.passed(cdf, 'B', registry=reg) is True


def test_recording_twice_leaves_one_row(tmp_path):
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record(cdf, 'B', registry=reg)
	cdf_validation.record(cdf, 'B', registry=reg)
	conn = sqlite3.connect(reg)
	try:
		n = conn.execute('SELECT COUNT(*) FROM touched').fetchone()[0]
	finally:
		conn.close()
	assert n == 1



def test_two_versions_of_one_file_are_separate_entries(tmp_path):
	"""v2.2.0 and v2.2.1 are different names, so validating one says nothing
	about the other."""
	old = _cdf(tmp_path, name='mms1_scm_brst_l2_scb_20190816010003_v2.2.0.cdf')
	new = _cdf(tmp_path, name='mms1_scm_brst_l2_scb_20190816010003_v2.2.1.cdf')
	reg = _reg(tmp_path)
	cdf_validation.record(old, 'B', registry=reg)
	assert cdf_validation.passed(old, 'B', registry=reg) is True
	assert cdf_validation.passed(new, 'B', registry=reg) is False


# ---------------------------------------------------------------------------
# record_many
# ---------------------------------------------------------------------------

def test_record_many_writes_every_job(tmp_path):
	reg = _reg(tmp_path)
	jobs = [(_cdf(tmp_path, name=f'mms1_scm_brst_l2_scb_201908160100{i:02d}_v2.2.1.cdf'), 'B')
	        for i in range(5)]
	assert cdf_validation.record_many(jobs, registry=reg) == 5
	assert all(cdf_validation.passed(p, 'B', registry=reg) for p, _ in jobs)



def test_record_many_on_nothing_writes_no_database(tmp_path):
	reg = _reg(tmp_path)
	assert cdf_validation.record_many([], registry=reg) == 0
	assert not os.path.exists(reg)


# ---------------------------------------------------------------------------
# degrading, not raising
# ---------------------------------------------------------------------------

def test_reader_does_not_create_the_database(tmp_path):
	"""A pool of workers each creating and locking a fresh sqlite file would be
	worse than the checks they are avoiding -- and a reader has nothing to
	record anyway."""
	reg = _reg(tmp_path)
	cdf_validation.passed(_cdf(tmp_path), 'B', registry=reg)
	assert not os.path.exists(reg)
	assert not os.path.exists(os.path.dirname(reg))



def test_corrupt_registry_degrades_to_not_passed(tmp_path):
	"""A wrong 'no' costs one validated open -- exactly the old behaviour. A
	raise would cost the pipeline."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record(cdf, 'B', registry=reg)
	assert cdf_validation.passed(cdf, 'B', registry=reg) is True
	with open(reg, 'wb') as fh:
		fh.write(b'this is not a database')
	assert cdf_validation.passed(cdf, 'B', registry=reg) is False


def test_registry_that_is_a_directory_degrades(tmp_path):
	"""Nothing about a nonsense path may raise out of a reader."""
	cdf = _cdf(tmp_path)
	reg = str(tmp_path / 'a_directory')
	os.makedirs(reg, exist_ok=True)
	assert cdf_validation.passed(cdf, 'B', registry=reg) is False
	assert cdf_validation.record(cdf, 'B', registry=reg) is False


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the validation-mode guard
# ---------------------------------------------------------------------------

def test_validation_guard_toggles_and_restores():
	"""CDFgetValidate reads 1 on and 0 off (characterised 2026-09-15), and the
	mode has to come back however the block leaves."""
	pytest.importorskip('spacepy')
	before = cdf_validation.get_mode()
	assert before in (0, 1)
	with cdf_validation.validation(False) as previous:
		assert previous == before
		assert cdf_validation.get_mode() == 0
	assert cdf_validation.get_mode() == before
	with cdf_validation.validation(True):
		assert cdf_validation.get_mode() == 1
	assert cdf_validation.get_mode() == before


def test_validation_guard_restores_after_an_exception():
	pytest.importorskip('spacepy')
	before = cdf_validation.get_mode()
	with pytest.raises(RuntimeError):
		with cdf_validation.validation(False):
			assert cdf_validation.get_mode() == 0
			raise RuntimeError('boom')
	assert cdf_validation.get_mode() == before


def test_validation_guard_nests():
	"""short_interval runs a ThreadPoolExecutor and the mode is global, so the
	guard is re-entrant and each level restores what it found."""
	pytest.importorskip('spacepy')
	before = cdf_validation.get_mode()
	with cdf_validation.validation(True):
		assert cdf_validation.get_mode() == 1
		with cdf_validation.validation(False):
			assert cdf_validation.get_mode() == 0
		assert cdf_validation.get_mode() == 1
	assert cdf_validation.get_mode() == before


# ---------------------------------------------------------------------------
# the first-touch half: sidecar tracking
# ---------------------------------------------------------------------------

def test_sidecar_unrecorded_is_not_done(tmp_path):
	"""False means compute it. A file nobody has touched is a first touch."""
	assert cdf_validation.sidecar_done(_cdf(tmp_path), 'B', registry=_reg(tmp_path)) is False


def test_sidecar_recorded_is_done(tmp_path):
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	assert cdf_validation.record_sidecar(cdf, 'B', registry=reg) is True
	assert cdf_validation.sidecar_done(cdf, 'B', registry=reg) is True



def test_the_two_flags_are_independent(tmp_path):
	"""Recording one must never clear the other -- they are separate facts about
	the same file version, written by different parts of the pipeline."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record(cdf, 'B', registry=reg)
	assert (cdf_validation.passed(cdf, 'B', registry=reg),
	        cdf_validation.sidecar_done(cdf, 'B', registry=reg)) == (True, False)
	cdf_validation.record_sidecar(cdf, 'B', registry=reg)
	assert (cdf_validation.passed(cdf, 'B', registry=reg),
	        cdf_validation.sidecar_done(cdf, 'B', registry=reg)) == (True, True)
	# and in the other order, on a second file
	other = _cdf(tmp_path, name='mms1_scm_brst_l2_scb_20190816020003_v2.2.1.cdf')
	cdf_validation.record_sidecar(other, 'B', registry=reg)
	cdf_validation.record(other, 'B', registry=reg)
	assert (cdf_validation.passed(other, 'B', registry=reg),
	        cdf_validation.sidecar_done(other, 'B', registry=reg)) == (True, True)



def test_both_flags_still_one_row(tmp_path):
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record(cdf, 'B', registry=reg)
	cdf_validation.record_sidecar(cdf, 'B', registry=reg)
	cdf_validation.record(cdf, 'B', registry=reg)
	conn = sqlite3.connect(reg)
	try:
		rows = conn.execute('SELECT status, sidecar FROM touched').fetchall()
	finally:
		conn.close()
	assert rows == [('pass', 1)]


def test_record_sidecars_batches(tmp_path):
	reg = _reg(tmp_path)
	jobs = [(_cdf(tmp_path, name=f'mms1_scm_brst_l2_scb_201908160200{i:02d}_v2.2.1.cdf'), 'B')
	        for i in range(4)]
	assert cdf_validation.record_sidecars(jobs, registry=reg) == 4
	assert all(cdf_validation.sidecar_done(p, 'B', registry=reg) for p, _ in jobs)


def test_sidecar_reader_does_not_create_the_database(tmp_path):
	reg = _reg(tmp_path)
	cdf_validation.sidecar_done(_cdf(tmp_path), 'B', registry=reg)
	assert not os.path.exists(reg)


# ---------------------------------------------------------------------------
# failures are stored too, and counted -- the 5-strikes rule
# ---------------------------------------------------------------------------

def test_status_is_none_until_validated(tmp_path):
	"""None means 'validate it'. Not-yet-seen and changed-since-seen both read
	None, so a reader never has to distinguish them."""
	assert cdf_validation.status(_cdf(tmp_path), 'B', registry=_reg(tmp_path)) is None


def test_failure_is_recorded_not_just_success(tmp_path):
	"""The whole point: the database says whether it passed OR failed, so a known
	bad file need never be opened again."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_failure(cdf, 'B', registry=reg)
	assert cdf_validation.status(cdf, 'B', registry=reg) == 'fail'
	assert cdf_validation.failed(cdf, 'B', registry=reg) is True
	assert cdf_validation.passed(cdf, 'B', registry=reg) is False


def test_record_failure_returns_the_running_count(tmp_path):
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	assert cdf_validation.record_failure(cdf, 'B', registry=reg) == 1
	assert cdf_validation.record_failure(cdf, 'B', registry=reg) == 2
	assert cdf_validation.attempts(cdf, 'B', registry=reg) == 2



def test_attempts_survive_the_file_being_deleted(tmp_path):
	"""At the limit the file is deleted, so the count cannot live on the file."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_failure(cdf, 'B', registry=reg)
	cdf_validation.record_failure(cdf, 'B', registry=reg)
	os.remove(cdf)
	assert cdf_validation.attempts(cdf, 'B', registry=reg) == 2
	assert cdf_validation.record_failure(cdf, 'B', registry=reg) == 3


def test_give_up_only_at_the_limit(tmp_path):
	"""Below the limit the file is left for the redownload loop; at it, it is
	deleted and recorded in the manifest."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	for n in range(1, cdf_validation.GIVE_UP_AFTER):
		cdf_validation.record_failure(cdf, 'B', registry=reg)
		assert cdf_validation.give_up(cdf, 'B', registry=reg) is False, f'gave up at {n}'
	cdf_validation.record_failure(cdf, 'B', registry=reg)
	assert cdf_validation.attempts(cdf, 'B', registry=reg) == cdf_validation.GIVE_UP_AFTER
	assert cdf_validation.give_up(cdf, 'B', registry=reg) is True




def test_corrupted_names_lists_only_the_given_up(tmp_path):
	"""What the manifest's corrupted section is built from."""
	reg = _reg(tmp_path)
	doomed = _cdf(tmp_path, name='mms1_scm_brst_l2_scb_20190816030003_v2.2.1.cdf')
	retrying = _cdf(tmp_path, name='mms1_scm_brst_l2_scb_20190816040003_v2.2.1.cdf')
	fine = _cdf(tmp_path, name='mms1_scm_brst_l2_scb_20190816050003_v2.2.1.cdf')
	for _ in range(cdf_validation.GIVE_UP_AFTER):
		cdf_validation.record_failure(doomed, 'B', registry=reg)
	cdf_validation.record_failure(retrying, 'B', registry=reg)
	cdf_validation.record_pass(fine, 'B', registry=reg)
	assert cdf_validation.corrupted_names(registry=reg) == [os.path.basename(doomed)]
	assert cdf_validation.corrupted_names('B', registry=reg) == [os.path.basename(doomed)]
	assert cdf_validation.corrupted_names('E', registry=reg) == []


def test_corrupted_names_on_no_database(tmp_path):
	assert cdf_validation.corrupted_names(registry=_reg(tmp_path)) == []


def test_recording_a_pass_does_not_clear_the_sidecar_flag(tmp_path):
	"""Three independent facts on one row; writing one must not disturb another."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_sidecar(cdf, 'B', registry=reg)
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	cdf_validation.record_failure(cdf, 'B', registry=reg)
	assert cdf_validation.sidecar_done(cdf, 'B', registry=reg) is True
	assert cdf_validation.attempts(cdf, 'B', registry=reg) == 1
	assert cdf_validation.status(cdf, 'B', registry=reg) == 'fail'


def test_failure_of_a_missing_file_still_counts(tmp_path):
	"""A download that produced nothing at all is still an attempt."""
	reg = _reg(tmp_path)
	gone = str(tmp_path / 'pydata' / 'mms1_scm_brst_l2_scb_20190816060003_v2.2.1.cdf')
	assert cdf_validation.record_failure(gone, 'B', registry=reg) == 1
	assert cdf_validation.attempts(gone, 'B', registry=reg) == 1


# ---------------------------------------------------------------------------
# the key is the name, and what that trades away
# ---------------------------------------------------------------------------

def test_lookup_never_opens_the_cdf(tmp_path):
	"""The point of the whole table: answer without paying the 60 ms open. Every
	stand-in here is a text file, not a CDF, so anything that tried to open one
	would blow up -- and a full pass/sidecar round trip goes through clean."""
	cdf, reg = _cdf(tmp_path, body=b'definitely not a CDF'), _reg(tmp_path)
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	cdf_validation.record_sidecar(cdf, 'B', registry=reg)
	assert cdf_validation.passed(cdf, 'B', registry=reg) is True
	assert cdf_validation.sidecar_done(cdf, 'B', registry=reg) is True


def test_a_deleted_file_is_unknown_but_still_counted(tmp_path):
	"""status is version-scoped, so a file that is gone cannot be "passed". The
	strike count is name-scoped and keeps answering -- which is the whole reason
	the two have different scopes."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	cdf_validation.record_failure(cdf, 'B', registry=reg)
	os.remove(cdf)
	assert cdf_validation.status(cdf, 'B', registry=reg) is None
	assert cdf_validation.attempts(cdf, 'B', registry=reg) == 1


def test_a_truncated_refetch_under_the_same_name_is_unknown_again(tmp_path):
	"""The hole a name-only key would leave open, and why the stat stays: a
	truncated download landing under a name that already passed must NOT read as
	good. Different bytes, same name -> unknown, so it gets validated."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	with open(cdf, 'wb') as fh:
		fh.write(b'truncated')
	assert cdf_validation.status(cdf, 'B', registry=reg) is None
	assert cdf_validation.passed(cdf, 'B', registry=reg) is False


def test_a_touch_that_only_moves_mtime_is_unknown_again(tmp_path):
	"""Size alone is not enough -- a rewrite can land the same length."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	st = os.stat(cdf)
	os.utime(cdf, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
	assert cdf_validation.status(cdf, 'B', registry=reg) is None


def test_new_bytes_clear_the_sidecar_too(tmp_path):
	"""A sidecar describes the samples it was built from. New bytes, new sidecar."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	cdf_validation.record_sidecar(cdf, 'B', registry=reg)
	with open(cdf, 'ab') as fh:
		fh.write(b'refetched')
	assert cdf_validation.sidecar_done(cdf, 'B', registry=reg) is False


def test_data_version_in_the_name_is_the_version(tmp_path):
	"""v2.2.0 and v2.2.1 are different names, so they are different rows. This is
	why the name alone is a sound key."""
	old = _cdf(tmp_path, name='mms1_scm_brst_l2_scb_20190816010003_v2.2.0.cdf')
	new = _cdf(tmp_path, name='mms1_scm_brst_l2_scb_20190816010003_v2.2.1.cdf')
	reg = _reg(tmp_path)
	cdf_validation.record_pass(old, 'B', registry=reg)
	assert cdf_validation.passed(old, 'B', registry=reg) is True
	assert cdf_validation.status(new, 'B', registry=reg) is None


def test_one_row_per_name_and_field(tmp_path):
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	for _ in range(3):
		cdf_validation.record_pass(cdf, 'B', registry=reg)
		cdf_validation.record_failure(cdf, 'B', registry=reg)
	conn = sqlite3.connect(reg)
	try:
		assert conn.execute('SELECT COUNT(*) FROM touched').fetchone()[0] == 1
	finally:
		conn.close()


# ---------------------------------------------------------------------------
# the counter counts a CURRENT corruption, not history
# ---------------------------------------------------------------------------

def test_a_pass_resets_the_counter(tmp_path):
	"""Graham, 2026-09-25: the counter is "strictly for counting how many times a
	corrupted file fails to redownload, not for total times downloaded ever". A
	file that broke twice and then came down clean is simply a good file."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_failure(cdf, 'B', registry=reg)
	cdf_validation.record_failure(cdf, 'B', registry=reg)
	assert cdf_validation.attempts(cdf, 'B', registry=reg) == 2
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	assert cdf_validation.attempts(cdf, 'B', registry=reg) == 0
	assert cdf_validation.give_up(cdf, 'B', registry=reg) is False


def test_the_counter_climbs_across_redownloads(tmp_path):
	"""Each failed refetch is another strike on the same name, whatever the bytes
	look like, so 5 is actually reachable."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	for i in range(1, 4):
		with open(cdf, 'wb') as fh:              # "refetched", different bytes
			fh.write(b'x' * i)
		assert cdf_validation.record_failure(cdf, 'B', registry=reg) == i


def test_a_pass_after_failures_clears_the_sidecar(tmp_path):
	"""A file that was corrupt and now passes is new bytes, so a sidecar computed
	from the old ones is void. This is the only thing that unsets the flag."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_sidecar(cdf, 'B', registry=reg)
	cdf_validation.record_failure(cdf, 'B', registry=reg)
	assert cdf_validation.sidecar_done(cdf, 'B', registry=reg) is True   # not yet replaced
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	assert cdf_validation.sidecar_done(cdf, 'B', registry=reg) is False


def test_a_pass_on_a_healthy_file_keeps_its_sidecar(tmp_path):
	"""Re-validating a file that never failed must not throw away work."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	cdf_validation.record_sidecar(cdf, 'B', registry=reg)
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	assert cdf_validation.sidecar_done(cdf, 'B', registry=reg) is True


def test_a_missing_file_can_still_be_recorded(tmp_path):
	"""A download that produced nothing is still a strike, and at the limit the
	file is gone but the count has to remain."""
	reg = _reg(tmp_path)
	gone = str(tmp_path / 'pydata' / 'mms1_scm_brst_l2_scb_20190816070003_v2.2.1.cdf')
	assert cdf_validation.record_failure(gone, 'B', registry=reg) == 1
	assert cdf_validation.attempts(gone, 'B', registry=reg) == 1


def test_record_many_records_every_job_even_absent_files(tmp_path):
	reg = _reg(tmp_path)
	good = _cdf(tmp_path)
	gone = str(tmp_path / 'pydata' / 'mms1_scm_brst_l2_scb_20190816999999_v2.2.1.cdf')
	assert cdf_validation.record_many([(good, 'B'), (gone, 'B')], registry=reg) == 2
	assert cdf_validation.passed(good, 'B', registry=reg) is True
	# the absent one got a row, but it cannot be "passed" -- there are no bytes
	assert cdf_validation.status(gone, 'B', registry=reg) is None


def test_read_during_an_open_write_transaction_does_not_raise(tmp_path):
	"""Workers read while the parent writes. With WAL and a busy timeout a reader
	sees the last committed state instead of an error."""
	cdf, reg = _cdf(tmp_path), _reg(tmp_path)
	cdf_validation.record_pass(cdf, 'B', registry=reg)
	other = _cdf(tmp_path, name='mms1_scm_brst_l2_scb_20190816010103_v2.2.1.cdf')
	writer = sqlite3.connect(reg, timeout=5.0)
	try:
		writer.execute('BEGIN IMMEDIATE')
		writer.execute("INSERT INTO touched (name, field, status) VALUES (?, 'B', 'pass')",
		               (os.path.basename(other),))
		assert cdf_validation.passed(cdf, 'B', registry=reg) is True      # committed
		assert cdf_validation.status(other, 'B', registry=reg) is None    # uncommitted
		writer.commit()
	finally:
		writer.close()


def test_a_write_without_a_root_records_nothing(tmp_path, monkeypatch):
	"""registry_path()'s default exists for READERS that are handed no
	pickle_root. A write must never take it, or a caller that forgets one records
	into data_paths.PICKLE_ROOT -- the real data drive. A test run did exactly
	that before this guard."""
	import data_paths
	sentinel = str(tmp_path / 'must_not_be_written')
	monkeypatch.setattr(data_paths, 'PICKLE_ROOT', sentinel)
	cdf = _cdf(tmp_path)
	assert cdf_validation.record_pass(cdf, 'B') is False
	assert cdf_validation.record_failure(cdf, 'B') is None
	assert cdf_validation.record_sidecar(cdf, 'B') is False
	assert cdf_validation.record_many([(cdf, 'B')]) == 0
	assert not os.path.exists(sentinel)
