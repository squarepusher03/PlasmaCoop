"""
Progress for an SDC download that is already running, measured from the disk.

event_export.download_missing hands the work to pyspedas in a subprocess, one
probe per call (see acquisition.acquire), so nothing in this process sees it happen. What it
does leave behind is files: pyspedas streams each CDF to a temporary file and
copies it into pydata/mms<probe>/<inst>/brst/l2/<mode>/YYYY/MM/DD/ once it is
complete. Watching those directories gives honest progress without asking the
SDC anything -- which matters, because the SDC rate-limits hard and an extra
API call right before a download is exactly what earned a 429.

The metric is time covered: for each (field, probe) being fetched, how many
minutes of the request lie at or before the newest file that has landed (CDF
names carry their start stamp). pyspedas works probe by probe and, within a
probe, in time order -- mms_load_data loops `for prb in probe` over the list
download_missing passes, and mms_files_in_interval sorts by time -- so that
newest stamp is how far through the request it has got. When files start
landing for a later (field, probe), every earlier one is finished. The bar
cannot know the total bytes in advance (that list lives in the SDC's reply,
inside the subprocess), so MB and file counts go in the status text instead.

No Qt in here: the window passes task.progress.emit as `report`.
"""
import math
import os
import re
import threading
import time
from collections import namedtuple
from datetime import timedelta

import event_export as ee

# Seconds between reports while a download runs.
POLL_S = 3.0

DATA_NAMES = {'B': 'B (SCM)', 'E': 'E (EDP)', 'F': 'FGM'}
_STAMP_RE = re.compile(r'_(\d{14})_')

Progress = namedtuple('Progress', 'done total text')


def _day_dirs(data_dir, inst, mode, probe, start, end):
	"""The pyspedas day directories a download of [start, end) can write to.

	Starts a day early: for burst data pyspedas also fetches the last 10 minutes
	of the previous day when the request begins within 10 minutes of midnight.
	FGM has no mode directory (its mode is '')."""
	base = os.path.join(data_dir, f'mms{probe}', inst, 'brst', 'l2', *([mode] if mode else []))
	day, last = (start - timedelta(days=1)).date(), end.date()
	while day <= last:
		yield os.path.join(base, f'{day:%Y}', f'{day:%m}', f'{day:%d}')
		day += timedelta(days=1)


def _listing(dirs):
	"""{path: (size, mtime_ns)} for every .cdf directly inside `dirs`."""
	out = {}
	for d in dirs:
		try:
			entries = os.scandir(d)
		except OSError:
			continue
		with entries:
			for entry in entries:
				if entry.name.endswith('.cdf') and entry.is_file():
					st = entry.stat()
					out[entry.path] = (st.st_size, st.st_mtime_ns)
	return out


def _elapsed(seconds):
	m, s = divmod(int(seconds), 60)
	h, m = divmod(m, 60)
	return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


class LandingWatcher:
	"""Measures a download_missing(rng, missing) call by the CDFs it lands.

	Build it BEFORE the download starts: it snapshots what is already on disk,
	and only files that are new (or rewritten) after that count."""

	def __init__(self, rng, missing, data_dir=None, clock=time.monotonic):
		self.rng = rng
		self._clock = clock
		data_dir = data_dir or ee.DATA_DIR
		by_letter = {}
		for probe, letter in missing:
			by_letter.setdefault(letter, set()).add(probe)
		# The order the work happens in: download_missing goes letter by letter
		# (sorted), and each pyspedas call goes probe by probe (sorted).
		self.pairs = [(letter, probe) for letter in sorted(by_letter) for probe in sorted(by_letter[letter])]
		self._dirs = {}
		for letter, probe in self.pairs:
			inst, mode = ee.INSTRUMENTS[letter]
			self._dirs[(letter, probe)] = list(_day_dirs(data_dir, inst, mode, probe, rng.start, rng.end))
		self._before = {pair: _listing(dirs) for pair, dirs in self._dirs.items()}
		self.span_min = max(1, math.ceil((rng.end - rng.start).total_seconds() / 60))
		self._t0 = clock()

	def _landed(self, pair):
		"""[(stamp, size)] for files that appeared or changed since the snapshot."""
		before = self._before[pair]
		out = []
		for path, (size, mtime) in _listing(self._dirs[pair]).items():
			if before.get(path) == (size, mtime):
				continue
			m = _STAMP_RE.search(os.path.basename(path))
			if m:
				out.append((ee.datetime.strptime(m.group(1), '%Y%m%d%H%M%S'), size))
		return out

	def _covered(self, stamp):
		minutes = (stamp - self.rng.start).total_seconds() // 60
		return int(min(max(minutes, 0), self.span_min))

	def poll(self, finished=False):
		"""Progress(done, total, text), in request minutes over every pair."""
		landed = {pair: self._landed(pair) for pair in self.pairs}
		active = max((i for i, pair in enumerate(self.pairs) if landed[pair]), default=-1)
		done = 0
		for i, pair in enumerate(self.pairs):
			if finished or i < active:
				done += self.span_min
			elif landed[pair]:
				done += self._covered(max(stamp for stamp, _size in landed[pair]))
		total = self.span_min * len(self.pairs)
		n_files = sum(len(v) for v in landed.values())
		mb = sum(size for v in landed.values() for _stamp, size in v) / 1e6
		elapsed = _elapsed(self._clock() - self._t0)
		if finished:
			text = f'Download finished: {n_files} file(s), {mb:.1f} MB in {elapsed}'
		elif active < 0:
			text = f'Downloading from the SDC: waiting for the first file ...   {elapsed} elapsed'
		else:
			letter, probe = self.pairs[active]
			newest = max(stamp for stamp, _size in landed[(letter, probe)])
			through = max(newest, self.rng.start)
			text = (f'Downloading from the SDC: {n_files} file(s), {mb:.1f} MB   '
			        f'MMS{probe} {DATA_NAMES[letter]} through {ee.label_ts(through)[:16]} UTC   '
			        f'{elapsed} elapsed')
		return Progress(done, total, text)


def download_with_progress(rng, missing, report, poll_s=POLL_S, download=None, data_dir=None,
                           cancelled=None):
	"""Run download(rng, missing) -- event_export.download_missing by default --
	and call report(done, total, text) straight away, every `poll_s` seconds
	while it runs, and once more when it returns. Returns the final Progress.

	The download runs on a helper thread and this function waits on it, so it
	blocks exactly as long as a plain download_missing call would; an exception
	the download raises is re-raised here. `cancelled`, when given, is handed to
	the download, which stops between SDC calls once it returns True."""
	download = download or ee.download_missing
	watcher = LandingWatcher(rng, missing, data_dir=data_dir)
	failure = []

	def run():
		try:
			if cancelled is None:
				download(rng, missing)
			else:
				download(rng, missing, cancelled=cancelled)
		except BaseException as exc:
			failure.append(exc)

	worker = threading.Thread(target=run, name='sdc-download', daemon=True)
	report(*watcher.poll())
	worker.start()
	while True:
		worker.join(poll_s)
		if not worker.is_alive():
			break
		report(*watcher.poll())
	if failure:
		raise failure[0]
	final = watcher.poll(finished=True)
	report(*final)
	return final
