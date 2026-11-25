"""Post-process verification: every request's CSVs checked by an independent process (Graham, 2026-11-04).

  "Post-process verification should start acquisition during the acquisition stage of the whole request so that
  it can process the FFTs into CSVs then subsequently compare the CSVs to the actual output. The CSVs from
  post-process verification should be deleted along with the CDFs it uses."
  "The flagging and storing of differing CSVs should be produced in batches, with their own folder per request,
  and it should store all per request debugging/logging artifacts."

How a request goes (the event window is the caller; Session is its side):

  Find events   Session(rng, fields, probes) starts the verifier as its own process. It lists CDAWeb and downloads
                the request's B/E burst CDFs itself (post_verify_fetch: no code shared with the app's downloads)
                into a staging folder of its own, <cdf_dir>/_post_verify/<id>/stage, day by day in time order, at
                most max_staged_gb ahead. The app's archive never sees these files.
  Run           begin_run(top_dir, lpf_hz, events) names the request folder and the events; event_done() reports
                each event as the app writes it; finish() when the Run ends.
  verifying     a day is checked once every event that starts on it has been written: post_verify_method writes
                the verifier's own CSV for every event, every row recomputed (full verification: Graham, 2026-11-04,
                "Stick to full verification mode"), and compares it with the app's
  clean-up      then the day's verifier CSVs and the CDFs no later day needs are deleted

Per request, <top_dir>/verification/ (VERIFY_DIR) holds only (Graham, 2026-11-04: "I don't need a report per day,
only a report per request and whatever invalid CDFs/CSVs there are that get captured from differing rows"):
  report.json     the request's one report, rewritten as it goes (output_explorer.py reads it while the request
                  runs): state, every event's verdict, the conflicts with their groups of differing rows (first and
                  last times), totals, and the verifier's log
  differing/      only when something differs: one folder per differing event with the app's CSV (app.csv), the
                  verifier's (verifier.csv) and the CDFs it was computed from, written a day's batch at a time.
                  The CDFs are moved there when it is on the CDF folder's drive; otherwise they stay in
                  <cdf_dir>/_post_verify/<id>/kept/ and the report says where
A differing event never stops a request.

Settings (app_config 'post_verify'): enabled, max_staged_gb.

    python post_verify.py run <job dir>       the verifier process (Session starts it)
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta

PROJECT = os.path.dirname(os.path.abspath(__file__))
VERIFY_DIR = 'verification'
REPORT = 'report.json'
DIFFERING = 'differing'
JOBS_DIR = '_post_verify'
POLL_S = 0.5
EVENT_TIME = '%Y-%m-%dT%H:%M:%S.%f'

# Event states, as report.json and the explorer show them.
WAITING, EXPORTED, NOT_EXPORTED = 'waiting', 'exported', 'not exported'
CHECKED, MATCHES, DIFFERS = 'checked', 'matches', 'differs'          # from post_verify_method.verify_day
UNVERIFIABLE, NO_CSV, ERROR = 'unverifiable', 'no csv', 'error'


def settings(cfg=None):
	import app_config
	cfg = app_config.load() if cfg is None else cfg
	s = {**app_config.DEFAULTS.get('post_verify', {}), **(cfg.get('post_verify') or {})}
	s['coverage'] = 'full'                  # the only mode (Graham, 2026-11-04: "Stick to full verification mode")
	return s


def enabled(cfg=None):
	return bool(settings(cfg).get('enabled'))


# ---------------------------------------------------------------------------- small file helpers

def write_json(path, obj):
	"""Atomic: a reader (the explorer) sees the old file or the new one, never half of one."""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	tmp = f'{path}.{os.getpid()}.tmp'
	with open(tmp, 'w', encoding='utf-8') as fh:
		json.dump(obj, fh, indent=1, default=str)
	for attempt in range(20):
		try:
			os.replace(tmp, path)
			return
		except PermissionError:          # Windows: the explorer has the file open for a moment
			time.sleep(0.05)
	os.replace(tmp, path)


def read_json(path, default=None):
	try:
		with open(path, encoding='utf-8') as fh:
			return json.load(fh)
	except (OSError, ValueError):
		return default


def _alive(pid):
	if not pid:
		return True
	if os.name == 'nt':
		import ctypes
		k32 = ctypes.windll.kernel32
		h = k32.OpenProcess(0x00100000, False, int(pid))       # SYNCHRONIZE
		if not h:
			return False
		try:
			return k32.WaitForSingleObject(h, 0) == 0x102      # WAIT_TIMEOUT: still running
		finally:
			k32.CloseHandle(h)
	try:
		os.kill(int(pid), 0)
		return True
	except ProcessLookupError:
		return False
	except PermissionError:
		return True


def _event_json(ev, top_dir):
	import event_export as ee
	return {'probe': ev.probe, 'field': ev.field, 'start': ev.start.strftime(EVENT_TIME),
	        'end': ev.end.strftime(EVENT_TIME), 'clipped_start': bool(ev.clipped_start),
	        'clipped_end': bool(ev.clipped_end), 'csv': os.path.join(ee.event_dir(top_dir, ev), ee.CSV_NAME)}


# ---------------------------------------------------------------------------- the app's side

class Session:
	"""One request's verifier, from Find events (acquisition) to the end of its Run."""

	def __init__(self, rng, fields, probes, cfg=None, log=print, spawn=True, job_extra=None):
		"""spawn=False: the caller runs Verifier(job_dir).main() itself (the census, in a thread). job_extra: the
		census's options -- all_days (download every day, events or not) and gap_modes ([[field, mode]] also
		downloaded, for its day hook only)."""
		import app_config
		from ffttest import field_burst
		cfg = app_config.load() if cfg is None else cfg
		s = settings(cfg)
		self.key = (rng.start, rng.end, tuple(fields), tuple(probes))
		self.log = log
		self.used = False
		self.job_id = f'{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}'
		self.job_dir = os.path.join(app_config.cdf_dir(cfg), JOBS_DIR, self.job_id)
		os.makedirs(self.job_dir, exist_ok=True)
		bursts = {}
		for f in fields:
			inst, mode, fs = field_burst(f)
			bursts[f] = [inst, mode, float(fs)]
		import event_export as ee
		write_json(os.path.join(self.job_dir, 'job.json'), {
			'id': self.job_id, 'parent_pid': os.getpid(), 'start': rng.start.strftime(EVENT_TIME),
			'end': rng.end.strftime(EVENT_TIME), 'fields': list(fields), 'probes': list(probes), 'bursts': bursts,
			'pad': ee.PAD, 'step': ee.STEP, 'num_bins': ee.NUM_BINS, 'coverage': s['coverage'],
			'max_staged_gb': float(s.get('max_staged_gb', 20)), 'concurrent': app_config.download_concurrent(cfg),
			**(job_extra or {})})
		self.proc = None
		if spawn:
			# No output file: the verifier logs to run.log, and an open file could not be deleted with its job folder
			# on Windows. A crash before its log is open goes to crash.txt.
			flags = 0x08000000 if os.name == 'nt' else 0                # CREATE_NO_WINDOW
			self.proc = subprocess.Popen([sys.executable, os.path.join(PROJECT, 'post_verify.py'), 'run',
			                              self.job_dir], cwd=PROJECT, stdout=subprocess.DEVNULL,
			                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, creationflags=flags)
		log(f'Post-process verification: started; it downloads its own copies of the CDFs into {self.job_dir}')

	def matches(self, rng, fields, probes):
		return not self.used and self.key == (rng.start, rng.end, tuple(fields), tuple(probes))

	def begin_run(self, top_dir, lpf_hz, events):
		"""The Run's request folder and its events, in the order it writes them. Index i is event_done's."""
		self.used = True
		self.top_dir = top_dir
		write_json(os.path.join(self.job_dir, 'run.json'), {
			'top_dir': top_dir, 'verify_dir': os.path.join(top_dir, VERIFY_DIR), 'lpf_hz': lpf_hz,
			'events': [_event_json(ev, top_dir) for ev in events]})
		self.log(f'Post-process verification: results in {os.path.join(top_dir, VERIFY_DIR)}')

	def event_done(self, i, outcome):
		import event_export as ee
		written = outcome in (ee.WRITTEN, ee.WRITTEN_NO_FCE)
		with open(os.path.join(self.job_dir, 'done.jsonl'), 'ab') as fh:
			fh.write((json.dumps({'i': i, 'outcome': 'written' if written else str(outcome)}) + '\n').encode('utf-8'))

	def finish(self, cancelled=False):
		write_json(os.path.join(self.job_dir, 'finished.json'), {'cancelled': bool(cancelled)})

	def abandon(self):
		"""The request went no further (a new Find, or the window closed before Run): stop and delete."""
		if not self.used:
			open(os.path.join(self.job_dir, 'abandon'), 'w').close()


# ---------------------------------------------------------------------------- the verifier process

class Verifier:
	def __init__(self, job_dir, day_hook=None):
		"""day_hook(day, files): called once per day, after its check and before its CDFs are deleted, with the
		day's downloaded files [(probe, field, mode, path)] -- the census reads its gap rows from them."""
		self.job_dir = job_dir
		self.day_hook = day_hook
		self.job = read_json(os.path.join(job_dir, 'job.json'))
		self.stage = os.path.join(job_dir, 'stage')
		self.kept_dir = os.path.join(job_dir, 'kept')
		self.start = datetime.strptime(self.job['start'], EVENT_TIME)
		self.end = datetime.strptime(self.job['end'], EVENT_TIME)
		last = (self.end - timedelta(microseconds=1)).date()
		self.days = []
		d = self.start.date()
		while d <= last:
			self.days.append(d)
			d += timedelta(days=1)
		self.lock = threading.Lock()
		self.log_lines = []
		self.log_path = os.path.join(job_dir, 'run.log')
		self.run = None
		self.verify_dir = None
		self.done = {}                     # event index -> outcome
		self.done_pos = 0
		self.finished = None
		self.day_state = {d.isoformat(): 'queued' for d in self.days}
		self.units = {}                    # day -> [(url, dst, size)]
		self.staged = set()                # dst paths on disk
		self.keep = set()                  # dst paths kept as evidence
		self.pending_keep = []             # (staged path, where it is kept), moved once no later day needs it
		self.processed = set()
		self.events_status = {}
		self.conflicts = []
		self.batches = 0
		self.day_errors = {}
		self.totals = {'days_checked': 0, 'csvs': 0, 'rows': 0, 'rows_recomputed': 0, 'events_recomputed': 0,
		               'differing_events': 0, 'differing_rows': 0, 'unverifiable': 0, 'downloaded_bytes': 0,
		               'download_failures': 0, 'max_rel_err': 0.0}
		self.state = 'acquiring'
		self.message = ''
		self.stop = threading.Event()

	# -- logging and status

	def log(self, msg):
		line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {msg}'
		print(line, flush=True)
		with self.lock:
			self.log_lines.append(line)
			path = self.log_path
		with open(path, 'a', encoding='utf-8') as fh:
			fh.write(line + '\n')

	def staged_bytes(self):
		total = 0
		for p in list(self.staged):
			try:
				total += os.path.getsize(p)
			except OSError:
				pass
		return total

	def write_status(self):
		"""The request's one report: in its verification folder once Run has named it, in the job folder before."""
		write_json(os.path.join(self.verify_dir or self.job_dir, REPORT), self._status())

	def _status(self):
		with self.lock:
			return {'state': self.state, 'message': self.message, 'updated': datetime.now().isoformat(timespec='seconds'),
			        'job_dir': self.job_dir, 'coverage': self.job['coverage'], 'range': [self.job['start'], self.job['end']],
			        'days': dict(self.day_state), 'staged_gb': round(self.staged_bytes() / 1e9, 2),
			        'events': dict(self.events_status), 'totals': dict(self.totals),
			        'conflicts': list(self.conflicts), 'batches': self.batches, 'day_errors': dict(self.day_errors),
			        'log': list(self.log_lines)}

	# -- the app's messages

	def poll_app(self):
		if self.run is None:
			run = read_json(os.path.join(self.job_dir, 'run.json'))
			if run:
				self.run = run
				for e in run['events']:
					e['start_dt'] = datetime.strptime(e['start'], EVENT_TIME)
					e['end_dt'] = datetime.strptime(e['end'], EVENT_TIME)
				self.verify_dir = run['verify_dir']
				os.makedirs(self.verify_dir, exist_ok=True)
				try:
					os.remove(os.path.join(self.job_dir, REPORT))
				except OSError:
					pass
				import post_verify_method as m
				with self.lock:
					for e in run['events']:
						key = m.event_key({**e}, run['top_dir'])
						e['key'] = key
						self.events_status[key] = {'state': WAITING, 'probe': e['probe'], 'field': e['field'],
						                           'start': e['start'], 'end': e['end']}
				days_with = {e['start_dt'].date() for e in run['events']}
				for d in self.days:
					if (d not in days_with and d.isoformat() not in self.processed
					        and not self.job.get('all_days')):
						self.day_state[d.isoformat()] = 'no events'
				self.extend_units()
				self.log(f'run: {len(run["events"])} event(s) into {run["top_dir"]}')
		if self.run is not None:
			path = os.path.join(self.job_dir, 'done.jsonl')
			try:
				with open(path, 'rb') as fh:                   # bytes: the offset must be a byte offset
					fh.seek(self.done_pos)
					chunk = fh.read()
			except OSError:
				chunk = b''
			if chunk:
				complete = chunk[:chunk.rfind(b'\n') + 1]     # a line still being written waits for the next poll
				self.done_pos += len(complete)
				for line in complete.decode('utf-8').splitlines():
					if not line.strip():
						continue
					d = json.loads(line)
					self.done[d['i']] = d['outcome']
					e = self.run['events'][d['i']]
					with self.lock:
						self.events_status[e['key']]['state'] = EXPORTED if d['outcome'] == 'written' else NO_CSV
			if self.finished is None:
				self.finished = read_json(os.path.join(self.job_dir, 'finished.json'))

	def abandoned(self):
		if os.path.exists(os.path.join(self.job_dir, 'abandon')):
			return 'abandoned by the app'
		if not _alive(self.job.get('parent_pid')):
			if self.run is None:
				return 'the app closed before Run'
			if self.finished is None:
				self.finished = {'cancelled': True, 'why': 'the app closed during the Run'}
		return None

	# -- what to download

	def list_files(self):
		self.listings = {}
		t0, t1 = self.start - timedelta(days=1), self.end + timedelta(days=1)
		import post_verify_fetch as pf
		sets = [(field, self.job['bursts'][field][1]) for field in self.job['fields']]
		sets += [tuple(x) for x in self.job.get('gap_modes', []) if tuple(x) not in sets]
		for probe in self.job['probes']:
			for field, mode in sets:
				files = pf.listing(probe, field, mode, t0, t1)
				self.listings[(probe, field, mode)] = files
				self.log(f'listed MMS{probe} {field} ({pf.INSTRUMENT[field]}/{mode}): {len(files)} file(s), '
				         f'{sum(f["size"] for f in files) / 1e9:.2f} GB from {t0:%Y-%m-%d} to {t1:%Y-%m-%d}')
		self.extend_units()

	def extend_units(self):
		"""Each day's files: those starting inside the request on that day, until the day ends (or its last event
		ends), plus the two files before and the one after, so every file has its neighbours."""
		if not hasattr(self, 'listings'):
			return
		import post_verify_fetch as pf
		with self.lock:
			for d in self.days:
				lo = datetime.combine(d, datetime.min.time())
				hi = min(lo + timedelta(days=1), self.end)
				lo = max(lo, self.start)
				if self.run is not None:
					ends = [e['end_dt'] for e in self.run['events'] if e['start_dt'].date() == d]
					hi = max([hi] + ends)
				unit = []
				for (probe, field, mode), files in self.listings.items():
					idx = [i for i, f in enumerate(files) if lo <= f['stamp'] < hi]
					if not idx:
						before = [i for i, f in enumerate(files) if f['stamp'] < lo]
						after = [i for i, f in enumerate(files) if f['stamp'] >= hi]
						idx = before[-2:] + after[:1]               # a request inside one file: it and its neighbours
					else:
						idx = list(range(max(idx[0] - 2, 0), min(idx[-1] + 2, len(files))))
					for i in idx:
						f = files[i]
						unit.append((f['url'], pf.local_path(self.stage, probe, field, mode, f), f['size']))
				self.units[d.isoformat()] = unit

	def needs_download(self, day):
		return self.day_state.get(day) not in ('no events', 'verified', 'skipped')

	# -- the download thread

	def downloader(self):
		import post_verify_fetch as pf
		cap = self.job['max_staged_gb'] * 1e9
		for d in self.days:
			day = d.isoformat()
			while not self.stop.is_set():
				if not self.needs_download(day):
					break
				with self.lock:
					unit = list(self.units.get(day, []))
				new = [j for j in unit if j[1] not in self.staged]
				need = sum(j[2] for j in new)
				earliest = min((x.isoformat() for x in self.days if x.isoformat() not in self.processed
				                and self.needs_download(x.isoformat())), default=day)
				if day == earliest or self.staged_bytes() + need <= cap:
					break
				time.sleep(POLL_S)
			if self.stop.is_set():
				return
			if not self.needs_download(day):
				continue
			with self.lock:
				unit = list(self.units.get(day, []))
				self.day_state[day] = 'downloading'
			failed = pf.download(unit, concurrent=self.job.get('concurrent', 8), cancelled=self.stop.is_set,
			                     log=self.log)
			got = 0
			for _url, dst, size in unit:
				if dst not in failed and os.path.exists(dst):
					if dst not in self.staged:
						got += size
					self.staged.add(dst)
			with self.lock:
				self.totals['downloaded_bytes'] += got
				self.totals['download_failures'] += len(failed)
				if self.day_state[day] == 'downloading':
					self.day_state[day] = 'download failed' if failed else 'staged'
			self.log(f'{day}: {len(unit)} file(s) staged ({got / 1e9:.2f} GB new), {len(failed)} failed')

	# -- checking a day

	def day_ready(self, day):
		if self.run is None or self.day_state.get(day) not in ('staged', 'download failed'):
			return False
		idx = [i for i, e in enumerate(self.run['events']) if e['start_dt'].date().isoformat() == day]
		if not (all(i in self.done for i in idx) or self.finished is not None):
			return False
		with self.lock:
			missing = [j for j in self.units.get(day, []) if j[1] not in self.staged]
		if missing and self.day_state[day] == 'staged':
			# the day's files grew once the Run named its events (an event running past midnight)
			import post_verify_fetch as pf
			failed = pf.download(missing, concurrent=self.job.get('concurrent', 8), log=self.log)
			for _u, dst, size in missing:
				if dst not in failed and os.path.exists(dst):
					self.staged.add(dst)
					self.totals['downloaded_bytes'] += size
			if failed:
				self.day_state[day] = 'download failed'
		return True

	def check_day(self, day):
		import post_verify_method as m
		run = self.run
		evs = []
		for i, e in enumerate(run['events']):
			if e['start_dt'].date().isoformat() != day:
				continue
			if i not in self.done:
				with self.lock:
					self.events_status[e['key']]['state'] = NOT_EXPORTED
				continue
			evs.append({'key': e['key'], 'probe': e['probe'], 'field': e['field'], 'start': e['start_dt'],
			            'end': e['end_dt'], 'clipped_start': e['clipped_start'], 'clipped_end': e['clipped_end'],
			            'outcome': self.done[i], 'csv': e['csv'] if self.done[i] == 'written' else None})
		if self.day_state[day] == 'download failed':
			with self.lock:
				for e in evs:
					st = self.events_status[e['key']]
					st['state'] = UNVERIFIABLE if e['csv'] else NO_CSV
					st['issues'] = ['its CDFs could not all be downloaded for the check']
				self.totals['unverifiable'] += sum(1 for e in evs if e['csv'])
				self.day_state[day] = 'verified'
			self.log(f'{day}: not checked, some of its CDFs could not be downloaded; {len(evs)} event(s) unverifiable')
			return
		with self.lock:
			self.day_state[day] = 'verifying'
		self.write_status()
		work = os.path.join(self.job_dir, 'work', day)
		bursts = {f: tuple(v) for f, v in self.job['bursts'].items()}
		try:
			rep = m.verify_day(day, evs, run['top_dir'], self.stage, run['lpf_hz'], (self.start, self.end),
			                   pad=self.job['pad'], step=self.job['step'], num_bins=self.job['num_bins'],
			                   bursts=bursts, mode=self.job['coverage'], work_dir=work)
		except Exception:
			tb = traceback.format_exc()
			self.log(f'{day}: the check itself raised; its events are left unjudged\n{tb}')
			with self.lock:
				self.day_state[day] = 'error'
				for e in evs:
					self.events_status[e['key']]['state'] = ERROR
				self.day_errors[day] = tb.strip().splitlines()[-1]
			shutil.rmtree(work, ignore_errors=True)
			return
		differing = {k: r for k, r in rep['events'].items() if r['state'] == DIFFERS}
		if differing:
			self.store_batch(day, rep, differing)
		with self.lock:
			for k, r in rep['events'].items():
				st = self.events_status.setdefault(k, {})
				st.update(state=r['state'])
				if 'compare' in r:
					st['rows_compared'] = r['compare']['rows_compared']
					st['differing_rows'] = r['compare']['differing']
				if r.get('issues'):
					st['issues'] = r['issues']
			self.totals['days_checked'] += 1
			self.totals['csvs'] += rep['csvs']
			self.totals['rows'] += rep['rows']
			self.totals['rows_recomputed'] += rep['rows_recomputed']
			self.totals['events_recomputed'] += len(rep['sampled'])
			self.totals['unverifiable'] += len(rep['unverifiable'])
			self.totals['max_rel_err'] = max(self.totals['max_rel_err'], rep['max_rel_err'])
			self.totals['differing_events'] += len(differing)
			self.totals['differing_rows'] += sum(r.get('compare', {}).get('differing', 0) for r in differing.values())
			self.day_state[day] = 'verified'
		shutil.rmtree(work, ignore_errors=True)
		self.log(f'{day}: {rep["csvs"]} CSV(s), {len(rep["sampled"])} recomputed ({rep["rows_recomputed"]} rows), '
		         f'{len(differing)} differing, {len(rep["unverifiable"])} unverifiable, max relative error '
		         f'{rep["max_rel_err"]:.2e}, {rep["seconds"]} s')

	def run_hook(self, day):
		if self.day_hook is None:
			return
		import post_verify_fetch as pf
		files = []
		with self.lock:
			unit = list(self.units.get(day, []))
		for (probe, field, mode), listed in self.listings.items():
			for f in listed:
				p = pf.local_path(self.stage, probe, field, mode, f)
				if any(p == j[1] for j in unit) and p in self.staged:
					files.append((probe, field, mode, p))
		try:
			self.day_hook(day, files)
		except Exception:
			tb = traceback.format_exc()
			self.log(f'{day}: the day hook raised\n{tb}')
			with self.lock:
				self.day_errors[day] = 'day hook: ' + tb.strip().splitlines()[-1]

	def store_batch(self, day, rep, differing):
		"""A checked day's differing events, as one batch: each one's two CSVs and its CDFs into
		verification/differing/<event>/, and its groups of differing rows into the report."""
		self.batches += 1
		same_drive = (os.path.splitdrive(os.path.abspath(self.verify_dir))[0].lower()
		              == os.path.splitdrive(os.path.abspath(self.stage))[0].lower())
		entries = []
		for key, r in differing.items():
			sub = os.path.join(self.verify_dir, DIFFERING, key.replace('/', '_').replace(':', '-'))
			os.makedirs(sub, exist_ok=True)
			if r.get('csv') and os.path.exists(r['csv']):
				shutil.copy2(r['csv'], os.path.join(sub, 'app.csv'))
			if r.get('verifier_csv') and os.path.exists(r['verifier_csv']):
				shutil.move(r['verifier_csv'], os.path.join(sub, 'verifier.csv'))
			kept = []
			for p in r.get('cdfs', []):
				self.keep.add(p)
				dst = (os.path.join(sub, os.path.basename(p)) if same_drive
				       else os.path.join(self.kept_dir, os.path.basename(p)))
				kept.append(dst)
				self.pending_keep.append((p, dst))
			entries.append({'event': key, 'day': day, 'batch': self.batches,
			                'folder': os.path.relpath(sub, self.verify_dir).replace('\\', '/'),
			                'groups': r.get('compare', {}).get('groups', []), 'issues': r.get('issues', []),
			                'rows_compared': r.get('compare', {}).get('rows_compared'), 'cdfs': kept})
		with self.lock:
			self.conflicts.extend(entries)
		self.log(f'{day}: batch {self.batches}: {len(entries)} differing event(s) stored in '
		         f'{os.path.join(self.verify_dir, DIFFERING)}')

	def clean(self, final=False):
		"""Delete the staged CDFs no later day needs; move kept ones to their batch."""
		with self.lock:
			needed = set()
			if not final:
				for d in self.days:
					day = d.isoformat()
					if day not in self.processed and self.needs_download(day):
						needed.update(j[1] for j in self.units.get(day, []))
		for src, dst in self.pending_keep:
			if src in needed or not os.path.exists(src):
				continue
			os.makedirs(os.path.dirname(dst), exist_ok=True)
			shutil.move(src, dst)
			self.staged.discard(src)
		self.pending_keep = [(s, d) for s, d in self.pending_keep if os.path.exists(s)]
		for p in list(self.staged):
			if p in needed or p in self.keep:
				continue
			try:
				os.remove(p)
			except OSError:
				pass
			self.staged.discard(p)
		if final:
			shutil.rmtree(self.stage, ignore_errors=True)

	# -- the loop

	def run_safely(self):
		"""main(), but a verifier that breaks says so where the explorer looks, and gives the space back."""
		try:
			return self.main()
		except Exception:
			tb = traceback.format_exc()
			self.stop.set()
			try:
				self.log(f'the verifier stopped on an error; its staging is deleted\n{tb}')
				with self.lock:
					self.state, self.message = ERROR, tb.strip().splitlines()[-1]
				self.write_status()
			finally:
				shutil.rmtree(self.stage, ignore_errors=True)
				shutil.rmtree(os.path.join(self.job_dir, 'work'), ignore_errors=True)
			return 1

	def main(self):
		self.log(f'post-process verification {self.job["id"]}: {self.job["start"]} to {self.job["end"]}, '
		         f'{"".join(self.job["fields"])}, MMS{",".join(map(str, self.job["probes"]))}, every row, '
		         f'at most {self.job["max_staged_gb"]:g} GB staged')
		try:
			self.list_files()
		except Exception as exc:  # noqa: BLE001
			self.state, self.message = ERROR, f'CDAWeb listing failed: {exc}'
			self.log(self.message)
		dl = threading.Thread(target=self.downloader, daemon=True) if self.state != ERROR else None
		if dl:
			dl.start()
		while True:
			why = self.abandoned()
			if why:
				self.log(f'stopped: {why}; staging deleted')
				self.stop.set()
				if dl:
					dl.join(timeout=60)
				shutil.rmtree(self.job_dir, ignore_errors=True)
				return 0
			self.poll_app()
			if self.state == ERROR:
				if self.run is not None:
					self.write_status()
					if self.finished is not None:
						break
				time.sleep(POLL_S)
				continue
			busy = False
			for d in self.days:
				day = d.isoformat()
				if day in self.processed:
					continue
				if not self.needs_download(day):
					self.processed.add(day)
					self.clean()
					continue
				if self.day_ready(day):
					self.state = 'verifying'
					self.check_day(day)
					self.run_hook(day)
					self.processed.add(day)
					self.clean()
					busy = True
				break                             # days are checked in order
			if self.run is not None:
				left = [d for d in self.days if d.isoformat() not in self.processed]
				if not left and self.finished is not None:
					break
				if not busy:
					self.state = 'verifying' if any(self.day_state[d.isoformat()] in ('staged', 'verifying')
					                                 for d in left) else 'acquiring'
			else:
				self.state = 'acquiring (waiting for Run)'
			self.write_status()
			if not busy:
				time.sleep(POLL_S)
		self.stop.set()
		if dl:
			dl.join(timeout=60)
		self.clean(final=True)
		with self.lock:
			for st in self.events_status.values():
				if st.get('state') in (WAITING, EXPORTED):
					st['state'] = ERROR if self.state == ERROR else NOT_EXPORTED
			self.state = 'done' if self.state != ERROR else ERROR
			t = self.totals
			self.message = (f'{t["days_checked"]} day(s) checked: {t["csvs"]} CSV(s), {t["events_recomputed"]} '
			                f'event(s) recomputed, {t["differing_events"]} differing')
		self.log(f'done: {self.message}')
		self.write_status()
		shutil.rmtree(os.path.join(self.job_dir, 'work'), ignore_errors=True)
		if os.path.isdir(self.kept_dir):           # evidence that could not go to the request's drive stays here
			for name in os.listdir(self.job_dir):
				if name != 'kept':
					p = os.path.join(self.job_dir, name)
					shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
		else:
			shutil.rmtree(self.job_dir, ignore_errors=True)
		return 0


def main(argv):
	if len(argv) == 3 and argv[1] == 'run':
		sys.path.insert(0, PROJECT)
		try:
			v = Verifier(argv[2])
		except Exception:
			with open(os.path.join(argv[2], 'crash.txt'), 'w', encoding='utf-8') as fh:
				fh.write(traceback.format_exc())
			return 1
		return v.run_safely()
	print(__doc__)
	return 2


if __name__ == '__main__':
	sys.exit(main(sys.argv))
