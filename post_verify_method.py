r"""Independent check of the event export's CSVs: the method of post-process verification (post_verify.py).

First written for the census as _wsl_scratch/csv_verify.py (Graham, 2026-10-07: "every chunk you should be verifying
that the CSVs are accurate and if they are not, cancel the census"), moved into the app on 2026-11-04 (Graham:
"Post-process verification should start acquisition during the acquisition stage of the whole request so that it can
process the FFTs into CSVs then subsequently compare the CSVs to the actual output"). csv_verify.py now stands for
this module. Given work_dir, verify_day writes its own CSV for every event it recomputes (verifier.csv: every row it
expects, cells where it recomputed them) and compares that file with the app's (compare_csvs), which groups the
rows that differ into runs with their first and last times. mode 'full' recomputes every row of every event.

It shares no code with the app's FFT path: nothing is imported from ffttest, short_interval, event_export,
fft_excel_export or spectro. From the raw CDFs and the method as documented it rebuilds what a CSV must hold:

  samples   the quality-good samples (SCM qf 'GGG'; EDP quality >= 2), times TT2000 -> UTC rounded to the
            microsecond -- checked against pycdf's own conversion on random samples of every file it reads
  windows   centres every 0.05 s on each file's own grid, from max(its stamp, first good sample - pad), until
            min(the next file's stamp, last good sample + pad + step); a window [c - pad, c + pad] is computed only
            when it holds at least n_full good samples, n_full = min(fs * 2 pad, (2 pad) // mean step) with the
            mean over the file's steps no wider than 1.5 periods
  PSD       |B| (|E|) of the window with its mean removed, symmetric Hann window, one-sided density:
            scipy.signal.periodogram, an implementation the app does not use
  channels  frequencies snapped to the 2 Hz grid, DC folded into 1 Hz, none above the ceiling; log edges from
            (pad, ceiling, NUM_BINS) by the documented walk, which must equal the request's bins README
  cells     the mean of every PSD value whose frequency lands in the channel and whose window centre lands in the
            row's time cell (floor((centre - start) / dt + 1e-9)), blank where nothing landed; a row with nothing
            in it is not written; then max_power and min_power rows
  events    an unclipped event starts and ends on a good sample that counts, with no sample that counts within 1.5 s
            outside it. A good sample counts when its run holds at least MIN_RUN_SAMPLES (10) good samples; a run
            ends wherever a good sample is missing (the next sits over 1.5 periods on) and is counted across file
            seams. Graham, 2026-10-09: "A lone good sample should never count since we're doing math that relies on
            continuous data. Single digit numbers of samples do not count as events"

Per day:
  every CSV     structure: the header, row labels on the event's grid, inside it and increasing, every cell of a
                row a positive finite number, the max/min rows
  sampled CSVs  one event per (probe, field) present that day plus the day's longest, picked by a generator
                seeded with the day. Every row whose window cannot borrow from a neighbouring file is recomputed
                and compared cell by cell (relative tolerance REL_TOL; up to MAX_VALUE_ROWS rows per event, at
                random); every slot of the event must have a row exactly when its window holds n_full good samples;
                the event's edges are checked.
A slot whose window is short and could borrow samples from the next or previous file is a seam: counted, not
recomputed. A stamp on disk in two versions, or a leap second inside a file, makes an event unverifiable
(counted). Neither is a failure; a mismatch is.

    verify_day(day, events, top_dir, data_dir, lpf_hz, rng) -> report;  report['passed'] is False on any mismatch
"""
import csv
import json
import math
import os
import random
import re
import time
from datetime import datetime, timedelta

import numpy as np

REL_TOL = 1e-4
ABS_FLOOR = 1e-12             # of the row's strongest cell: below it a difference is rounding noise (counted)
EVENT_GAP_S = 1.5             # one event gap everywhere (ffttest.SEGMENT_GAP_SECONDS)
JITTER_PERIODS = 1.5          # steps wider than this many periods are holes, not cadence
MIN_RUN_SAMPLES = 10          # a run of fewer good samples never counts toward an event (Graham, 2026-10-09)
RUN_GAP_PERIODS = 1.5         # a run ends where a good sample is missing: the next sits over this many periods on
MAX_DISPLAY_COLS = 200_000    # a longer event's cells widen past the step (short_interval)
MAX_VALUE_ROWS = 2000
CONVERSION_PROBES = 32
NAME = re.compile(r'^mms([1-4])_(scm|edp)_brst_l2_(scb|schb|dce)_(\d{14})_v([\d.]+)\.cdf$')
LABEL = re.compile(r'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d\d$')
EPOCH = datetime(1970, 1, 1)
DEFAULT_BURSTS = {'B': ('scm', 'scb', 8192.0), 'E': ('edp', 'dce', 8192.0)}


class Unverifiable(Exception):
	pass


# ---------------------------------------------------------------------------- method pieces

def log_edges(pad, lpf_hz, num_bins):
	"""Channel edges in Hz: the grid frequencies (multiples of 1/(2 pad), 0 shown as 1) walked down from the top,
	each step dividing the index by lpf ** (1 / num_bins)."""
	df = 1.0 / (2 * pad)
	grid = sorted({int(round(k * df)) for k in range(int(lpf_hz / df) + 1)})
	grid = [f for f in grid if 0 <= f <= lpf_hz]
	grid = sorted({1 if f == 0 else f for f in grid})
	factor = lpf_hz ** (1.0 / num_bins)
	h, out = len(grid) - 1, []
	while grid[h] >= df and h > 0:
		out.append(grid[h])
		h = int(h // factor)
	out.append(grid[0])
	return out[::-1]


def readme_edges(path):
	with open(path, encoding='utf-8') as fh:
		m = re.search(r'Bin edges \(Hz\): \[([\d, ]+)\]', fh.read())
	return [int(x) for x in m.group(1).split(',')] if m else None


def label_of(start, seconds):
	t = start + timedelta(seconds=float(seconds))
	return f'{t:%Y-%m-%d %H:%M:%S}.{t.microsecond // 10_000:02d}'


def ns_of(dt):
	return (dt - EPOCH) // timedelta(microseconds=1) * 1000


def cell_width(start, end, step):
	span = (end - start).total_seconds()
	dt = max(step, span / MAX_DISPLAY_COLS) if span > 0 else step
	return dt, int(math.floor(span / dt + 1e-9)) + 1


def psd(window_xyz, fs):
	"""One window's one-sided density PSD of the magnitude, mean removed: (freqs, power).

	The magnitude and its mean are taken in float32, the CDF's own precision, adding the squares z, y, x in that
	order as the pipeline does. That fixes the input to the last bit, which the DC channel needs: with the mean
	removed it is a near-cancellation, and one float32 ulp of difference in |B| moved it by up to 30% (measured,
	2026-10-07). Every other channel agrees to ~1e-6 either way."""
	from scipy.signal import periodogram
	x, y, z = window_xyz
	mag = np.sqrt(z * z + y * y + x * x)
	mag = mag - np.mean(mag)
	n = len(mag)
	return periodogram(mag.astype(np.float64), fs=fs, window=np.hanning(n), detrend=False,
	                   return_onesided=True, scaling='density')


def channel_means(fft_list, edges, lpf_hz, pad):
	"""Mean power per channel over every (frequency, power) of the FFTs given."""
	df = 1.0 / (2 * pad)
	sums, counts = np.zeros(len(edges)), np.zeros(len(edges))
	for f, p in fft_list:
		snapped = (np.round(f / df) * df).astype(int)
		keep = (snapped >= 0) & (snapped <= lpf_hz)
		snapped, p = snapped[keep], p[keep]
		snapped[snapped == 0] = 1
		ch = np.searchsorted(edges, snapped, side='left')
		ok = ch < len(edges)
		sums += np.bincount(ch[ok], weights=p[ok], minlength=len(edges))
		counts += np.bincount(ch[ok], minlength=len(edges))
	return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


# ---------------------------------------------------------------------------- reading the CDFs

def variables(field, probe, mode):
	if field == 'B':
		return 'Epoch', f'mms{probe}_scm_acb_gse_{mode}_brst_l2', f'mms{probe}_scm_qf_scm123_{mode}_brst_l2'
	return (f'mms{probe}_edp_epoch_brst_l2', f'mms{probe}_edp_dce_gse_brst_l2',
	        f'mms{probe}_edp_quality_brst_l2')


def utc_ns(cdf, epoch_var, rng):
	"""Every sample's UTC time in int64 ns, rounded to the microsecond as pycdf rounds (half up). Between leap
	seconds UTC is TT2000 plus a constant, a whole number of milliseconds (J2000 is 11:58:55.816 UTC); pycdf's
	conversion of the first and the last sample gives it, and pycdf's conversion of random samples checks the
	result."""
	from spacepy import pycdf
	raw = np.asarray(cdf.raw_var(epoch_var)[...], dtype=np.int64)
	if raw.size == 0:
		return raw

	def offset(i):
		return int(round((ns_of(pycdf.lib.tt2000_to_datetime(int(raw[i]))) - int(raw[i])) / 1e6)) * 1_000_000

	k0, k1 = offset(0), offset(raw.size - 1)
	if k0 != k1:
		raise Unverifiable('a leap second inside the file')
	out = ((raw + k0 + 500) // 1000) * 1000
	for i in rng.sample(range(raw.size), min(CONVERSION_PROBES, raw.size)):
		if ns_of(pycdf.lib.tt2000_to_datetime(int(raw[i]))) != out[i]:
			raise AssertionError(f'time conversion disagrees with pycdf at record {i}')
	return out


class FileData:
	"""One CDF's good samples: times (int64 ns), x, y, z (float32), plus its stamp."""

	def __init__(self, path, field, probe, mode, rng):
		from spacepy import pycdf
		self.path, self.name = path, os.path.basename(path)
		self.stamp_ns = ns_of(datetime.strptime(NAME.match(self.name).group(4), '%Y%m%d%H%M%S'))
		epoch_var, data_var, q_var = variables(field, probe, mode)
		with pycdf.CDF(path) as cdf:
			t = utc_ns(cdf, epoch_var, rng)
			if field == 'B':
				good = np.asarray(cdf.raw_var(q_var)[...]) == b'GGG'
			else:
				good = np.asarray(cdf[q_var][...]) >= 2
			data = np.asarray(cdf[data_var][...])
		self.t = t[good]
		self.xyz = data[good].astype(np.float32, copy=False)
		self.n_good = int(good.sum())


def day_listing(data_dir, inst, mode, probe, first_day, last_day):
	"""{stamp_ns: [paths]} of the CDFs on disk for one probe and mode, first_day..last_day."""
	out, day = {}, first_day
	while day <= last_day:
		folder = os.path.join(data_dir, f'mms{probe}', inst, 'brst', 'l2', mode, f'{day:%Y}', f'{day:%m}', f'{day:%d}')
		try:
			names = os.listdir(folder)
		except FileNotFoundError:
			names = []
		for name in names:
			m = NAME.match(name)
			if m and m.group(3) == mode and int(m.group(1)) == probe:
				stamp = ns_of(datetime.strptime(m.group(4), '%Y%m%d%H%M%S'))
				out.setdefault(stamp, []).append(os.path.join(folder, name))
		day += timedelta(days=1)
	return out


# ---------------------------------------------------------------------------- one CSV

def read_csv(path):
	"""(header, rows) in the tall layout. A wide CSV (one column per time cell, the bins as rows, max_power and
	min_power as the last two columns) is turned to the tall layout first."""
	with open(path, newline='') as fh:
		rows = list(csv.reader(fh))
	header, body = rows[0], rows[1:]
	if header and header[0] != 'time_utc' and header[-2:] == ['max_power', 'min_power']:
		labels = header[1:]
		names = [r[0] for r in body]
		tall = [[labels[j]] + [r[1 + j] for r in body] for j in range(len(labels))]
		return ['time_utc'] + names, tall
	return header, body


def structure(ev, path, edges, step):
	"""Structural problems of one CSV ([] when none), and its rows as {cell index: [floats]}."""
	problems = []
	header, rows = read_csv(path)
	n_chan = len(edges)
	if header != ['time_utc'] + [f'bin_{i}' for i in range(n_chan)]:
		return [f'header {header[:3]}... is not time_utc + bin_0..bin_{n_chan - 1}'], {}
	if len(rows) < 3 or [r[0] for r in rows[-2:]] != ['max_power', 'min_power']:
		return ['no data rows, or the max_power/min_power rows are missing'], {}
	dt, n_time = cell_width(ev['start'], ev['end'], step)
	cells, last_k = {}, -1
	for r in rows[:-2]:
		if len(r) != n_chan + 1 or not LABEL.match(r[0]):
			problems.append(f'malformed row {r[:2]}')
			continue
		label_t = datetime.strptime(r[0], '%Y-%m-%d %H:%M:%S.%f')
		k = int(round(((label_t - ev['start']).total_seconds() + 0.005) / dt))
		if not 0 <= k < n_time or label_of(ev['start'], k * dt) != r[0]:
			problems.append(f'row {r[0]} is not on the event\'s {dt:g} s grid inside it')
			continue
		if k <= last_k:
			problems.append(f'row {r[0]} does not come after the row before it')
			continue
		last_k = k
		try:
			vals = [float(v) for v in r[1:]]
		except ValueError:
			problems.append(f'row {r[0]} has a blank or non-number cell')
			continue
		if not all(math.isfinite(v) and v > 0 for v in vals):
			problems.append(f'row {r[0]} has a cell that is not a positive finite number')
			continue
		cells[k] = vals
	if cells:
		cols = np.array(list(cells.values()))
		for name, want in (('max_power', cols.max(axis=0)), ('min_power', cols.min(axis=0))):
			got = next(r for r in rows[-2:] if r[0] == name)
			try:
				got = np.array([float(v) for v in got[1:]])
			except ValueError:
				problems.append(f'{name} row has a blank or non-number cell')
				continue
			if got.shape != want.shape or not np.allclose(got, want, rtol=1e-9, atol=0):
				problems.append(f'{name} row is not the column {name[:3]}imum')
	return problems, cells


def recompute(ev, cells, files, edges, lpf_hz, pad, step, fs, rnd, out, earlier_unloaded=False, collect=None,
              max_rows=MAX_VALUE_ROWS):
	"""Compare one event's CSV rows with the recomputation. Appends to out['failures']; fills counters.

	files: {stamp_ns: FileData} in stamp order, from the file before the event's first through the file after its
	last, so every file the event touches has its neighbours. earlier_unloaded: the listing has files before the
	first of them (whose borrowing is then unknown; that first file's centres lie before the event anyway).

	Which windows exist is decided for every slot, borrowing included: a window short of n_full takes the previous
	file's samples in [c - pad, first sample) when it starts before this file's first sample (and there is a
	previous file), then the next file's samples after this file's last one up to c + pad when it reaches the next
	file's stamp, and exists when that fills it. A window that borrowed is not recomputed (a seam row).

	collect: a dict to fill instead of comparing ('unknown': cells whose rows cannot be decided, 'exists': cells
	that must have a row, 'values': {cell: recomputed channel means}); post-process verification writes it out as
	its own CSV and compares the files (compare_csvs). max_rows: None recomputes every row that can be."""
	start_ns, end_ns = ns_of(ev['start']), ns_of(ev['end'])
	pad_ns, step_ns = int(pad * 1e9), int(step * 1e9)
	n_expected = int(fs * 2 * pad)
	dt, n_time = cell_width(ev['start'], ev['end'], step)
	stamps = sorted(files)
	by_cell = {}                                   # cell -> [(file, lo, hi, exists, borrowed, unknown)]
	for i, stamp in enumerate(stamps):
		fd = files[stamp]
		if fd.t.size == 0:
			continue
		prev = files[stamps[i - 1]] if i > 0 else None
		nxt = files[stamps[i + 1]] if i + 1 < len(stamps) else None
		d = np.diff(fd.t)
		d = d[d <= JITTER_PERIODS * 1e9 / fs]
		n_full = min(n_expected, int((2 * pad_ns) // d.mean())) if d.size else n_expected
		s = max(fd.stamp_ns, int(fd.t[0]) - pad_ns)
		e = int(fd.t[-1]) + pad_ns + step_ns
		if nxt is not None:
			e = min(nxt.stamp_ns, e)
		j0 = max(0, -(-(start_ns - s) // step_ns))
		j1 = (min(e, end_ns + 1) - s + step_ns - 1) // step_ns
		if j1 <= j0:
			continue
		c = s + np.arange(j0, j1, dtype=np.int64) * step_ns
		c = c[(c >= start_ns) & (c <= end_ns) & (c < e)]
		if c.size == 0:
			continue
		lo = np.searchsorted(fd.t, c - pad_ns, side='left')
		hi = np.searchsorted(fd.t, c + pad_ns, side='right')
		n = hi - lo
		head = c - pad_ns < fd.t[0]
		back = np.zeros(c.size, dtype=np.int64)
		unknown = np.zeros(c.size, dtype=bool)
		if prev is not None and prev.t.size:
			back = np.where(head, np.searchsorted(prev.t, fd.t[0], side='left')
			                - np.searchsorted(prev.t, c - pad_ns, side='left'), 0)
		elif prev is None and i == 0 and earlier_unloaded:
			unknown = head & (n < n_full)
		fwd = np.zeros(c.size, dtype=np.int64)
		if nxt is not None and nxt.t.size:
			last_in = fd.t[np.maximum(hi - 1, 0)]
			reach = nxt.stamp_ns <= c + pad_ns
			fwd = np.where(reach, np.searchsorted(nxt.t, c + pad_ns, side='right')
			               - np.searchsorted(nxt.t, last_in, side='right'), 0)
		have = n + np.minimum(back, np.maximum(n_full - n, 0))
		have = have + np.minimum(fwd, np.maximum(n_full - have, 0))
		exists = (n > 0) & (have >= n_full)
		borrowed = exists & (n < n_full)
		ix = np.floor(((c - start_ns) / 1e9) / dt + 1e-9).astype(np.int64)
		for q in range(c.size):
			by_cell.setdefault(int(ix[q]), []).append((fd, int(lo[q]), int(hi[q]), bool(exists[q]),
			                                          bool(borrowed[q]), bool(unknown[q])))
	slots = seam = 0
	value_cells = []
	for k in range(n_time):
		entries = by_cell.get(k, [])
		if any(e[5] for e in entries):
			seam += 1
			if collect is not None:
				collect['unknown'].add(k)
			continue
		slots += 1
		want = any(e[3] for e in entries)
		if collect is not None:
			if want:
				collect['exists'].add(k)
				if any(e[4] for e in entries):
					seam += 1                              # exists by borrowing: not recomputed
				else:
					value_cells.append(k)
			continue
		if want != (k in cells):
			out['failures'].append(f'{tag(ev)}: cell {label_of(ev["start"], k * dt)} '
			                       + ('has no row, but a window there holds a full set of good samples'
			                          if want else 'has a row, but no window there holds a full set of good samples'))
		elif want and not any(e[4] for e in entries):
			value_cells.append(k)
		elif want:
			seam += 1                                  # exists by borrowing: not recomputed
	out['slots_checked'] += slots
	out['seam_slots'] += seam
	if max_rows is not None and len(value_cells) > max_rows:
		value_cells = sorted(rnd.sample(value_cells, max_rows))
	worst = 0.0
	for k in value_cells:
		ffts = [psd((fd.xyz[lo:hi, 0], fd.xyz[lo:hi, 1], fd.xyz[lo:hi, 2]), fs)
		        for fd, lo, hi, exists, _b, _u in by_cell[k] if exists]
		want = channel_means(ffts, edges, lpf_hz, pad)
		if collect is not None:
			collect['values'][k] = want
			continue
		got = np.array(cells[k])
		rel = np.abs(got - want) / np.abs(want)
		# A cell more than 120 dB under the row's strongest is rounding noise, not a measurement (a stuck sensor's
		# flat stretch has no power left once its mean is gone): it is held to that floor instead.
		floor = ABS_FLOOR * np.nanmax(want)
		within = (np.abs(got - want) <= REL_TOL * np.abs(want)) | (np.abs(got - want) <= floor)
		at_floor = within & (rel > REL_TOL)
		out['cells_at_floor'] += int(np.count_nonzero(at_floor))
		worst = max(worst, float(np.nanmax(np.where(at_floor, 0.0, rel))))   # floor cells kept out of the worst
		out['cells_checked'] += len(got)
		if not np.all(within):
			ch = int(np.nanargmax(np.where(within, 0.0, rel)))
			out['failures'].append(f'{tag(ev)}: cell {label_of(ev["start"], k * dt)} bin_{ch} is {got[ch]:.6e}, '
			                       f'recomputed {want[ch]:.6e} (relative error {rel[ch]:.2e})')
	out['rows_recomputed'] += len(value_cells)
	out['max_rel_err'] = max(out['max_rel_err'], worst)
	return len(value_cells), worst


def counted_samples(t, fs):
	"""The good samples that count toward an event, sorted: those whose run holds at least MIN_RUN_SAMPLES. t is the
	good-sample times of every loaded file together, so a run that crosses a file seam is counted whole, and a
	sample two overlapping files both hold is counted once."""
	t = np.unique(t)
	if t.size == 0:
		return t
	breaks = np.flatnonzero(np.diff(t) > RUN_GAP_PERIODS * 1e9 / fs)
	lengths = np.diff(np.concatenate(([0], breaks + 1, [t.size])))
	return t[np.repeat(lengths >= MIN_RUN_SAMPLES, lengths)]


def edges_check(ev, files, rng_start, rng_end, fs, out):
	"""An unclipped event starts and ends on a good sample that counts (counted_samples), with no sample that counts
	within EVENT_GAP_S outside it."""
	start_ns, end_ns = ns_of(ev['start']), ns_of(ev['end'])
	gap_ns = int(EVENT_GAP_S * 1e9) - 1_000_000
	t = np.concatenate([fd.t for fd in files.values() if fd.t.size]) if files else np.array([], dtype=np.int64)
	t = counted_samples(t, fs)
	for which, at, clipped, edge in (('start', start_ns, ev['clipped_start'], ns_of(rng_start)),
	                                 ('end', end_ns, ev['clipped_end'], ns_of(rng_end))):
		if clipped:
			if at != edge:
				out['failures'].append(f'{tag(ev)}: clipped at its {which}, but not at the range\'s {which}')
			continue
		i = np.searchsorted(t, at)
		if i >= t.size or t[i] != at:
			out['failures'].append(f'{tag(ev)}: its {which} {ev[which]} is not a good sample of a run of '
			                       f'{MIN_RUN_SAMPLES} or more')
			continue
		near = t[(t > at - gap_ns) & (t < at)] if which == 'start' else t[(t > at) & (t < at + gap_ns)]
		if near.size:
			out['failures'].append(f'{tag(ev)}: a good sample of a run of {MIN_RUN_SAMPLES} or more lies '
			                       f'{abs(int(near[-1 if which == "start" else 0]) - at) / 1e9:.3f} s '
			                       f'{"before its start" if which == "start" else "after its end"}, inside the '
			                       f'{EVENT_GAP_S:g} s event gap')


def tag(ev):
	return f'MMS{ev["probe"]} {ev["field"]} {ev["start"]:%Y-%m-%d %H:%M:%S.%f}'


# ---------------------------------------------------------------------------- the verifier's own CSV

VERIFIER_CSV = 'verifier.csv'


def write_verifier_csv(path, ev, collect, n_chan, step):
	"""The verifier's CSV for one event, tall: a row for every time cell that must have one, its cells the
	recomputed channel means, or blank where the row was not recomputed (a seam row, or one the sample left out).
	Beside it, <path>.json: the cells whose rows cannot be decided (left out of the comparison)."""
	dt, n_time = cell_width(ev['start'], ev['end'], step)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', newline='') as fh:
		w = csv.writer(fh)
		w.writerow(['time_utc'] + [f'bin_{i}' for i in range(n_chan)])
		for k in sorted(collect['exists']):
			vals = collect['values'].get(k)
			w.writerow([label_of(ev['start'], k * dt)]
			           + ([f'{v:.9e}' for v in vals] if vals is not None else [''] * n_chan))
	with open(path + '.json', 'w') as fh:
		json.dump({'unknown': sorted(int(k) for k in collect['unknown']), 'dt': dt, 'n_time': n_time}, fh)
	return path


def read_verifier_csv(path, ev, step):
	"""{cell: [floats] or None} of a verifier CSV, and the set of undecided cells."""
	dt, _n = cell_width(ev['start'], ev['end'], step)
	with open(path + '.json') as fh:
		unknown = set(json.load(fh)['unknown'])
	rows = {}
	with open(path, newline='') as fh:
		r = csv.reader(fh)
		next(r)
		for row in r:
			label_t = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
			k = int(round(((label_t - ev['start']).total_seconds() + 0.005) / dt))
			rows[k] = [float(v) for v in row[1:]] if row[1] != '' else None
	return rows, unknown


def compare_csvs(app_csv, verifier_csv, ev, edges, step):
	"""The app's CSV against the verifier's, cell by cell where the verifier has values, row presence everywhere it
	decided. The rows that differ, grouped into runs of consecutive decided rows: {'rows_compared', 'rows_valued',
	'differing', 'max_rel_err', 'groups': [{'start', 'stop', 'rows', 'missing', 'extra', 'values',
	'max_rel_err', 'worst'}]}. 'missing': the verifier has the row and the app does not; 'extra': the reverse."""
	_problems, app = structure(ev, app_csv, edges, step)
	mine, unknown = read_verifier_csv(verifier_csv, ev, step)
	dt, n_time = cell_width(ev['start'], ev['end'], step)
	out = {'rows_compared': 0, 'rows_valued': 0, 'differing': 0, 'max_rel_err': 0.0, 'groups': []}
	group = None
	for k in range(n_time):
		if k in unknown:
			continue
		want, have = k in mine, k in app
		if not want and not have:
			continue                                       # both say no row: matches, and ends no group
		out['rows_compared'] += 1
		kind, rel_max, worst = None, 0.0, None
		if want != have:
			kind = 'missing' if want else 'extra'
		elif mine[k] is not None:
			out['rows_valued'] += 1
			want_v, got = np.array(mine[k]), np.array(app[k])
			rel = np.abs(got - want_v) / np.abs(want_v)
			floor = ABS_FLOOR * np.nanmax(want_v)
			within = (np.abs(got - want_v) <= REL_TOL * np.abs(want_v)) | (np.abs(got - want_v) <= floor)
			ok_rel = np.where(within & (rel > REL_TOL), 0.0, rel)
			out['max_rel_err'] = max(out['max_rel_err'], float(np.nanmax(ok_rel)))
			if not np.all(within):
				kind = 'values'
				ch = int(np.nanargmax(np.where(within, 0.0, rel)))
				rel_max = float(rel[ch])
				worst = {'time': label_of(ev['start'], k * dt), 'bin': ch, 'app': float(got[ch]),
				         'verifier': float(want_v[ch])}
		if kind is None:
			group = None
			continue
		out['differing'] += 1
		label = label_of(ev['start'], k * dt)
		if group is None:
			group = {'start': label, 'stop': label, 'rows': 0, 'missing': 0, 'extra': 0, 'values': 0,
			         'max_rel_err': 0.0, 'worst': None}
			out['groups'].append(group)
		group['stop'] = label
		group['rows'] += 1
		group[kind] += 1
		if rel_max > group['max_rel_err']:
			group['max_rel_err'], group['worst'] = rel_max, worst
	return out


def event_key(ev, top_dir):
	"""The event's folder relative to the request folder (MMS1/B/<event folder>), or a tag when it has no CSV. An
	event that carries its own 'key' keeps it."""
	if ev.get('key'):
		return ev['key']
	if ev.get('csv'):
		return os.path.relpath(os.path.dirname(ev['csv']), top_dir).replace('\\', '/')
	return tag(ev)


# ---------------------------------------------------------------------------- one day

def verify_day(day, events, top_dir, data_dir, lpf_hz, rng, pad=0.25, step=0.05, num_bins=25, bursts=None,
               mode='sample', work_dir=None):
	"""events: [{'probe', 'field', 'start', 'end', 'clipped_start', 'clipped_end', 'outcome', 'csv'}], start/end
	datetimes; 'csv' the path the export wrote (None when it wrote none). rng: the request's (start, end).

	mode 'sample' recomputes one event per (probe, field) and the day's longest, up to MAX_VALUE_ROWS rows each;
	'full' every row of every written event. work_dir: write the verifier's CSVs there and compare them with the
	app's (post-process verification); report['events'] then holds one entry per event, keyed as event_key."""
	t0 = time.monotonic()
	bursts = bursts or DEFAULT_BURSTS
	out = {'day': day, 'passed': True, 'mode': mode, 'csvs': 0, 'rows': 0, 'sampled': [], 'unverifiable': [],
	       'cells_checked': 0, 'cells_at_floor': 0, 'rows_recomputed': 0, 'slots_checked': 0, 'seam_slots': 0,
	       'max_rel_err': 0.0, 'failures': [], 'rel_tol': REL_TOL, 'abs_floor': ABS_FLOOR,
	       'min_run_samples': MIN_RUN_SAMPLES, 'events': {}}
	edges = log_edges(pad, lpf_hz, num_bins)
	for fld in sorted({e['field'] for e in events}):
		readme = os.path.join(top_dir, f'{fld}_bins_README.md')
		got = readme_edges(readme) if os.path.exists(readme) else None
		if got != edges:
			out['failures'].append(f'{fld}_bins_README.md edges {got} differ from {edges}')

	def result(ev, **kw):
		r = out['events'].setdefault(event_key(ev, top_dir), {
			'probe': ev['probe'], 'field': ev['field'], 'start': str(ev['start']), 'end': str(ev['end']),
			'csv': ev.get('csv'), 'state': 'checked', 'issues': []})
		r.update(kw)
		return r

	parsed = {}
	for ev in events:
		has = ev.get('csv') and os.path.exists(ev['csv'])
		if ev['outcome'] == 'written' and not has:
			out['failures'].append(f'{tag(ev)}: export says written, but there is no CSV')
			result(ev, state='differs')['issues'].append('export says written, but there is no CSV')
			continue
		if ev['outcome'] != 'written':
			if has:
				out['failures'].append(f'{tag(ev)}: export says {ev["outcome"]}, but a CSV is there')
				result(ev, state='differs')['issues'].append(f'export says {ev["outcome"]}, but a CSV is there')
			else:
				result(ev, state='no csv', outcome=ev['outcome'])
			continue
		problems, cells = structure(ev, ev['csv'], edges, step)
		out['csvs'] += 1
		out['rows'] += len(cells)
		out['failures'].extend(f'{tag(ev)}: {p}' for p in problems[:5])
		r = result(ev, rows=len(cells))
		if problems:
			r['state'] = 'differs'
			r['issues'].extend(problems[:5])
		parsed[id(ev)] = cells

	written = [ev for ev in events if id(ev) in parsed and parsed[id(ev)]]
	rnd = random.Random(f'csv_verify {day}')
	if mode == 'full':
		chosen, max_rows = list(written), None
	else:
		chosen, max_rows = [], MAX_VALUE_ROWS
		for key in sorted({(ev['probe'], ev['field']) for ev in written}):
			group = [ev for ev in written if (ev['probe'], ev['field']) == key]
			chosen.append(rnd.choice(group))
		if written:
			longest = max(written, key=lambda ev: (ev['end'] - ev['start'], ev['probe'], ev['field']))
			if all(longest is not c for c in chosen):
				chosen.append(longest)
	for ev in chosen:
		inst, mode_, fs = bursts[ev['field']]
		listing = day_listing(data_dir, inst, mode_, ev['probe'], ev['start'].date() - timedelta(days=1),
		                      ev['end'].date() + timedelta(days=1))
		lo_ns, hi_ns = ns_of(ev['start']) - int(4e9), ns_of(ev['end']) + int(4e9)
		stamps = sorted(listing)
		# The files that can reach the event -- from the last stamp at or before its start - 4 s through the last at
		# or before its end + 4 s -- plus one more at each end, so each of those has its neighbours.
		first = max([i for i, s in enumerate(stamps) if s <= lo_ns] or [0])
		last = max([i for i, s in enumerate(stamps) if s <= hi_ns] or [first])
		i0, i1 = max(first - 1, 0), min(last + 1, len(stamps) - 1)
		sel = stamps[i0:i1 + 1]
		info = {'probe': ev['probe'], 'field': ev['field'], 'start': str(ev['start']), 'end': str(ev['end'])}
		n_fail = len(out['failures'])
		try:
			if any(len(listing[s]) > 1 for s in sel):
				raise Unverifiable('a stamp on disk in more than one version')
			files = {s: FileData(listing[s][0], ev['field'], ev['probe'], mode_, rnd) for s in sel}
		except Unverifiable as exc:
			out['unverifiable'].append({**info, 'why': str(exc)})
			result(ev, state='unverifiable')['issues'].append(str(exc))
			continue
		except AssertionError as exc:
			out['failures'].append(f'{tag(ev)}: {exc}')
			result(ev, state='differs')['issues'].append(str(exc))
			continue
		r = result(ev, files=[os.path.basename(f.path) for f in files.values()],
		           cdfs=[f.path for f in files.values()])
		if work_dir is None:
			n_rows, worst = recompute(ev, parsed[id(ev)], files, edges, lpf_hz, pad, step, fs, rnd, out,
			                          earlier_unloaded=i0 > 0, max_rows=max_rows)
		else:
			collect = {'unknown': set(), 'exists': set(), 'values': {}}
			n_rows, _w = recompute(ev, parsed[id(ev)], files, edges, lpf_hz, pad, step, fs, rnd, out,
			                       earlier_unloaded=i0 > 0, collect=collect, max_rows=max_rows)
			mine = write_verifier_csv(os.path.join(work_dir, event_key(ev, top_dir), VERIFIER_CSV), ev, collect,
			                          len(edges), step)
			cmp_ = compare_csvs(ev['csv'], mine, ev, edges, step)
			worst = cmp_['max_rel_err']
			out['max_rel_err'] = max(out['max_rel_err'], worst)
			out['cells_checked'] += cmp_['rows_valued'] * len(edges)
			r.update(verifier_csv=mine, compare=cmp_)
			if cmp_['differing']:
				out['failures'].append(f'{tag(ev)}: {cmp_["differing"]} row(s) differ, in {len(cmp_["groups"])} '
				                       f'group(s), the first {cmp_["groups"][0]["start"]} - '
				                       f'{cmp_["groups"][0]["stop"]}')
		edges_check(ev, files, *rng, fs, out)
		new = out['failures'][n_fail:]
		r['issues'].extend(f[len(tag(ev)) + 2:] if f.startswith(tag(ev)) else f for f in new
		                   if 'row(s) differ' not in f)
		r['state'] = 'differs' if new or r['issues'] else 'matches'
		r['rows_recomputed'] = n_rows
		out['sampled'].append({**info, 'rows': len(parsed[id(ev)]), 'rows_recomputed': n_rows,
		                       'max_rel_err': worst, 'files': [os.path.basename(f.path) for f in files.values()]})
	out['passed'] = not out['failures']
	out['seconds'] = round(time.monotonic() - t0, 2)
	return out
