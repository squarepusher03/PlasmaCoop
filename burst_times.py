#!.venv/bin/python3
"""List MMS burst segment start/end times (minute resolution) from CDF filenames.

Each segment is scoped to a single (probe, instrument) so you can see exactly
which spacecraft + which sensor was burst-recording when. Output is color-coded
by probe; instrument shows as a SCM / EDP tag.

Examples:
    python burst_times.py 2017-01-25
    python burst_times.py 2017-01-25 2017-01-26 2019-08-16
    python burst_times.py --range 2017-01-25 2017-02-01
    python burst_times.py --all
    python burst_times.py 2017-01-25 --probe 1 --inst edp --mode dce --gap 5
    python burst_times.py 2017-01-25 --no-color
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta

from data_paths import DATA_DIR  # per-platform E:/ vs /mnt/e

FNAME_RE = re.compile(r'_(\d{14})_')
PATH_RE  = re.compile(r'mms([1-4])[/\\](scm|edp)[/\\]brst[/\\]l2[/\\](scb|dce)')

RESET = '\033[0m'
PROBE_COLOR = {
	'1': '\033[96m',  # bright cyan
	'2': '\033[92m',  # bright green
	'3': '\033[93m',  # bright yellow
	'4': '\033[95m',  # bright magenta
}
INST_STYLE = {
	'scm': '\033[1m',   # bold
	'edp': '\033[2m',   # dim
}


def parse_args():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('dates', nargs='*', help='dates YYYY-MM-DD (any number)')
	p.add_argument('--range', nargs=2, metavar=('START', 'END'), help='date range inclusive')
	p.add_argument('--all', action='store_true', help='scan every date found')
	p.add_argument('--data-dir', default=DATA_DIR)
	p.add_argument('--probe', default='*', help='1..4 or * for all')
	p.add_argument('--inst', default='*', choices=['*', 'scm', 'edp'])
	p.add_argument('--mode', default='*', choices=['*', 'scb', 'dce'])
	p.add_argument('--gap', type=float, default=5.0,
	               help='minutes between consecutive files that splits two bursts (default 5)')
	p.add_argument('--no-color', action='store_true', help='disable ANSI colors')
	p.add_argument('--sort', choices=['time', 'group'], default='time',
	               help='within a day: by start time (default) or grouped by probe/inst')
	p.add_argument('--check', action='store_true',
	               help='matrix view: one row per burst, columns show which (probe,inst) recorded it')
	p.add_argument('--tolerance', type=float, default=2.0,
	               help='minutes — segments whose starts differ by less are treated as the same burst (default 2)')
	return p.parse_args()


def wanted_dates(args):
	if args.all:
		return None
	dates = set()
	for d in args.dates:
		dates.add(datetime.strptime(d, '%Y-%m-%d').date())
	if args.range:
		s = datetime.strptime(args.range[0], '%Y-%m-%d').date()
		e = datetime.strptime(args.range[1], '%Y-%m-%d').date()
		cur = s
		while cur <= e:
			dates.add(cur)
			cur += timedelta(days=1)
	if not dates:
		raise SystemExit('No dates given. Pass YYYY-MM-DD args, --range, or --all.')
	return dates


def collect(data_dir, probe, inst, mode, dates):
	"""Return {(probe, inst, mode): [datetime, ...]} sorted per key."""
	buckets = {}
	for root, _, files in os.walk(data_dir):
		path = root.replace('\\', '/')
		m_path = PATH_RE.search(path)
		if not m_path:
			continue
		p_, i_, mo_ = m_path.group(1), m_path.group(2), m_path.group(3)
		if probe != '*' and p_ != probe:
			continue
		if inst != '*' and i_ != inst:
			continue
		if mode != '*' and mo_ != mode:
			continue
		for f in files:
			if not f.endswith('.cdf'):
				continue
			m = FNAME_RE.search(f)
			if not m:
				continue
			try:
				t = datetime.strptime(m.group(1), '%Y%m%d%H%M%S')
			except ValueError:
				continue
			if dates is not None and t.date() not in dates:
				continue
			buckets.setdefault((p_, i_, mo_), []).append(t)
	for k in buckets:
		buckets[k].sort()
	return buckets


def group_bursts(times, gap_minutes):
	if not times:
		return []
	bursts = [[times[0]]]
	for prev, nxt in zip(times, times[1:]):
		if (nxt - prev).total_seconds() / 60.0 > gap_minutes:
			bursts.append([nxt])
		else:
			bursts[-1].append(nxt)
	return bursts


def tag(probe, inst, use_color):
	label = f'P{probe} {inst.upper()}'
	if not use_color:
		return f'[{label}]'
	pc = PROBE_COLOR.get(probe, '')
	istyle = INST_STYLE.get(inst, '')
	return f'[{pc}{istyle}{label}{RESET}]'


def fmt_minute(t):
	return t.strftime('%H:%M')


def cross_check(entries, tolerance, requested_probe, requested_inst, use_color):
	"""Cluster segments across (probe,inst) by start-time proximity.

	Prints a row per cluster showing which (probe,inst) recorded each burst.
	"""
	# Determine expected columns from what the user asked for.
	probes = ['1', '2', '3', '4'] if requested_probe == '*' else [requested_probe]
	insts = ['scm', 'edp'] if requested_inst == '*' else [requested_inst]
	cols = [(p, i) for p in probes for i in insts]

	# Sort all entries by start.
	entries = sorted(entries, key=lambda e: e[0])

	clusters = []
	for e in entries:
		start = e[0]
		placed = False
		for c in clusters:
			if abs((start - c['ref']).total_seconds()) / 60.0 <= tolerance:
				c['items'].append(e)
				# extend bounds
				c['min_start'] = min(c['min_start'], e[0])
				c['max_end']   = max(c['max_end'],   e[1])
				placed = True
				break
		if not placed:
			clusters.append({'ref': start, 'items': [e],
			                 'min_start': e[0], 'max_end': e[1]})

	clusters.sort(key=lambda c: c['min_start'])

	# Header
	col_labels = [f'P{p}{i.upper()[0]}' for (p, i) in cols]  # e.g. P1S, P1E, P2S...
	header = f"  {'burst (min..max)':<24}  " + '  '.join(f'{x:>3}' for x in col_labels) + '   coverage'
	print(header)
	print('  ' + '-' * (len(header) - 2))

	missing_rows = 0
	for c in clusters:
		day = c['min_start'].date()
		# Group cluster items by (probe, inst)
		seen = {}
		for st, en, p_, i_, mo_, n in c['items']:
			seen.setdefault((p_, i_), []).append((st, en, n))
		cells = []
		ok = True
		for (p, i) in cols:
			if (p, i) in seen:
				if use_color:
					pc = PROBE_COLOR.get(p, '')
					cells.append(f'{pc}  ✓{RESET}')
				else:
					cells.append('  ✓')
			else:
				ok = False
				if use_color:
					cells.append('\033[91m  ✗\033[0m')
				else:
					cells.append('  ✗')
		if not ok:
			missing_rows += 1
		span = f"{c['min_start'].strftime('%H:%M')} → {c['max_end'].strftime('%H:%M')}"
		date_tag = c['min_start'].strftime('%Y-%m-%d')
		status = 'OK' if ok else 'MISSING'
		status_str = (f'\033[92m{status}\033[0m' if ok and use_color
		              else f'\033[91m{status}\033[0m' if use_color else status)
		print(f"  {date_tag} {span:<11}  " + '  '.join(cells) + f'   {status_str}')

	print()
	print(f'Checked {len(clusters)} burst clusters against {len(cols)} (probe,inst) columns '
	      f'within ±{tolerance:.0f} min tolerance.')
	if missing_rows:
		print(f'{missing_rows} cluster(s) have missing coverage — see ✗ above.')
	else:
		print('All clusters fully covered by every requested probe/instrument.')


def main():
	args = parse_args()
	dates = wanted_dates(args)
	use_color = (not args.no_color) and sys.stdout.isatty()
	buckets = collect(args.data_dir, args.probe, args.inst, args.mode, dates)

	if not buckets:
		print('No CDFs found matching filters.')
		return

	# Build flat list of (start, end, probe, inst, n_files) entries.
	entries = []
	total_files = 0
	for (p_, i_, mo_), times in buckets.items():
		total_files += len(times)
		for b in group_bursts(times, args.gap):
			entries.append((b[0], b[-1], p_, i_, mo_, len(b)))

	if args.check:
		cross_check(entries, args.tolerance, args.probe, args.inst, use_color)
		return

	# Sort: by day always; then time or grouped within day.
	if args.sort == 'time':
		entries.sort(key=lambda e: (e[0].date(), e[0], e[2], e[3]))
	else:
		entries.sort(key=lambda e: (e[0].date(), e[2], e[3], e[0]))

	last_day = None
	for start, end, p_, i_, mo_, n in entries:
		day = start.date()
		if day != last_day:
			print(f'\n=== {day} ===')
			last_day = day
		dur = (end - start).total_seconds() / 60.0
		print(f'  {tag(p_, i_, use_color)} {fmt_minute(start)} → {fmt_minute(end)}   '
		      f'({n} files, span {dur:.1f} min)')

	# Legend
	print()
	if use_color:
		legend = '  '.join(tag(p, 'scm', True) for p in ['1', '2', '3', '4'])
		legend += '   ' + tag('1', 'scm', True) + '=SCM bold  ' + tag('1', 'edp', True) + '=EDP dim'
		print('Legend: ' + legend)
	print(f'Total: {len(entries)} segments across {total_files} files '
	      f'({len(buckets)} (probe,inst) buckets).')


if __name__ == '__main__':
	main()
