"""The output explorer: the request folders as a file tree, with each event's post-process verification beside it
(Graham, 2026-11-04: "it shouldn't be another 'app' with a UI, but rather we should have a file explorer kind of
wrapper that looks through the folder it outputs too and shows it's post-process verification process. This file
explorer should be useable while requests are in process").

  tree    the folder as a tree, filled as folders are opened, the opened ones refreshed every REFRESH_MS. The
          Verification column shows each event folder's state from its request's verification/report.json
          (post_verify.py), and each probe and field folder a count; the request folder shows the verifier's state
  pane    the differing files, and nothing else (Graham: "a list of each file that has differing rows, then allow
          them to drop down into a list of each group of rows that differs along with the times (start & stop for
          each group). This is only for conflicts and differing rows"): one entry per differing CSV, opening to its
          groups of differing rows with their first and last times, and any event-level issue. Hidden until there
          is a differing file, and closeable (Graham, 2026-11-04: "Keep the differing rows pane hidden; the pane
          should be closeable and should only appear if there's a differing file"). Closed, it comes back when a
          new differing file appears; the Differing files button opens it again meanwhile

It only reads, so it can stay open while the app runs requests. Double-click opens a file, a conflict's batch
folder, or a group's app CSV.

    python output_explorer.py [folder]
"""
import os
import sys
from datetime import datetime

from PyQt5.QtCore import QTimer, QUrl, Qt
from PyQt5.QtGui import QBrush, QColor, QDesktopServices
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                             QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

import post_verify as pv

REFRESH_MS = 2000
PATH_ROLE = Qt.UserRole
KIND_ROLE = Qt.UserRole + 1
DUMMY = '…'

# state -> (text, colour)
LOOK = {
	pv.WAITING: ('waiting for export', '#777777'),
	pv.EXPORTED: ('queued', '#555555'),
	'downloading': ('downloading CDFs', '#1F5FA8'),
	'computing': ('computing', '#1F5FA8'),
	pv.CHECKED: ('✔ structure checked', '#2E7D32'),
	pv.MATCHES: ('✔ matches', '#1B5E20'),
	pv.DIFFERS: ('✖ differs', '#B71C1C'),
	pv.UNVERIFIABLE: ('⚠ unverifiable', '#9A5B00'),
	pv.NO_CSV: ('— no CSV', '#777777'),
	pv.NOT_EXPORTED: ('not exported', '#777777'),
	pv.ERROR: ('⚠ check failed', '#9A5B00'),
}
DONE_STATES = (pv.CHECKED, pv.MATCHES, pv.DIFFERS, pv.UNVERIFIABLE, pv.NO_CSV, pv.NOT_EXPORTED, pv.ERROR)


def human(n):
	for unit in ('B', 'KB', 'MB', 'GB'):
		if n < 1024 or unit == 'GB':
			return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
		n /= 1024


class Statuses:
	"""Each request folder's verification/status.json, re-read when it changes."""

	def __init__(self):
		self._cache = {}

	def of(self, request_dir):
		path = os.path.join(request_dir, pv.VERIFY_DIR, pv.REPORT)
		try:
			mtime = os.stat(path).st_mtime_ns
		except OSError:
			return None
		hit = self._cache.get(path)
		if hit and hit[0] == mtime:
			return hit[1]
		data = pv.read_json(path)
		if data is None:                      # being replaced this instant: keep the last one
			return hit[1] if hit else None
		self._cache[path] = (mtime, data)
		return data


def request_of(path, root):
	"""The request folder holding path (the nearest folder at or above it with a verification folder), or None."""
	p = os.path.abspath(path)
	top = os.path.abspath(root)
	while True:
		if os.path.isdir(os.path.join(p, pv.VERIFY_DIR)):
			return p
		if p == top or os.path.dirname(p) == p:
			return None
		p = os.path.dirname(p)


def event_state(status, key):
	"""An event's state as shown: a queued event shows its day's download or check while that runs."""
	st = (status.get('events') or {}).get(key)
	if not st:
		return None
	state = st.get('state')
	if state == pv.EXPORTED:
		day = (st.get('start') or '')[:10]
		day_state = (status.get('days') or {}).get(day)
		if day_state == 'downloading':
			return 'downloading'
		if day_state == 'verifying':
			return 'computing'
	return state


class Explorer(QMainWindow):
	def __init__(self, folder=''):
		super().__init__()
		self.setWindowTitle('Output explorer')
		self.resize(1300, 800)
		self.statuses = Statuses()
		self.root = ''
		central = QWidget()
		self.setCentralWidget(central)
		col = QVBoxLayout(central)
		bar = QHBoxLayout()
		self.path_edit = QLineEdit()
		self.path_edit.setReadOnly(True)
		browse = QPushButton('Choose folder ...')
		browse.clicked.connect(self._browse)
		self.pane_btn = QPushButton('')
		self.pane_btn.setStyleSheet('color: #B71C1C;')
		self.pane_btn.clicked.connect(self._show_pane)
		self.pane_btn.hide()
		bar.addWidget(QLabel('Folder'))
		bar.addWidget(self.path_edit, 1)
		bar.addWidget(browse)
		bar.addWidget(self.pane_btn)
		col.addLayout(bar)
		self.summary = QLabel('')
		self.summary.setWordWrap(True)
		col.addWidget(self.summary)

		split = QSplitter(Qt.Horizontal)
		self.tree = QTreeWidget()
		self.tree.setHeaderLabels(['Name', 'Verification', 'Size', 'Modified'])
		self.tree.header().setSectionResizeMode(0, QHeaderView.Interactive)
		self.tree.setColumnWidth(0, 430)
		self.tree.setColumnWidth(1, 210)
		self.tree.itemExpanded.connect(self._expanded)
		self.tree.itemDoubleClicked.connect(self._open_item)
		split.addWidget(self.tree)
		right = QWidget()
		rcol = QVBoxLayout(right)
		rcol.setContentsMargins(0, 0, 0, 0)
		head = QHBoxLayout()
		self.conflict_label = QLabel('Differing files')
		close = QPushButton('\u2715')
		close.setFixedWidth(28)
		close.setToolTip('Close this pane. It opens again when another file differs, or from the Differing files '
		                 'button.')
		close.clicked.connect(self._close_pane)
		head.addWidget(self.conflict_label, 1)
		head.addWidget(close)
		rcol.addLayout(head)
		self.conflicts = QTreeWidget()
		self.conflicts.setHeaderLabels(['Differing CSV / rows', 'Rows', 'Detail'])
		self.conflicts.setColumnWidth(0, 330)
		self.conflicts.setColumnWidth(1, 50)
		self.conflicts.itemDoubleClicked.connect(self._open_conflict)
		rcol.addWidget(self.conflicts, 1)
		split.addWidget(right)
		split.setSizes([800, 500])
		col.addWidget(split, 1)
		self.pane = right
		self.pane.hide()
		self._closed_at = None          # how many differing files there were when the pane was closed
		self._n_conflicts = 0
		self._conflict_sig = None
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.refresh)
		self.timer.start(REFRESH_MS)
		if folder:
			self.set_root(folder)

	# -- the tree

	def _browse(self):
		path = QFileDialog.getExistingDirectory(self, 'Choose a request folder, or the folder that holds them',
		                                        self.root or os.path.expanduser('~'))
		if path:
			self.set_root(path)

	def set_root(self, folder):
		self.root = os.path.normpath(folder)
		self.path_edit.setText(self.root)
		self.tree.clear()
		self._conflict_sig = None
		top = self._make_item(self.tree.invisibleRootItem(), self.root, os.path.basename(self.root) or self.root)
		top.setExpanded(True)
		self.refresh()

	def _make_item(self, parent, path, name):
		item = QTreeWidgetItem([name, '', '', ''])
		item.setData(0, PATH_ROLE, path)
		is_dir = os.path.isdir(path)
		item.setData(0, KIND_ROLE, 'dir' if is_dir else 'file')
		if is_dir:
			item.addChild(QTreeWidgetItem([DUMMY]))
		parent.addChild(item)
		return item

	def _expanded(self, item):
		self._sync_children(item)

	def _sync_children(self, item):
		path = item.data(0, PATH_ROLE)
		try:
			entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
		except OSError:
			entries = []
		have = {}
		for i in range(item.childCount() - 1, -1, -1):
			ch = item.child(i)
			p = ch.data(0, PATH_ROLE)
			if p is None:
				item.removeChild(ch)                 # the placeholder
				continue
			have[p] = ch
		want = {e.path: e for e in entries}
		for p, ch in list(have.items()):
			if p not in want:
				item.removeChild(ch)
		for e in entries:
			ch = have.get(e.path) or self._make_item(item, e.path, e.name)
			if not e.is_dir():
				try:
					st = e.stat()
					ch.setText(2, human(st.st_size))
					ch.setText(3, datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'))
				except OSError:
					pass
		item.sortChildren(0, Qt.AscendingOrder)

	def _walk_open(self, item, out):
		out.append(item)
		if item.isExpanded():
			for i in range(item.childCount()):
				self._walk_open(item.child(i), out)

	def refresh(self):
		if not self.root:
			return
		top = self.tree.topLevelItem(0)
		if top is None:
			return
		items = []
		self._walk_open(top, items)
		for it in items:
			if it.isExpanded() and it.data(0, KIND_ROLE) == 'dir':
				self._sync_children(it)
		items = []
		self._walk_open(top, items)
		for it in items:
			self._label(it)
		self._refresh_conflicts()

	def _label(self, item):
		path = item.data(0, PATH_ROLE)
		if not path or item.data(0, KIND_ROLE) != 'dir':
			return
		req = request_of(path, self.root)
		status = self.statuses.of(req) if req else None
		if status is None:
			item.setText(1, '')
			return
		rel = os.path.relpath(path, req).replace('\\', '/')
		if rel == '.':
			text = status.get('state', '')
			ev = status.get('events') or {}
			done = sum(1 for k in ev if event_state(status, k) in DONE_STATES)
			bad = sum(1 for k in ev if event_state(status, k) == pv.DIFFERS)
			text = f'{text}: {done}/{len(ev)} done' + (f', {bad} differ' if bad else '')
			colour = '#B71C1C' if bad else ('#1B5E20' if status.get('state') == 'done' else '#1F5FA8')
			self._paint(item, text, colour)
			return
		ev = status.get('events') or {}
		if rel in ev:
			state = event_state(status, rel)
			text, colour = LOOK.get(state, (state or '', '#555555'))
			n = ev[rel].get('differing_rows')
			if state == pv.DIFFERS and n:
				text += f' ({n} rows)'
			self._paint(item, text, colour)
			return
		under = [k for k in ev if k.startswith(rel + '/')]
		if under:
			done = sum(1 for k in under if event_state(status, k) in DONE_STATES)
			bad = sum(1 for k in under if event_state(status, k) == pv.DIFFERS)
			self._paint(item, f'{done}/{len(under)} done' + (f', {bad} differ' if bad else ''),
			            '#B71C1C' if bad else ('#1B5E20' if done == len(under) else '#555555'))
		else:
			item.setText(1, '')

	@staticmethod
	def _paint(item, text, colour):
		if item.text(1) != text:
			item.setText(1, text)
		item.setForeground(1, QBrush(QColor(colour)))

	# -- conflicts

	def _requests(self):
		"""Every request folder at or below the root, one level of nesting deep (root, or root/<request>)."""
		out = []
		if os.path.isdir(os.path.join(self.root, pv.VERIFY_DIR)):
			out.append(self.root)
		try:
			for e in os.scandir(self.root):
				if e.is_dir() and os.path.isdir(os.path.join(e.path, pv.VERIFY_DIR)):
					out.append(e.path)
		except OSError:
			pass
		return out

	def _refresh_conflicts(self):
		reqs = self._requests()
		rows, lines = [], []
		for req in reqs:
			status = self.statuses.of(req)
			if not status:
				continue
			t = status.get('totals') or {}
			lines.append(f'{os.path.basename(req)}: {status.get("state")}. {t.get("days_checked", 0)} day(s) '
			             f'checked, {t.get("csvs", 0)} CSV(s), {t.get("rows_recomputed", 0):,} rows recomputed, '
			             f'{t.get("differing_events", 0)} differing; {status.get("staged_gb", 0):.2f} GB staged '
			             f'(updated {status.get("updated", "")[11:]})')
			for c in status.get('conflicts') or []:
				rows.append((req, c))
		self.summary.setText('\n'.join(lines) if lines else 'No post-process verification in this folder.')
		self._n_conflicts = len(rows)
		self._place_pane()
		sig = tuple((r, c['event'], c.get('batch')) for r, c in rows)
		if sig == self._conflict_sig:
			return
		self._conflict_sig = sig
		self.conflicts.clear()
		self.conflict_label.setText(f'Differing files: {len(rows)}')
		for req, c in rows:
			st = (self.statuses.of(req).get('events') or {}).get(c['event'], {})
			name = f'MMS{st.get("probe", "?")} {st.get("field", "?")}  {c["event"].rsplit("/", 1)[-1]}'
			n_rows = sum(g['rows'] for g in c.get('groups', []))
			top = QTreeWidgetItem([name, str(n_rows), f'day {c.get("day", "")}'])
			top.setData(0, PATH_ROLE, os.path.join(req, pv.VERIFY_DIR, c['folder']))
			top.setForeground(0, QBrush(QColor('#B71C1C')))
			self.conflicts.addTopLevelItem(top)
			for g in c.get('groups', []):
				kinds = ', '.join(f'{g[k]} {k}' for k in ('values', 'missing', 'extra') if g.get(k))
				detail = kinds + (f'; max relative error {g["max_rel_err"]:.2e}' if g.get('max_rel_err') else '')
				child = QTreeWidgetItem([f'{g["start"]}  →  {g["stop"]}', str(g['rows']), detail])
				child.setData(0, PATH_ROLE, os.path.join(req, pv.VERIFY_DIR, c['folder'], 'app.csv'))
				top.addChild(child)
			for issue in c.get('issues', []):
				child = QTreeWidgetItem(['event issue', '', issue])
				child.setData(0, PATH_ROLE, os.path.join(req, pv.VERIFY_DIR, pv.REPORT))
				top.addChild(child)

	# -- the pane

	def _place_pane(self):
		"""Shown only while a differing file exists; once closed, it stays closed until another one appears."""
		n = self._n_conflicts
		if n == 0:
			self.pane.hide()
			self._closed_at = None
		elif self._closed_at is None or n > self._closed_at:
			self._closed_at = None
			self.pane.show()
		self.pane_btn.setText(f'Differing files ({n})')
		self.pane_btn.setVisible(n > 0 and not self.pane.isVisible())

	def _close_pane(self):
		self._closed_at = self._n_conflicts
		self.pane.hide()
		self._place_pane()

	def _show_pane(self):
		self._closed_at = None
		self._place_pane()

	# -- opening things

	def _open_item(self, item, _col):
		path = item.data(0, PATH_ROLE)
		if path and item.data(0, KIND_ROLE) == 'file':
			QDesktopServices.openUrl(QUrl.fromLocalFile(path))

	def _open_conflict(self, item, _col):
		path = item.data(0, PATH_ROLE)
		if path and os.path.exists(path):
			QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def main(argv):
	QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
	app = QApplication(argv)
	win = Explorer(argv[1] if len(argv) > 1 else '')
	win.show()
	if len(argv) < 2:
		win._browse()
	return app.exec_()


if __name__ == '__main__':
	sys.exit(main(sys.argv))
