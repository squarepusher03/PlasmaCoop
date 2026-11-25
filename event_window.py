"""
The event-export window. Start it with event_ui.py; the work itself is in
event_export.py.

Long steps -- local-data check, downloads, event discovery, FFTs and drawing --
run one at a time on a background QThread. Everything that touches a widget or
asks a question runs on the main thread: a finished step hands its result to
the next step through a queued signal.
"""
import multiprocessing
import os
import subprocess
import sys
import threading
import traceback

from PyQt5.QtCore import (QDate, QDateTime, QItemSelection, QItemSelectionModel, QObject,
                          QSize, QThread, QTimer, QUrl, Qt, pyqtSignal)
from PyQt5.QtGui import QBrush, QColor, QDesktopServices, QFont, QTextCursor, QValidator
from PyQt5.QtWidgets import (
	QAbstractItemView, QApplication, QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog,
	QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
	QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSpinBox,
	QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

import app_config
import post_verify
import preproc_archive
import event_export as ee
import event_images
import ffttest
from download_progress import download_with_progress

# One colour per spacecraft, as (text, row tint). The mission's own scheme paints
# MMS1 black, but a black-on-grey row reads as chrome -- a heading, a disabled
# row, part of the window -- rather than as MMS1's data, so MMS1 takes purple
# here. MMS2-4 keep the usual red / green / blue.
PROBE_COLORS = {1: ('#4A148C', '#EADDF4'), 2: ('#B71C1C', '#F8D9D9'),
                3: ('#1B5E20', '#D8EFD9'), 4: ('#0D47A1', '#D6E4F7')}
# A grouped row spans several spacecraft, so it takes none of their colours:
# teal, which no probe uses -- and not grey, which reads as window chrome.
GROUP_COLORS = ('#004D40', '#D2EEEA')
WARN_TEXT, WARN_TINT = '#3E2A00', '#FFC247'
DATA_NAMES = {'B': 'B (SCM)', 'E': 'E (EDP)', 'F': 'FGM'}
COLUMNS = ('Probe', 'Field', 'FGM', 'Start (UTC)', 'End (UTC)', 'Duration (s)')
# How much of an event the probe's FGM data covers (FGM feeds the f_ce line):
# a symbol and colour per state. The cell also shows the percentage, and its
# tooltip lists the uncovered stretches.
FGM_LOOK = {ee.FGM_FULL: ('●', '#1B5E20'), ee.FGM_PARTIAL: ('◐', '#9A5B00'), ee.FGM_NONE: ('○', '#8B1A1A')}
EXCEL_NOTE = ('* Excel shows at most 16,384 columns. The wide layout gives every twentieth of a '
              'second its own column, so an event longer than about 13½ minutes opens in Excel '
              'with its end cut off. The tall layout runs time down the rows instead, so Excel '
              'shows every event in full.')
_CANCELLED = object()


class _ClampingSpinBox(QSpinBox):
	"""A spin box that pulls an out-of-range entry to the bound it overshot.

	A plain QSpinBox REJECTS text outside its range and silently restores the
	value it was showing before, so typing 50 into a 100..4094 box leaves
	whatever happened to be there -- 3000, say -- and nothing tells the user
	their number was discarded. Worse, it can look correct by accident: typing
	5000 while the box already reads 4094 "returns" 4094, which is the right
	answer for the wrong reason.

	Clamping in the editingFinished slot cannot fix this, because by the time
	that slot runs Qt has already thrown the typed text away -- which is why
	_snap_lpf's own min/max was dead code. The correction has to happen where
	the text is interpreted, so the entry is accepted here and converted with a
	clamp. That is what makes "below the minimum" actually mean the minimum.
	"""

	def _typed_digits(self, text):
		"""The typed number alone, with prefix/suffix and padding removed."""
		body = text
		for fix in (self.prefix(), self.suffix()):
			if fix:
				body = body.replace(fix, '')
		return body.strip()

	def validate(self, text, pos):
		body = self._typed_digits(text)
		if body in ('', '-', '+'):
			return QValidator.Intermediate, text, pos      # mid-typing, not a number yet
		try:
			int(body)
		except ValueError:
			return QValidator.Invalid, text, pos
		# Deliberately Acceptable even when out of range: refusing it here is
		# exactly what made the old value come back instead of the bound.
		return QValidator.Acceptable, text, pos

	def valueFromText(self, text):
		body = self._typed_digits(text)
		try:
			typed = int(body)
		except ValueError:
			return self.value()
		return min(max(typed, self.minimum()), self.maximum())


class _LogStream(QObject):
	"""Stand-in for sys.stdout / sys.stderr: whatever any thread prints lands
	in the log pane (through a queued signal) and still reaches the console."""
	text = pyqtSignal(str)
	encoding = 'utf-8'

	def __init__(self, passthrough):
		super().__init__()
		self._passthrough = passthrough

	def write(self, s):
		if s:
			self.text.emit(s)
			if self._passthrough is not None:
				try:
					self._passthrough.write(s)
				except Exception:
					pass
		return len(s) if s else 0

	def flush(self):
		if self._passthrough is not None:
			try:
				self._passthrough.flush()
			except Exception:
				pass

	def isatty(self):
		return False


class _Task(QObject):
	"""Runs fn(task) on a worker thread. fn may emit task.progress."""
	done = pyqtSignal(object)
	failed = pyqtSignal(str)
	progress = pyqtSignal(int, int, str)

	def __init__(self, fn):
		super().__init__()
		self._fn = fn

	def run(self):
		try:
			# Every task is a request: the preprocessed archive's background scan waits while it runs, and its
			# rows are checked before use (preproc_archive.request; Graham, 2026-11-03: "the per request pass
			# should take precedence and the launch pass resumes after the request").
			with preproc_archive.request():
				result = self._fn(self)
		except ee.Cancelled:
			result = _CANCELLED
		except Exception:
			self.failed.emit(traceback.format_exc())
			return
		self.done.emit(result)


class EventWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle('FFT Calculator')
		self.resize(1150, 980)
		self._cancel = threading.Event()
		self._busy = False
		self._listed = False
		self._events = []           # every event Find events found, on every probe
		self._rows = []             # what the table lists: Events, or EventGroups when grouped
		self._found = None          # (fields, probes, coverage) of the last Find, to rebuild rows from
		self._fgm_runs = {}         # probe -> FGM runs over the request range
		self._thread = self._task = self._then = None
		self._always_continue = False
		self._quitting = False      # the window was closed while busy: it closes itself once the step has stopped
		self._archive_scan = None   # preproc_archive.BackgroundScan, started by main() at launch
		self._lpf_step = int(round(1 / (2 * ee.PAD)))       # one FFT bin
		self._last_root = ''        # the folder Run last saved into, where its picker opens next
		self._last_top = ''         # the request folder Run last wrote, which the output explorer opens
		self._verify = None         # post_verify.Session: started at Find events, fed by Run

		central = QWidget()
		self.setCentralWidget(central)
		root = QVBoxLayout(central)
		# Two tabs (Graham, 2026-11-03: "a second tab in the UI to hide all of the more 'settings' like options").
		# Events keeps what changes from run to run, the frequency ceiling and the files per event included;
		# Settings holds the rest. The log sits below both, so it shows whichever tab is open.
		self.tabs = QTabWidget()
		events_tab, settings_tab = QWidget(), QWidget()
		events = QVBoxLayout(events_tab)
		events.addWidget(self._build_request_box())
		events.addWidget(self._build_data_box())
		events.addWidget(self._build_run_options_box())
		events.addLayout(self._build_events_area(), 4)
		events.addLayout(self._build_run_row())
		settings = QVBoxLayout(settings_tab)
		settings.addWidget(self._build_settings_box())
		settings.addWidget(self._build_archive_box())
		settings.addWidget(self._build_verify_box())
		settings.addWidget(self._build_download_box())
		settings.addStretch(1)
		self.tabs.addTab(events_tab, 'Events')
		self.tabs.addTab(settings_tab, 'Settings')
		root.addWidget(self.tabs, 4)
		self.log_view = QPlainTextEdit()
		self.log_view.setReadOnly(True)
		self.log_view.setMaximumBlockCount(5000)
		mono = QFont('Consolas')
		mono.setStyleHint(QFont.Monospace)
		self.log_view.setFont(mono)
		root.addWidget(self.log_view, 1)
		note = QLabel(EXCEL_NOTE)
		note.setWordWrap(True)
		note.setStyleSheet('color: #555; font-size: 9pt;')
		root.addWidget(note)

		self.stdout_stream = _LogStream(sys.stdout)
		self.stderr_stream = _LogStream(sys.stderr)
		self.stdout_stream.text.connect(self._append_log)
		self.stderr_stream.text.connect(self._append_log)

		self._inputs = [self.start_date, self.start_time, self.end_date, self.end_time,
		                *self.field_boxes.values(), *self.probe_boxes.values(),
		                self.lpf, self.fce, self.tall, self.wide, self.csv_box, self.graph_box,
		                self.group_box, *self.include_boxes.values(),
		                self.select_all_btn, self.clear_sel_btn,
		                *(w for pair in self.power_edits.values() for w in pair),
		                *self.archive_boxes.values(), self.verify_enabled, self.verify_cap,
		                self.download_source, self.download_concurrent]
		self._load_settings()       # the settings it had when it last closed, where they still fit
		self._refresh_state()

	# ------------------------------------------------------------------ layout

	def _build_request_box(self):
		box = QGroupBox('Request (UTC)')
		grid = QGridLayout(box)
		self.start_date, self.start_time = self._date_edit(), self._time_edit()
		self.end_date, self.end_time = self._date_edit(), self._time_edit()
		for row, (label, date_edit, time_edit) in enumerate(
				(('Start', self.start_date, self.start_time), ('End', self.end_date, self.end_time))):
			grid.addWidget(QLabel(label), row, 0)
			grid.addWidget(date_edit, row, 1)
			grid.addWidget(time_edit, row, 2)
		hint = QLabel('Times are optional: HH:MM, HH:MM:SS or HH:MM:SS.ff. A blank start time is 00:00; '
		              'a blank end time includes the whole end date (the range stops at 00:00 the next day).')
		hint.setWordWrap(True)
		grid.addWidget(hint, 2, 0, 1, 3)
		grid.setColumnStretch(2, 1)
		return box

	def _date_edit(self):
		today = QDateTime.currentDateTimeUtc().date()
		edit = QDateEdit(today)
		edit.setCalendarPopup(True)
		edit.setDisplayFormat('yyyy-MM-dd')
		edit.setDateRange(QDate(2015, 3, 1), today)           # MMS launched in March 2015
		edit.setMinimumWidth(edit.fontMetrics().horizontalAdvance('0000-00-00') + 48)   # digits + calendar arrow
		edit.dateChanged.connect(self._request_changed)
		return edit

	def _time_edit(self):
		edit = QLineEdit()
		edit.setPlaceholderText('optional   HH:MM[:SS[.ff]]')
		edit.setMaxLength(11)
		edit.textChanged.connect(self._request_changed)
		return edit

	def _build_data_box(self):
		box = QGroupBox('Data')
		grid = QGridLayout(box)
		fields_row, probes_row = QHBoxLayout(), QHBoxLayout()
		self.field_boxes = {}
		for fld, text in (('B', 'B   SCM magnetic'), ('E', 'E   EDP electric')):
			cb = QCheckBox(text)
			cb.setChecked(True)
			cb.toggled.connect(self._request_changed)
			self.field_boxes[fld] = cb
			fields_row.addWidget(cb)
		self.probe_boxes = {}
		for probe in ee.PROBES:
			cb = QCheckBox(f'MMS{probe}')
			cb.setChecked(True)
			text, tint = PROBE_COLORS[probe]
			cb.setStyleSheet(f'QCheckBox {{ color: {text}; background: {tint}; '
			                 f'padding: 3px 8px; border-radius: 3px; }}')
			cb.toggled.connect(self._request_changed)
			self.probe_boxes[probe] = cb
			probes_row.addWidget(cb)
		fields_row.addStretch(1)
		probes_row.addStretch(1)
		grid.addWidget(QLabel('Fields'), 0, 0)
		grid.addLayout(fields_row, 0, 1)
		grid.addWidget(QLabel('Probes'), 1, 0)
		grid.addLayout(probes_row, 1, 1)
		return box

	def _build_run_options_box(self):
		"""The two FFT settings that stay on the Events tab: the frequency ceiling and the files per event."""
		box = QGroupBox('Output')
		grid = QGridLayout(box)
		lo, hi = ee.lpf_bounds()
		self.lpf = _ClampingSpinBox()
		self.lpf.setRange(lo, hi)
		self.lpf.setSingleStep(self._lpf_step)
		self.lpf.setValue(ee.DEFAULT_LPF_HZ)
		self.lpf.setSuffix(' Hz')
		self.lpf.setToolTip(f'Highest frequency kept: {lo} to {hi} Hz, in {self._lpf_step} Hz steps (one FFT '
		                    f'bin). {hi} Hz is the Nyquist row for this sample rate. A value outside the range '
		                    f'is pulled to the bound it overshot. Changing it costs no recompute: the FFT cache '
		                    f'holds the full spectrum and the ceiling is applied when rows are binned.')
		self.lpf.valueChanged.connect(self._refresh_state)
		self.lpf.editingFinished.connect(self._snap_lpf)
		# Which files each event folder gets (Graham, 2026-10-07): the CSV, the graph or both. The f_ce line,
		# the colour-scale range and the event images belong to the graphs, so they rest while Graphs is off.
		self.csv_box = QCheckBox('CSV (ffts.csv)')
		self.graph_box = QCheckBox('Graphs (graph.png and the event images)')
		for cb in (self.csv_box, self.graph_box):
			cb.setChecked(True)
			cb.toggled.connect(self._refresh_state)
		grid.addWidget(QLabel('Frequency ceiling'), 0, 0)
		grid.addWidget(self.lpf, 0, 1)
		grid.addWidget(QLabel('Files per event'), 1, 0)
		grid.addWidget(self.csv_box, 1, 1)
		grid.addWidget(self.graph_box, 1, 2)
		grid.setColumnStretch(3, 1)
		return box

	def _build_settings_box(self):
		box = QGroupBox('FFT settings')
		grid = QGridLayout(box)
		pad = QDoubleSpinBox()
		pad.setDecimals(2)
		pad.setSuffix(' s')
		pad.setValue(ee.PAD)
		pad.setEnabled(False)
		pad.setToolTip('Half of the FFT window. Fixed for now: the graphing is built around it.')
		bins = QSpinBox()
		bins.setValue(ee.NUM_BINS)
		bins.setEnabled(False)
		bins.setToolTip('Number of log-frequency bins. Fixed for now: the graphing is built around it.')
		self.fce = QCheckBox('Draw the f_ce line (electron cyclotron frequency, from FGM)')
		self.fce.toggled.connect(self._refresh_state)
		self.tall = QRadioButton('Tall: one row per time step')
		self.wide = QRadioButton('Wide: one column per time step *')
		self.tall.setChecked(True)
		# One min/max pair per field: B and E sit orders of magnitude apart
		# (1e-6..1e-2 against 1e-4..10), so a single shared pair would be wrong
		# for one of them whenever both are selected. Prefilled with each field's
		# own default, so leaving them alone reproduces the previous images.
		power_row = QHBoxLayout()
		self.power_edits = {}
		for fld in ee.FIELDS:
			default = ee.PowerScale.for_field(fld)
			lo_edit, hi_edit = QLineEdit(f'{default.vmin:g}'), QLineEdit(f'{default.vmax:g}')
			for edit in (lo_edit, hi_edit):
				edit.setMaximumWidth(90)
				edit.setToolTip('Colour-scale limit only. It changes graph.png and nothing else: '
				                'ffts.csv holds the same numbers whatever this is set to, and the '
				                'request folder is named the same, so re-running at a new scale '
				                'replaces the previous graphs.')
				edit.textChanged.connect(self._refresh_state)
			self.power_edits[fld] = (lo_edit, hi_edit)
			power_row.addWidget(QLabel(f'{fld}   min'))
			power_row.addWidget(lo_edit)
			power_row.addWidget(QLabel('max'))
			power_row.addWidget(hi_edit)
		power_row.addStretch(1)
		grid.addWidget(QLabel('Half-window (PAD)'), 0, 0)
		grid.addWidget(pad, 0, 1)
		grid.addWidget(QLabel('Log-frequency bins'), 0, 2)
		grid.addWidget(bins, 0, 3)
		grid.addWidget(QLabel('f_ce'), 1, 0)
		grid.addWidget(self.fce, 1, 1, 1, 4)
		grid.addWidget(QLabel('CSV layout'), 2, 0)
		grid.addWidget(self.tall, 2, 1)
		grid.addWidget(self.wide, 2, 2, 1, 2)
		grid.addWidget(QLabel('Color-scale power range'), 3, 0)
		grid.addLayout(power_row, 3, 1, 1, 4)
		grid.setColumnStretch(4, 1)
		return box

	# The archive's and the downloads' settings live in their own sections of the settings file (app_config
	# 'archive' and 'download'), where preproc_archive and the downloaders read them; the window shows them and
	# writes them back when it closes.
	ARCHIVE_SETTINGS = (
		('enabled', 'Keep the data in the preprocessed archive (mms_preprocessed.sqlite)'),
		('delete_cdfs', 'Delete each CDF once its data is archived and verified'),
		('verify_each_request', "Verify each request's archived data from the drive before using it"),
		('verify_on_launch', 'Scan the whole archive in the background, without end (requests go first)'),
	)

	def _build_archive_box(self):
		box = QGroupBox('Preprocessed archive')
		column = QVBoxLayout(box)
		self.archive_boxes = {}
		for key, text in self.ARCHIVE_SETTINGS:
			cb = QCheckBox(text)
			self.archive_boxes[key] = cb
			column.addWidget(cb)
		self.archive_boxes['enabled'].toggled.connect(self._refresh_archive_boxes)
		note = QLabel('Changes take effect the next time the app starts.')
		note.setWordWrap(True)
		note.setStyleSheet('color: #555; font-size: 9pt;')
		column.addWidget(note)
		return box

	def _refresh_archive_boxes(self, *_):
		on = self.archive_boxes['enabled'].isChecked() and self.archive_boxes['enabled'].isEnabled()
		for key, cb in self.archive_boxes.items():
			if key != 'enabled':
				cb.setEnabled(on)

	def _build_verify_box(self):
		"""Post-process verification (post_verify.py): an independent process checks every request's CSVs."""
		box = QGroupBox('Post-process verification')
		grid = QGridLayout(box)
		self.verify_enabled = QCheckBox('Check every row of every request\'s CSVs with an independent process '
		                                '(its own downloads, deleted once checked)')
		self.verify_cap = QSpinBox()
		self.verify_cap.setRange(1, 10000)
		self.verify_cap.setSuffix(' GB')
		self.verify_cap.setToolTip('How far ahead its downloads may run: the most it keeps on disk at once.')
		self.verify_enabled.toggled.connect(
			lambda on: self.verify_cap.setEnabled(on and self.verify_enabled.isEnabled()))
		grid.addWidget(self.verify_enabled, 0, 0, 1, 3)
		grid.addWidget(QLabel('Downloads ahead'), 1, 0)
		grid.addWidget(self.verify_cap, 1, 1)
		note = QLabel('It starts downloading at Find events and checks each day once Run has written it. The '
		              'request folder\'s verification folder gets its report, and the CSVs and CDFs of any event '
		              'that differs; the Output explorer shows them while requests run.')
		note.setWordWrap(True)
		note.setStyleSheet('color: #555; font-size: 9pt;')
		grid.addWidget(note, 2, 0, 1, 3)
		grid.setColumnStretch(2, 1)
		return box

	def _build_download_box(self):
		box = QGroupBox('Downloads')
		grid = QGridLayout(box)
		self.download_source = QComboBox()
		for source, text in (('cdaweb', 'CDAWeb (NASA SPDF)'), ('sdc', 'MMS SDC (LASP, through pyspedas)')):
			self.download_source.addItem(text, source)
		self.download_concurrent = QSpinBox()
		self.download_concurrent.setRange(1, 32)
		self.download_concurrent.setToolTip('How many files CDAWeb downloads at once.')
		self.download_source.currentIndexChanged.connect(
			lambda *_: self.download_concurrent.setEnabled(self.download_source.isEnabled()
			                                               and self.download_source.currentData() == 'cdaweb'))
		grid.addWidget(QLabel('Source'), 0, 0)
		grid.addWidget(self.download_source, 0, 1)
		grid.addWidget(QLabel('Files at once'), 1, 0)
		grid.addWidget(self.download_concurrent, 1, 1)
		grid.setColumnStretch(2, 1)
		return box

	def _apply_app_settings(self, cfg):
		"""The archive and download sections of the settings file into the Settings tab."""
		archive = {**app_config.DEFAULTS.get('archive', {}), **(cfg.get('archive') or {})}
		for key, cb in self.archive_boxes.items():
			cb.setChecked(bool(archive.get(key, False)))
		try:
			source = app_config.download_source(cfg)
		except ValueError:
			source = app_config.DEFAULTS['download']['source']
		self.download_source.setCurrentIndex(max(0, self.download_source.findData(source)))
		self.download_concurrent.setValue(min(32, app_config.download_concurrent(cfg)))
		self._refresh_archive_boxes()
		self.download_concurrent.setEnabled(self.download_source.currentData() == 'cdaweb')
		pv = post_verify.settings(cfg)
		self.verify_enabled.setChecked(bool(pv.get('enabled')))
		self.verify_cap.setValue(int(round(float(pv.get('max_staged_gb', 20)))))
		self.verify_cap.setEnabled(self.verify_enabled.isChecked())

	def _app_settings(self):
		return {'archive': {key: cb.isChecked() for key, cb in self.archive_boxes.items()},
		        'download': {'source': self.download_source.currentData(),
		                     'concurrent': self.download_concurrent.value()},
		        'post_verify': {'enabled': self.verify_enabled.isChecked(),
		                        'max_staged_gb': self.verify_cap.value()}}

	def _build_events_area(self):
		column = QVBoxLayout()
		top = QHBoxLayout()
		self.find_btn = QPushButton('Find events')
		self.find_btn.clicked.connect(self._find_clicked)
		self.events_status = QLabel('')
		self.events_status.setWordWrap(True)
		top.addWidget(self.find_btn)
		top.addWidget(self.events_status, 1)
		column.addLayout(top)

		# How the found events are listed. Neither control finds or reads anything
		# again: both rebuild the rows from the events Find events already found.
		view = QHBoxLayout()
		self.group_box = QCheckBox(f'Group probes whose events start within {ee.GROUP_START_S:g} s')
		self.group_box.setToolTip('One row per group, listing its probes by number. A group runs from its first '
		                          'start to its latest end, and Run writes every probe in it over that whole window.')
		self.group_box.toggled.connect(self._rebuild_rows)
		view.addWidget(self.group_box)
		view.addSpacing(24)
		view.addWidget(QLabel('Include probes:'))
		self.include_boxes = {}
		for probe in ee.PROBES:
			cb = QCheckBox(str(probe))
			cb.setChecked(True)
			cb.setToolTip(f'Untick to leave MMS{probe} out of the found events: the list is recalculated '
			              f'without it, grouped or not, and Run and the event images leave it out too.')
			cb.toggled.connect(self._rebuild_rows)
			self.include_boxes[probe] = cb
			view.addWidget(cb)
		view.addStretch(1)
		column.addLayout(view)

		self.table = QTableWidget(0, len(COLUMNS))
		self.table.setHorizontalHeaderLabels(COLUMNS)
		self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
		self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
		self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
		self.table.verticalHeader().setVisible(False)
		header = self.table.horizontalHeader()
		header.setSectionResizeMode(QHeaderView.ResizeToContents)
		header.setStretchLastSection(True)
		self.table.itemSelectionChanged.connect(self._refresh_state)
		# A strong highlight that also shows while the window is inactive; the
		# native one fades to near-white over the probe tints.
		self.table.setStyleSheet('QTableWidget::item:selected { background-color: #1F5FBF; color: white; }')
		column.addWidget(self.table, 1)

		bottom = QHBoxLayout()
		self.select_all_btn = QPushButton('Select all events')
		self.select_all_btn.clicked.connect(self._select_all)
		self.clear_sel_btn = QPushButton('Clear selection')
		self.clear_sel_btn.clicked.connect(self.table.clearSelection)
		self.sel_label = QLabel('')
		tip = QLabel('Click, Shift+click and Ctrl+click select rows the way Explorer selects files.')
		tip.setStyleSheet('color: #555;')
		bottom.addWidget(self.select_all_btn)
		bottom.addWidget(self.clear_sel_btn)
		bottom.addWidget(self.sel_label)
		bottom.addStretch(1)
		bottom.addWidget(tip)
		column.addLayout(bottom)
		return column

	def _build_run_row(self):
		row = QHBoxLayout()
		self.run_btn = QPushButton('Run')
		self.run_btn.clicked.connect(self._run_clicked)
		self.cancel_btn = QPushButton('Cancel')
		self.cancel_btn.setEnabled(False)
		self.cancel_btn.clicked.connect(self._cancel_clicked)
		self.progress = QProgressBar()
		self.progress.setRange(0, 1)
		self.progress.setValue(0)
		self.status_label = QLabel('')
		self.status_label.setWordWrap(True)
		# Not one of the inputs: the explorer is its own process and can be opened while a request runs.
		self.explorer_btn = QPushButton('Output explorer')
		self.explorer_btn.setToolTip('The request folders as a file tree, with each event\'s post-process '
		                             'verification and a list of the CSVs that differ. Usable while requests run.')
		self.explorer_btn.clicked.connect(self._open_explorer)
		row.addWidget(self.run_btn)
		row.addWidget(self.cancel_btn)
		row.addWidget(self.progress, 1)
		row.addWidget(self.status_label, 2)
		row.addWidget(self.explorer_btn)
		return row

	def _open_explorer(self):
		folder = self._last_top if self._last_top and os.path.isdir(self._last_top) else self._last_root
		args = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_explorer.py')]
		subprocess.Popen(args + ([folder] if folder else []), cwd=os.path.dirname(os.path.abspath(__file__)))

	# ------------------------------------------------------------------ post-process verification

	def _verify_cfg(self):
		"""The settings file with the Settings tab's current values over it (the file is saved only on close)."""
		cfg = dict(app_config.load())
		for section, values in self._app_settings().items():
			cfg[section] = {**(cfg.get(section) or {}), **values}
		return cfg

	def _start_verify(self, rng, fields, probes):
		"""The request's acquisition stage begins: start its verifier, which downloads its own CDFs meanwhile."""
		self._end_verify()
		cfg = self._verify_cfg()
		if not post_verify.enabled(cfg):
			return None
		try:
			self._verify = post_verify.Session(rng, fields, probes, cfg=cfg)
		except Exception as exc:  # noqa: BLE001 -- verification never stops a request
			print(f'Post-process verification could not start: {exc}')
			self._verify = None
		return self._verify

	def _end_verify(self):
		"""Drop a verifier the request never reached Run with."""
		if self._verify is not None and not self._verify.used:
			self._verify.abandon()
		self._verify = None

	# ------------------------------------------------------------------ state

	def _fields(self):
		return tuple(f for f in ee.FIELDS if self.field_boxes[f].isChecked())

	def _probes(self):
		return tuple(p for p in ee.PROBES if self.probe_boxes[p].isChecked())

	def _excluded_probes(self):
		return {p for p, cb in self.include_boxes.items() if not cb.isChecked()}

	def _range(self):
		"""(RequestRange, None), or (None, why not)."""
		try:
			start_t = ee.parse_time_text(self.start_time.text())
			end_t = ee.parse_time_text(self.end_time.text())
			return ee.resolve_range(self.start_date.date().toPyDate(), start_t,
			                        self.end_date.date().toPyDate(), end_t), None
		except ValueError as exc:
			return None, str(exc)

	def _scales(self):
		"""({field: PowerScale}, None), or (None, why not).

		Only the ticked fields are checked: a nonsense value sitting in the box
		of a field you are not exporting should not block the run.
		"""
		out, problem = {}, None
		for fld, (lo_edit, hi_edit) in self.power_edits.items():
			if fld not in self._fields():
				lo_edit.setStyleSheet('')
				hi_edit.setStyleSheet('')
				continue
			try:
				out[fld] = ee.PowerScale(float(lo_edit.text()), float(hi_edit.text()))
				lo_edit.setStyleSheet('')
				hi_edit.setStyleSheet('')
			except ValueError as exc:
				bad = 'QLineEdit { border: 2px solid #C62828; }'
				lo_edit.setStyleSheet(bad)
				hi_edit.setStyleSheet(bad)
				reason = str(exc) if 'power' in str(exc).lower() else 'needs two numbers, e.g. 1e-6 and 1e-2'
				problem = problem or f'{fld} power range: {reason}'
		return (None, problem) if problem else (out, None)

	def _selected_rows(self):
		"""The selected rows' Events / EventGroups, in table order."""
		rows = sorted({ix.row() for ix in self.table.selectionModel().selectedRows()})
		out = []
		for r in rows:
			item = self.table.item(r, 0)
			idx = item.data(Qt.UserRole) if item is not None else None
			if idx is not None and idx < len(self._rows):
				out.append(self._rows[idx])
		return out

	def _selected_events(self):
		"""The per-probe Events Run works on: a listed event as it is, and a
		grouped row as one Event per probe over the whole group window."""
		out = []
		for row in self._selected_rows():
			out.extend(row.as_events() if isinstance(row, ee.EventGroup) else [row])
		return out

	def _refresh_state(self, *_):
		for edit in (self.start_time, self.end_time):
			try:
				ee.parse_time_text(edit.text())
				edit.setStyleSheet('')
			except ValueError:
				edit.setStyleSheet('QLineEdit { border: 2px solid #C62828; }')
		rng, err = self._range()
		fields, probes = self._fields(), self._probes()
		problems = [err] if err else []
		if not fields:
			problems.append('Tick B and/or E.')
		if not probes:
			problems.append('Tick at least one probe.')
		self.find_btn.setEnabled(not self._busy and not problems)
		graphs = self.graph_box.isChecked()
		for widget in (self.fce, *(w for pair in self.power_edits.values() for w in pair)):
			widget.setEnabled(not self._busy and graphs)
		if not (self.csv_box.isChecked() or graphs):
			problems.append('Tick CSV and/or Graphs.')

		# Run asks for the save folder itself, so the request folder's name is
		# previewed on the button instead of in a path box.
		if rng is not None and fields:
			name = ee.top_folder_name(rng, fields, self.lpf.value(), self._fce_on())
			self.run_btn.setToolTip(f'Asks where to save, then writes the request folder {name}')
		else:
			self.run_btn.setToolTip('')

		n_selected = len(self._selected_rows())
		self.sel_label.setText(f'{n_selected} of {len(self._rows)} event(s) selected')
		if not self._listed:
			problems.append('Press Find events.')
		elif n_selected == 0:
			problems.append('Select at least one event.')
		# Checked after find_btn deliberately: a bad colour range must not stop
		# you listing events, it only stops the run that would draw with it.
		_scales, scale_err = self._scales()
		if scale_err and graphs:
			problems.append(scale_err)
		self.run_btn.setEnabled(not self._busy and not problems)
		if not self._busy:
			self.status_label.setText('   '.join(problems))

	def _fce_on(self):
		"""The f_ce line is a graph setting: on only while Graphs is ticked too."""
		return self.fce.isChecked() and self.graph_box.isChecked()

	def _request_changed(self, *_):
		"""Dates, times, fields or probes changed: the list no longer matches."""
		if self._listed or self.table.rowCount():
			self._clear_events('The request changed -- press Find events to list its events.')
		self._refresh_state()

	def _clear_events(self, message=''):
		self._listed = False
		self.table.clearSelection()
		self.table.clearSpans()
		self.table.setRowCount(0)
		self._events = []
		self._rows = []
		self._found = None
		self._fgm_runs = {}
		# A new list starts with every probe included again.
		for cb in self.include_boxes.values():
			cb.blockSignals(True)
			cb.setChecked(True)
			cb.blockSignals(False)
		self.events_status.setText(message)

	def _snap_lpf(self):
		value = self.lpf.value()
		snapped = int(round(value / self._lpf_step)) * self._lpf_step
		snapped = min(max(snapped, self.lpf.minimum()), self.lpf.maximum())
		if snapped != value:
			self.lpf.setValue(snapped)

	def _set_busy(self, busy, text=''):
		self._busy = busy
		for widget in self._inputs:
			widget.setEnabled(not busy)
		if not busy:                # the Settings tab's dependent controls rest again where they should
			self._refresh_archive_boxes()
			self.download_concurrent.setEnabled(self.download_source.currentData() == 'cdaweb')
		self.cancel_btn.setEnabled(busy)
		if busy:
			self.progress.setRange(0, 0)
			self.status_label.setText(text)
		else:
			self.progress.setRange(0, 1)
			self.progress.setValue(0)
		self._refresh_state()

	# ------------------------------------------------------------------ background steps

	def _start(self, text, fn, then, always_continue=False):
		"""Run fn(task) off the main thread, then then(result) back on it."""
		self._then, self._always_continue = then, always_continue
		self._set_busy(True, text)
		thread = QThread(self)
		task = _Task(fn)
		task.moveToThread(thread)
		thread.started.connect(task.run)
		task.progress.connect(self._task_progress)
		task.done.connect(self._task_done)
		task.failed.connect(self._task_failed)
		task.done.connect(thread.quit)
		task.failed.connect(thread.quit)
		thread.finished.connect(task.deleteLater)
		thread.finished.connect(thread.deleteLater)
		self._thread, self._task = thread, task
		thread.start()

	def _task_progress(self, done, total, text):
		if total <= 0:
			self.progress.setRange(0, 0)
		else:
			self.progress.setRange(0, total)
			self.progress.setValue(done)
		if text:
			self.status_label.setText(text)

	def _task_done(self, result):
		then, always = self._then, self._always_continue
		self._then = None
		self._set_busy(False)
		if self._quitting:
			self._close_after_stop()
			return
		if result is _CANCELLED or (self._cancel.is_set() and not always):
			print('Cancelled.')
			self.status_label.setText('Cancelled.')
			return
		then(result)

	def _task_failed(self, tb):
		self._then = None
		self._set_busy(False)
		print(tb)
		if self._quitting:
			self._close_after_stop()
			return
		last = tb.strip().splitlines()[-1] if tb.strip() else 'Unknown error'
		QMessageBox.critical(self, 'Something went wrong', f'{last}\n\nThe full traceback is in the log.')

	def _cancel_clicked(self):
		self._cancel.set()
		self.cancel_btn.setEnabled(False)
		self.status_label.setText('Cancelling ...')
		print('Cancel requested -- downloads under way and the file being written are stopped and deleted; '
		      'everything already finished is kept.')

	# ------------------------------------------------------------------ find events

	def _find_clicked(self):
		rng, err = self._range()
		fields, probes = self._fields(), self._probes()
		if err or not fields or not probes:
			return
		self._cancel.clear()
		self._clear_events('')
		print(f'--- Find events: {ee.label_ts(rng.start)} up to {ee.label_ts(rng.end)} UTC, '
		      f'{ee.fields_suffix(fields)}, {", ".join(f"MMS{p}" for p in probes)}')
		self._start_verify(rng, fields, probes)
		# FGM is checked -- and offered for download -- together with B and E: it
		# feeds the f_ce line and the FGM coverage column of the event list.
		cancel = self._cancel
		self._start('Checking local data ...',
		            lambda task: ee.check_coverage(rng, fields + ('F',), probes, cancelled=cancel.is_set),
		            lambda coverage: self._after_coverage(rng, fields, probes, coverage))

	def _after_coverage(self, rng, fields, probes, coverage):
		# A pair whose validation deleted a corrupt file is offered for download
		# too (ee.Coverage.refetch), even when other files still cover the range.
		refetch = set(getattr(coverage, 'refetch', ()))
		missing = sorted({k for k, n in coverage.items() if n == 0} | refetch)
		if missing:
			choice = self._ask_missing(missing, 'Missing or corrupt local data for' if refetch
			                           else 'No local data for')
			if choice == 'cancel':
				print('Find events cancelled.')
				return
			if choice == 'download':
				cancel = self._cancel

				def download(task):
					download_with_progress(rng, missing, report=task.progress.emit, cancelled=cancel.is_set)
					return ee.check_coverage(rng, fields + ('F',), probes, cancelled=cancel.is_set)
				self._start('Downloading from the SDC ...', download,
				            lambda cov: self._discover(rng, fields, probes, cov))
				return
		self._discover(rng, fields, probes, coverage)

	def _discover(self, rng, fields, probes, coverage):
		with_p = tuple(p for p in probes if any(coverage.get((p, f), 0) for f in fields))
		with_f = tuple(f for f in fields if any(coverage.get((p, f), 0) for p in probes))
		cancel = self._cancel

		def work(task):
			events = ee.find_events(rng, with_f, with_p, cancelled=cancel.is_set)
			if cancel.is_set():
				raise ee.Cancelled
			task.progress.emit(0, 0, 'Reading FGM coverage ...')
			# FGM runs per probe, kept so every row -- one event, or a grouped
			# window -- is measured against them without reading FGM again.
			return events, {p: ee.fgm_runs(p, rng) for p in sorted({e.probe for e in events})}

		self._start('Finding events ...', work,
		            lambda result: self._show_events(fields, probes, coverage, *result))

	def _ask_missing(self, missing, lead):
		"""'download', 'continue' or 'cancel': offer the download first; if it is
		declined, ask whether to go on with what is on disk."""
		names = ', '.join(f'MMS{p} {DATA_NAMES[letter]}' for p, letter in missing)
		box = QMessageBox(self)
		box.setIcon(QMessageBox.Question)
		box.setWindowTitle('Missing data')
		box.setText(f'{lead}: {names}.')
		box.setInformativeText('Download it from the MMS Science Data Center now? Whatever the SDC has '
		                       'for this range is fetched; this can take a while.')
		download = box.addButton('Download', QMessageBox.AcceptRole)
		box.addButton("Don't download", QMessageBox.RejectRole)
		box.exec_()
		if box.clickedButton() is download:
			return 'download'
		box = QMessageBox(self)
		box.setIcon(QMessageBox.Question)
		box.setWindowTitle('Missing data')
		box.setText('Run anyway with the data already on disk, or cancel?')
		go = box.addButton('Run with what is on disk', QMessageBox.AcceptRole)
		box.addButton('Cancel', QMessageBox.RejectRole)
		box.exec_()
		return 'continue' if box.clickedButton() is go else 'cancel'

	def _show_events(self, fields, probes, coverage, events, fgm_runs=None):
		self._events = list(events)
		self._fgm_runs = dict(fgm_runs or {})
		self._found = (tuple(fields), tuple(probes), dict(coverage))
		self._listed = True
		files = '   '.join(f'MMS{p}: ' + ', '.join([f'{f} {coverage.get((p, f), 0)}' for f in fields]
		                                           + [f'FGM {coverage.get((p, "F"), 0)}'])
		                   for p in probes)
		self.events_status.setText(f'{len(self._events)} event(s).   Local files -- {files}')
		print(f'Found {len(self._events)} event(s).')
		self._rebuild_rows()

	def _rebuild_rows(self, *_):
		"""List the found events: leave out unticked probes, then group them when
		grouping is on. Recalculates the list only -- nothing is found again."""
		if self._found is None:
			self._refresh_state()
			return
		fields, probes, coverage = self._found
		excluded = self._excluded_probes()
		kept = ee.without_probes(self._events, excluded)
		grouped = self.group_box.isChecked()
		self._rows = ee.group_events(kept) if grouped else kept
		found = {(e.probe, e.field) for e in self._events}
		warnings = []
		for p in probes:
			if p in excluded:
				continue
			for f in fields:
				n = coverage.get((p, f), 0)
				if n == 0:
					warnings.append(f'MMS{p} {DATA_NAMES[f]}: no local data files in this interval')
				elif (p, f) not in found:
					warnings.append(f'MMS{p} {DATA_NAMES[f]}: {n} file(s), but no quality-good data in this interval')
		self.table.clearSelection()
		self.table.clearSpans()
		self.table.setRowCount(0)
		self.table.setHorizontalHeaderItem(0, QTableWidgetItem('Probes' if grouped else 'Probe'))
		self.table.setRowCount(len(warnings) + len(self._rows))
		for r, text in enumerate(warnings):
			self._set_warning_row(r, text)
		for i, row in enumerate(self._rows):
			if grouped:
				self._set_group_row(len(warnings) + i, i, row)
			else:
				self._set_event_row(len(warnings) + i, i, row)
		self._refresh_state()

	def _set_warning_row(self, r, text):
		for c in range(self.table.columnCount()):
			item = QTableWidgetItem('⚠  ' + text if c == 0 else '')
			item.setFlags(Qt.ItemIsEnabled)                    # not selectable
			item.setBackground(QBrush(QColor(WARN_TINT)))
			item.setForeground(QBrush(QColor(WARN_TEXT)))
			# The text spans every column; without a small size hint it would
			# size the Probe column to its own full width.
			item.setData(Qt.SizeHintRole, QSize(1, 1))
			if c == 0:
				font = item.font()
				font.setBold(True)
				item.setFont(font)
			self.table.setItem(r, c, item)
		self.table.setSpan(r, 0, 1, self.table.columnCount())

	def _fgm_coverage(self, probe, start, end):
		return ee.coverage_of(start, end, self._fgm_runs.get(probe, []))

	@staticmethod
	def _fgm_tip(probe, cov):
		if cov.state == ee.FGM_FULL:
			return f'MMS{probe}: FGM covers the whole window (100%).'
		if cov.state == ee.FGM_NONE:
			return f'MMS{probe}: no FGM data in this window (0%), so no f_ce line there.'
		gaps = ', '.join(f'{ee.label_ts(a)[11:]} – {ee.label_ts(b)[11:]}' for a, b in cov.gaps[:4])
		more = f' and {len(cov.gaps) - 4} more' if len(cov.gaps) > 4 else ''
		return f'MMS{probe}: FGM covers {cov.percent}% of the window; missing {gaps}{more}.'

	def _add_row_items(self, r, idx, cells, text, tint, fgm_colour, tips):
		for c, value in enumerate(cells):
			item = QTableWidgetItem(value)
			item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
			item.setBackground(QBrush(QColor(tint)))
			item.setForeground(QBrush(QColor(fgm_colour if c == 2 else text)))
			if c == 0:
				item.setData(Qt.UserRole, idx)
			if c == 2:
				item.setTextAlignment(Qt.AlignCenter)
				font = item.font()
				font.setBold(True)
				item.setFont(font)
			elif c == 5:
				item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
			if tips.get(c):
				item.setToolTip(tips[c])
			self.table.setItem(r, c, item)

	def _set_event_row(self, r, idx, ev):
		cov = self._fgm_coverage(ev.probe, ev.start, ev.end)
		symbol, colour = FGM_LOOK[cov.state]
		cells = (f'MMS{ev.probe}', ev.field, f'{symbol} {cov.percent}%', ee.label_ts(ev.start),
		         ee.label_ts(ev.end), f'{ev.duration_s:.2f}')
		# The request range can cut an event; that is said where the time is shown.
		tips = {2: self._fgm_tip(ev.probe, cov),
		        3: ev.start.isoformat(sep=' ') + ' UTC' + ('  (cut by the request range)' if ev.clipped_start else ''),
		        4: ev.end.isoformat(sep=' ') + ' UTC' + ('  (cut by the request range)' if ev.clipped_end else '')}
		text, tint = PROBE_COLORS[ev.probe]
		self._add_row_items(r, idx, cells, text, tint, colour, tips)

	def _set_group_row(self, r, idx, group):
		covs = [(p, self._fgm_coverage(p, group.start, group.end)) for p in group.probes]
		cells = (', '.join(str(p) for p in group.probes), group.field,
		         ' '.join(f'{p}{FGM_LOOK[cov.state][0]}' for p, cov in covs),
		         ee.label_ts(group.start), ee.label_ts(group.end), f'{group.duration_s:.2f}')
		members = '\n'.join(f'MMS{m.probe}: {ee.label_ts(m.start)} – {ee.label_ts(m.end)}'
		                    + ('  (cut by the request range)' if m.clipped_start or m.clipped_end else '')
		                    for m in group.members)
		tips = {0: f'Events starting within {ee.GROUP_START_S:g} s of the first:\n{members}',
		        2: '\n'.join(self._fgm_tip(p, cov) for p, cov in covs),
		        3: members, 4: members}
		text, tint = GROUP_COLORS
		self._add_row_items(r, idx, cells, text, tint, text, tips)

	def _select_all(self):
		model = self.table.model()
		selection = QItemSelection()
		last = self.table.columnCount() - 1
		for r in range(self.table.rowCount()):
			item = self.table.item(r, 0)
			if item is not None and item.data(Qt.UserRole) is not None:
				selection.select(model.index(r, 0), model.index(r, last))
		self.table.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)

	# ------------------------------------------------------------------ run

	def _choose_root(self):
		"""Ask where the request folder goes. '' when the picker is cancelled."""
		start = self._last_root or os.path.expanduser('~')
		path = QFileDialog.getExistingDirectory(self, 'Choose where the request folder goes', start)
		if not path:
			return ''
		self._last_root = os.path.normpath(path)
		return self._last_root

	def _run_clicked(self):
		events = self._selected_events()
		rng, err = self._range()
		if err or not events:
			return
		graphs = self.graph_box.isChecked()
		scales, scale_err = self._scales()
		if scale_err and graphs:
			QMessageBox.warning(self, 'Cannot run', scale_err)
			return
		root = self._choose_root()
		if not root:
			print('Run cancelled: no save folder chosen.')
			return
		settings = ee.ExportSettings(root=root, rng=rng, fields=self._fields(),
		                             lpf_hz=self.lpf.value(), fce=self._fce_on(),
		                             layout='tall' if self.tall.isChecked() else 'wide',
		                             scales=scales if graphs else None,
		                             csv=self.csv_box.isChecked(), graph=graphs)
		try:
			settings.validate()
		except ValueError as exc:
			QMessageBox.warning(self, 'Cannot run', str(exc))
			return
		if os.path.isdir(settings.top_dir):
			answer = QMessageBox.question(
				self, 'Folder already exists',
				f'A folder for this request already exists:\n\n{settings.top_dir}\n\n'
				'Proceed? Events you run again have their files overwritten; everything else '
				'in the folder is left alone.')
			if answer != QMessageBox.Yes:
				return
		self._cancel.clear()
		# No separate FGM step: Find events already offered FGM's download, and
		# the FGM column shows each event's coverage. f_ce uses what is on disk.
		self._export(settings, events)

	def _export(self, settings, events):
		total = len(events)
		cancel = self._cancel
		# Every requested probe gets a row in every event image, minus any left out.
		probes = tuple(p for p in self._probes() if p not in self._excluded_probes())
		plans = event_images.plan_images(events) if settings.graph else []      # event images are graphs too
		steps = total + len(plans)
		verify = None
		if settings.csv:
			verify = self._verify if self._verify is not None and self._verify.matches(
				settings.rng, self._fields(), self._probes()) else self._start_verify(
				settings.rng, self._fields(), self._probes())
			if verify is not None:
				verify.begin_run(settings.top_dir, settings.lpf_hz, events)
				self._verify = None             # this Run has it now
		self._last_top = settings.top_dir

		def work(task):
			try:
				# The Run's FFTs live in scratch and are deleted when it ends, cancelled or not
				# (Graham, 2026-11-19: no FFT cache between requests).
				with ffttest.fft_scratch_scope(ee.PICKLE_ROOT):
					return run(task)
			finally:
				ee.forget_prepared()           # the binned FFTs are this Run's alone
				if verify is not None:
					verify.finish(cancelled=cancel.is_set())

		discarded = []

		def run(task):
			task.progress.emit(0, 0, 'Computing FFTs ...')
			# Binned in the FFT workers, nothing written to disk (Graham, 2026-11-23: "Get rid of the binning
			# scratch"); Cancel or quitting stops it once the workers finish the files they hold.
			prepared = ee.prepare_export(
				settings, events, extra_windows=event_images.fft_windows(plans, probes), cancelled=cancel.is_set,
				progress=lambda done, n: task.progress.emit(done, n, f'Computing FFTs: {done} of {n} file(s) ...'))
			outcomes = {}
			for i, ev in enumerate(events):
				if cancel.is_set():
					break
				task.progress.emit(i, steps, f'Writing event {i + 1} of {total} ...')
				outcome = ee.export_event(settings, ev, binned=prepared, cancelled=cancel.is_set)
				if outcome == ee.DISCARDED:      # cancelled while being written: deleted, so not counted as done
					discarded.append(ev)
					break
				outcomes[ev] = outcome
				if verify is not None:
					verify.event_done(i, outcomes[ev])
			images = []
			for j, plan in enumerate(plans):
				if cancel.is_set():
					break
				task.progress.emit(total + j, steps, f'Drawing event image {j + 1} of {len(plans)} ...')
				images.extend(event_images.write_event_images(settings, [plan], probes, cancelled=cancel.is_set,
				                                              binned=prepared))
			task.progress.emit(len(outcomes) + len(images), steps, '')
			return outcomes, images

		print(f'--- Run: {total} event(s), {len(plans)} event image(s) -> {settings.top_dir}')
		self._start('Exporting ...', work,
		            lambda result: self._export_done(settings, events, *result, n_plans=len(plans),
		                                             discarded=len(discarded)),
		            always_continue=True)

	def _export_done(self, settings, events, outcomes, images=(), n_plans=0, discarded=0):
		written = sum(1 for o in outcomes.values() if o != ee.NO_FFT_ROWS)
		unusual = [(ev, o) for ev, o in outcomes.items() if o != ee.WRITTEN]
		lines = [f'{written} of {len(events)} event folder(s) written to:', settings.top_dir]
		if len(outcomes) < len(events):
			lines.append(f'Cancelled: {len(events) - len(outcomes)} event(s) not written'
			             + (' (the one being written when the cancel came was deleted; the events finished before '
			                'it are kept).' if discarded else '.'))
		if n_plans:
			drawn = sum(1 for _plan, path in images if path)
			lines.append(f'{drawn} of {n_plans} event image(s) drawn into {event_images.IMAGES_DIR}.')
		if unusual:
			lines += ['', 'Not written normally:']
			lines += [f'  MMS{ev.probe} {ev.field} {ee.label_ts(ev.start)}: {o}' for ev, o in unusual[:12]]
			if len(unusual) > 12:
				lines.append(f'  ... and {len(unusual) - 12} more (see the log)')
		print('\n'.join(lines))
		box = QMessageBox(self)
		box.setWindowTitle('Export finished')
		box.setText('\n'.join(lines))
		open_btn = box.addButton('Open folder', QMessageBox.ActionRole)
		box.addButton(QMessageBox.Ok)
		box.exec_()
		if box.clickedButton() is open_btn:
			QDesktopServices.openUrl(QUrl.fromLocalFile(settings.top_dir))

	# ------------------------------------------------------------------ settings

	# Graham, 2026-10-06: "default or previously selected settings that should save every time the app
	# closes". They live in the settings file's event_ui section (app_config). A value that is missing, or
	# no longer fits its widget (a date out of range, a frequency outside the ceiling's range), leaves that
	# widget at its default.

	def _settings(self):
		"""The window's settings now, as plain JSON values."""
		return {
			'start_date': self.start_date.date().toString('yyyy-MM-dd'),
			'start_time': self.start_time.text(),
			'end_date': self.end_date.date().toString('yyyy-MM-dd'),
			'end_time': self.end_time.text(),
			'fields': {f: cb.isChecked() for f, cb in self.field_boxes.items()},
			'probes': {str(p): cb.isChecked() for p, cb in self.probe_boxes.items()},
			'lpf_hz': self.lpf.value(),
			'fce': self.fce.isChecked(),
			'layout': 'tall' if self.tall.isChecked() else 'wide',
			'csv': self.csv_box.isChecked(),
			'graph': self.graph_box.isChecked(),
			'power': {f: [lo.text(), hi.text()] for f, (lo, hi) in self.power_edits.items()},
			'group': self.group_box.isChecked(),
			'include': {str(p): cb.isChecked() for p, cb in self.include_boxes.items()},
			'save_folder': self._last_root,
		}

	def _apply_settings(self, saved):
		"""Put saved settings into the widgets, skipping any value that no longer fits."""
		for key, edit in (('start_date', self.start_date), ('end_date', self.end_date)):
			date = QDate.fromString(str(saved.get(key, '')), 'yyyy-MM-dd')
			if date.isValid() and edit.minimumDate() <= date <= edit.maximumDate():
				edit.setDate(date)
		for key, edit in (('start_time', self.start_time), ('end_time', self.end_time)):
			if isinstance(saved.get(key), str):
				edit.setText(saved[key][:edit.maxLength()])
		for group, boxes in (('fields', self.field_boxes), ('probes', self.probe_boxes),
		                     ('include', self.include_boxes)):
			ticks = saved.get(group)
			if isinstance(ticks, dict):
				for key, cb in boxes.items():
					if isinstance(ticks.get(str(key)), bool):
						cb.setChecked(ticks[str(key)])
		lpf = saved.get('lpf_hz')
		if isinstance(lpf, int) and not isinstance(lpf, bool) and self.lpf.minimum() <= lpf <= self.lpf.maximum():
			self.lpf.setValue(lpf)
		if isinstance(saved.get('fce'), bool):
			self.fce.setChecked(saved['fce'])
		if saved.get('layout') in ('tall', 'wide'):
			(self.tall if saved['layout'] == 'tall' else self.wide).setChecked(True)
		for key, cb in (('csv', self.csv_box), ('graph', self.graph_box)):
			if isinstance(saved.get(key), bool):
				cb.setChecked(saved[key])
		power = saved.get('power')
		if isinstance(power, dict):
			for fld, (lo_edit, hi_edit) in self.power_edits.items():
				pair = power.get(fld)
				if isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, str) for v in pair):
					lo_edit.setText(pair[0])
					hi_edit.setText(pair[1])
		if isinstance(saved.get('group'), bool):
			self.group_box.setChecked(saved['group'])
		if isinstance(saved.get('save_folder'), str):
			self._last_root = saved['save_folder']

	def _load_settings(self):
		cfg = app_config.load()
		self._apply_settings(cfg.get('event_ui') or {})
		self._apply_app_settings(cfg)

	def _save_settings(self):
		try:
			app_config.update('event_ui', self._settings())
			for section, values in self._app_settings().items():
				app_config.update(section, values)
		except OSError as exc:
			print(f'Could not save the settings to {app_config.path()}: {exc}')

	# ------------------------------------------------------------------ misc

	def _append_log(self, text):
		self.log_view.moveCursor(QTextCursor.End)
		self.log_view.insertPlainText(text)
		self.log_view.moveCursor(QTextCursor.End)

	def start_archive_scan(self):
		"""The background scan of the whole preprocessed archive, when the settings ask for it
		(archive.verify_on_launch): it runs for as long as the window is open, yielding to every task."""
		if preproc_archive.verify_on_launch() and self._archive_scan is None:
			self._archive_scan = preproc_archive.BackgroundScan(ee.DATA_DIR, log=print).start()
		return self._archive_scan

	def _stop_archive_scan(self):
		if self._archive_scan is not None:
			self._archive_scan.cancel()
			self._archive_scan.join(2)

	# Quitting while busy (Graham, 2026-11-23: "the program should wait for the workers to finish, delete their
	# output, then cancel mid run"): the step is cancelled and the window stays open until it has stopped -- the FFT
	# workers finish the files they hold, what they computed is dropped, the pool is shut down -- and then closes
	# itself the normal way. Closing it again meanwhile offers to force it. What a cancel deletes (Graham,
	# 2026-11-24: "the only files that should be deleted are ones being currently downloaded and output artifacts
	# currently being generated, the rest should persist"): the .part files of downloads under way and the .partial
	# files of the event or image being written; finished downloads, events and images all stay.

	def closeEvent(self, event):
		if self._busy and self._quitting:
			answer = QMessageBox.question(
				self, 'Still stopping',
				'The window closes by itself once the running step has stopped. Force quit now instead? Its '
				'worker processes are ended where they are, and the downloads and files being written are '
				'deleted; everything already finished is kept.')
			if answer == QMessageBox.Yes and self._busy:
				self._force_quit()
			event.ignore()
			return
		if self._busy:
			answer = QMessageBox.question(
				self, 'Still working',
				'A step is still running. Quit anyway? It is stopped first: downloads under way and the event or '
				'image being written are stopped and deleted, the FFT workers finish the files they are on (what '
				'they computed is dropped), and then the window closes. Everything already finished is kept.')
			if answer != QMessageBox.Yes:
				event.ignore()
				return
			if self._busy:                 # (it may have finished while the question was open)
				self._quitting = True
				self._cancel.set()
				self.cancel_btn.setEnabled(False)
				self.status_label.setText('Quitting -- waiting for the running step to stop ...')
				print('Quit requested -- the running step is stopped first, then the window closes.')
				event.ignore()
				return
		self._save_settings()
		self._stop_archive_scan()
		self._end_verify()
		self._wait_for_thread()
		event.accept()

	def _close_after_stop(self):
		"""The step stopped after a quit: close now (from the event loop, not inside the task's end signal)."""
		print('The running step has stopped; closing.')
		QTimer.singleShot(0, self.close)

	def _wait_for_thread(self):
		"""The task thread's own event loop is quitting by now; let it end before the window goes."""
		thread = self._thread
		if thread is None:
			return
		try:
			thread.wait(10_000)
		except RuntimeError:                # already deleted (deleteLater after it finished)
			pass

	def _force_quit(self):
		"""The second close while stopping: end this process's worker processes, give the task thread a moment to
		delete its own partial files, delete any still listed (downloads' .part, outputs' .partial), drop its FFT
		scratch, exit. os._exit skips atexit and would leave the workers running (they outlive the window)."""
		self._cancel.set()
		self._save_settings()
		self._stop_archive_scan()
		self._end_verify()
		for _ in range(3):
			kids = multiprocessing.active_children()
			if not kids:
				break
			for p in kids:
				p.terminate()
			for p in kids:
				p.join(5)
		thread = self._thread
		try:
			if thread is not None:
				thread.wait(3_000)          # the write or download under way sees the cancel and cleans up itself
		except RuntimeError:
			pass
		import cdaweb
		left = cdaweb.discard_partial_downloads() + ee.discard_partial_outputs()
		if left:                          # still held open: the next download / Run into that folder sweeps them
			print(f'Force quit: {len(left)} partial file(s) still open could not be deleted: {left}')
		ffttest._drop_all_fft_scratch()
		os._exit(1)


def main():
	QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
	QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
	app = QApplication(sys.argv)
	window = EventWindow()
	saved = sys.stdout, sys.stderr
	sys.stdout, sys.stderr = window.stdout_stream, window.stderr_stream
	try:
		window.show()
		window.start_archive_scan()       # after the log is wired, so its lines show in the window
		return app.exec_()
	finally:
		sys.stdout, sys.stderr = saved


if __name__ == '__main__':
	# Launched directly, this module would become the script every FFT worker
	# re-imports on Windows (PyQt5 and the whole pipeline, once per worker).
	sys.exit('Start the UI with:  python event_ui.py')
