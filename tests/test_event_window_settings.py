"""The event window keeps its settings in the settings file (Graham, 2026-10-06: "default or previously selected
settings that should save every time the app closes"). Needs PyQt5 (the Windows venv has it; plasma does not)."""
import os

import pytest

pytest.importorskip('PyQt5')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLBACKEND', 'Agg')

import app_config  # noqa: E402

SAVED = {'start_date': '2019-08-16', 'start_time': '07:40', 'end_date': '2019-08-16', 'end_time': '07:50',
         'fields': {'B': True, 'E': False}, 'probes': {'1': True, '2': False, '3': True, '4': False},
         'lpf_hz': 2000, 'fce': True, 'layout': 'wide', 'csv': True, 'graph': False,
         'power': {'B': ['1e-7', '1e-3'], 'E': ['1e-4', '10']},
         'group': True, 'include': {'1': True, '2': True, '3': False, '4': True}, 'save_folder': 'C:/out'}


@pytest.fixture
def app():
	from PyQt5.QtWidgets import QApplication
	return QApplication.instance() or QApplication([])


def window():
	from event_window import EventWindow
	return EventWindow()


def test_a_first_start_shows_the_defaults_and_closing_saves_them(app):
	w = window()
	defaults = w._settings()
	w.close()
	assert app_config.load()['event_ui'] == defaults


def test_the_saved_settings_come_back_on_the_next_start(app):
	app_config.update('event_ui', SAVED)
	w = window()
	assert w._settings() == SAVED
	w.close()


def test_closing_saves_what_was_changed(app):
	w = window()
	w.fce.setChecked(not w.fce.isChecked())
	w.end_time.setText('09:30')
	w.probe_boxes[4].setChecked(False)
	expected = w._settings()
	w.close()
	assert app_config.load()['event_ui'] == expected
	assert expected['end_time'] == '09:30' and expected['probes']['4'] is False


def test_a_saved_value_that_no_longer_fits_leaves_the_default(app):
	defaults = window()._settings()
	app_config.update('event_ui', {'start_date': '1999-01-01', 'lpf_hz': 10 ** 9, 'layout': 'diagonal',
	                               'fields': {'B': 'yes'}, 'power': {'B': [1, 2]}, 'csv': 'no', 'graph': 0})
	w = window()
	got = w._settings()
	for key in ('start_date', 'lpf_hz', 'layout', 'fields', 'power', 'csv', 'graph'):
		assert got[key] == defaults[key], key
	w.close()


# The file choice (Graham, 2026-10-07: "If there is no option to export only CSVs or only graphs in the app, it
# should be created"): CSV and Graphs boxes, both ticked by default.

def test_both_files_are_ticked_by_default(app):
	w = window()
	assert w.csv_box.isChecked() and w.graph_box.isChecked()
	w.close()


def test_graphs_off_rests_the_fce_box_and_the_colour_range(app):
	w = window()
	w.graph_box.setChecked(False)
	assert not w.fce.isEnabled()
	assert not any(e.isEnabled() for pair in w.power_edits.values() for e in pair)
	w.fce.setChecked(True)
	assert not w._fce_on()                       # f_ce is a graph setting
	w.graph_box.setChecked(True)
	assert w.fce.isEnabled() and w._fce_on()
	assert all(e.isEnabled() for pair in w.power_edits.values() for e in pair)
	w.close()


def test_neither_file_is_a_problem_run_names(app):
	w = window()
	w.csv_box.setChecked(False)
	w.graph_box.setChecked(False)
	assert 'Tick CSV and/or Graphs.' in w.status_label.text()
	assert not w.run_btn.isEnabled()
	w.csv_box.setChecked(True)
	assert 'Tick CSV and/or Graphs.' not in w.status_label.text()
	w.close()


def test_without_graphs_no_event_image_is_planned_drawn_or_computed(app, monkeypatch, tmp_path):
	from dataclasses import replace
	from datetime import datetime
	import event_export as ee
	import event_images
	calls = {'plan': 0, 'draw': 0, 'extra': []}

	def plan(events):
		calls['plan'] += 1
		return ['a plan']

	def draw(*a, **k):
		calls['draw'] += 1
		return []

	monkeypatch.setattr(event_images, 'plan_images', plan)
	monkeypatch.setattr(event_images, 'write_event_images', draw)
	monkeypatch.setattr(event_images, 'fft_windows', lambda plans, probes: {('B', 1): ['window']} if plans else {})
	monkeypatch.setattr(ee, 'prepare_export',
	                    lambda s, events, extra_windows=None, **k: calls['extra'].append(extra_windows))
	monkeypatch.setattr(ee, 'export_event', lambda s, ev, **k: ee.WRITTEN)

	class Task:
		class progress:
			emit = staticmethod(lambda *a: None)

	w = window()
	monkeypatch.setattr(w, '_start', lambda text, fn, then, always_continue=False: fn(Task()))
	rng = ee.RequestRange(datetime(2019, 8, 16), datetime(2019, 8, 17))
	ev = ee.Event(1, 'B', datetime(2019, 8, 16, 7, 40), datetime(2019, 8, 16, 7, 41), False, False)
	csv_only = ee.ExportSettings(root=str(tmp_path), rng=rng, fields=('B',), lpf_hz=3000, fce=False, graph=False)
	w._export(csv_only, [ev])
	assert calls['plan'] == 0 and calls['draw'] == 0 and calls['extra'] == [{}]
	w._export(replace(csv_only, graph=True), [ev])
	assert calls['plan'] == 1 and calls['draw'] == 1 and calls['extra'][1] == {('B', 1): ['window']}
	w.close()
