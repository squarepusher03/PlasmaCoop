"""The export's file choice (Graham, 2026-10-07: "If there is no option to export only CSVs or only graphs in the
app, it should be created"): ExportSettings.csv / .graph pick which files an event folder gets, both by default.
What a file holds does not depend on the choice -- the CSV-only CSV and the graph-only PNG are byte-identical to
the files a default export writes."""
import os
import sys
from datetime import datetime, timedelta

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import event_export as ee
from short_interval import ShortIntervalPlotter

START = datetime(2021, 2, 23, 21, 24, 55, 50_869)
EVENT = ee.Event(1, 'B', START, START + timedelta(seconds=2), False, False)


def settings(root, **kw):
	rng = ee.RequestRange(datetime(2021, 2, 23), datetime(2021, 2, 24))
	return ee.ExportSettings(root=str(root), rng=rng, fields=('B',), lpf_hz=3000, fce=kw.pop('fce', False), **kw)


@pytest.fixture
def binned(monkeypatch):
	"""export_event writes the grid prepare_export binned; here a fixed grid is prepared instead, so every run sees
	the same numbers and no FFT is computed."""
	bins, n_chan = ShortIntervalPlotter._analytic_log_bins(ee.PAD, 3000, ee.NUM_BINS)
	n_time = 40
	mean_g = 10 ** np.random.default_rng(0).uniform(-6, -2, size=(n_time, n_chan))
	grid = (mean_g, np.arange(n_time + 1) * 0.05, np.arange(n_chan + 1), bins, n_chan, 1e-6, 1e-2, 0.05)
	store = ee.BinnedFfts()
	store.put(EVENT.field, EVENT.probe, EVENT.start, EVENT.end, grid)
	monkeypatch.setattr(ee, '_prepared', store)
	return grid


def files_of(s):
	folder = ee.event_dir(s.top_dir, EVENT)
	return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def read(s, name):
	with open(os.path.join(ee.event_dir(s.top_dir, EVENT), name), 'rb') as fh:
		return fh.read()


def test_both_files_are_the_default():
	s = settings('x')
	assert s.csv is True and s.graph is True


def test_neither_file_is_refused():
	with pytest.raises(ValueError, match='CSV'):
		settings('x', csv=False, graph=False).validate()
	settings('x', csv=True, graph=False).validate()
	settings('x', csv=False, graph=True).validate()


def test_the_choice_does_not_rename_the_request_folder():
	assert settings('x', csv=False).top_dir == settings('x').top_dir == settings('x', graph=False).top_dir


def test_default_writes_both(tmp_path, binned):
	s = settings(tmp_path)
	assert ee.export_event(s, EVENT, log=lambda *_: None) == ee.WRITTEN
	assert files_of(s) == [ee.CSV_NAME, ee.PNG_NAME]


def test_csv_only_writes_no_graph(tmp_path, binned):
	s = settings(tmp_path, graph=False)
	assert ee.export_event(s, EVENT, log=lambda *_: None) == ee.WRITTEN
	assert files_of(s) == [ee.CSV_NAME]


def test_graph_only_writes_no_csv(tmp_path, binned):
	s = settings(tmp_path, csv=False)
	assert ee.export_event(s, EVENT, log=lambda *_: None) == ee.WRITTEN
	assert files_of(s) == [ee.PNG_NAME]


def test_csv_only_reads_no_fgm_even_with_fce_on(tmp_path, binned, monkeypatch):
	def no_fgm(*a, **k):
		raise AssertionError('f_ce was read for an export without a graph')
	monkeypatch.setattr(ee, 'collect_fce', no_fgm)
	s = settings(tmp_path, graph=False, fce=True)
	assert ee.export_event(s, EVENT, log=lambda *_: None) == ee.WRITTEN      # never WRITTEN_NO_FCE: no graph


def test_graph_only_still_draws_fce_and_reports_its_absence(tmp_path, binned, monkeypatch):
	seen = []
	monkeypatch.setattr(ee, 'collect_fce', lambda *a, **k: seen.append(1) or (np.array([]), np.array([])))
	s = settings(tmp_path, csv=False, fce=True)
	assert ee.export_event(s, EVENT, log=lambda *_: None) == ee.WRITTEN_NO_FCE and seen == [1]


def test_csv_only_csv_and_graph_only_png_are_byte_identical_to_the_default(tmp_path, binned):
	both = settings(tmp_path / 'both')
	ee.export_event(both, EVENT, log=lambda *_: None)
	csv_only = settings(tmp_path / 'csv', graph=False)
	ee.export_event(csv_only, EVENT, log=lambda *_: None)
	graph_only = settings(tmp_path / 'graph', csv=False)
	ee.export_event(graph_only, EVENT, log=lambda *_: None)
	assert read(csv_only, ee.CSV_NAME) == read(both, ee.CSV_NAME)
	assert read(graph_only, ee.PNG_NAME) == read(both, ee.PNG_NAME)
