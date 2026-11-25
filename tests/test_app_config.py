"""The settings file (app_config). conftest gives each test its own file through PLASMACOOP_CONFIG."""
import json
import os

import pytest

import app_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_file_gives_the_defaults():
	cfg = app_config.load()
	assert cfg == app_config.DEFAULTS
	assert cfg['download'] == {'source': 'cdaweb', 'concurrent': 8}
	assert app_config.download_source() == 'cdaweb' and app_config.download_concurrent() == 8


def test_a_partial_file_keeps_the_defaults_it_leaves_out():
	with open(app_config.path(), 'w') as fh:
		json.dump({'download': {'concurrent': 4}, 'event_ui': {'fce': True}}, fh)
	cfg = app_config.load()
	assert cfg['download'] == {'source': 'cdaweb', 'concurrent': 4}
	assert cfg['cdf_dir'] == 'E:/PlasmaCoop/pydata' and cfg['event_ui'] == {'fce': True}


@pytest.mark.parametrize('content', ['', 'not json', '[1, 2]'])
def test_an_unreadable_file_gives_the_defaults_and_says_so(content, capsys):
	with open(app_config.path(), 'w') as fh:
		fh.write(content)
	assert app_config.load() == app_config.DEFAULTS
	assert 'could not be read' in capsys.readouterr().out


def test_save_then_load_round_trips_and_leaves_no_temporary_file():
	cfg = app_config.load()
	cfg['event_ui'] = {'start_date': '2019-08-16', 'probes': {'1': True, '2': False}}
	app_config.save(cfg)
	assert app_config.load() == cfg
	assert not os.path.exists(app_config.path() + '.tmp')


def test_update_merges_into_one_section():
	app_config.update('event_ui', {'fields': {'B': True, 'E': False}, 'lpf_hz': 3000})
	app_config.update('event_ui', {'fields': {'E': True}})
	cfg = app_config.load()
	assert cfg['event_ui'] == {'fields': {'B': True, 'E': True}, 'lpf_hz': 3000}
	assert cfg['download'] == app_config.DEFAULTS['download']


@pytest.mark.parametrize('given, nt, expected', [
	('E:/PlasmaCoop/pydata', False, '/mnt/e/PlasmaCoop/pydata'),
	('E:\\PlasmaCoop\\pydata', False, '/mnt/e/PlasmaCoop/pydata'),
	('/mnt/e/PlasmaCoop/pydata', True, 'E:/PlasmaCoop/pydata'),
	('E:/PlasmaCoop/pydata', True, 'E:/PlasmaCoop/pydata'),
	('/mnt/e/PlasmaCoop/pydata', False, '/mnt/e/PlasmaCoop/pydata'),
	('/home/x/pydata', False, '/home/x/pydata'),
])
def test_native_path_names_the_same_folder_per_platform(given, nt, expected):
	assert app_config.native_path(given, nt=nt) == expected


def test_an_unknown_source_is_an_error():
	app_config.update('download', {'source': 'ftp'})
	with pytest.raises(ValueError):
		app_config.download_source()


def test_an_unmounted_wsl_drive_is_refused(monkeypatch):
	monkeypatch.setattr(app_config.os, 'name', 'posix')
	monkeypatch.setattr(app_config.os.path, 'ismount', lambda p: False)
	with pytest.raises(OSError, match='not mounted'):
		app_config.require_drive('/mnt/q/PlasmaCoop/pydata')
	app_config.require_drive('/home/somewhere/pydata')          # not a drive path: nothing to check


def test_grahams_settings_file_downloads_cdfs_to_the_e_drive_folder():
	"""Graham: "my config should say CDFs are downloaded to that E: drive folder"."""
	with open(os.path.join(REPO, 'plasmacoop_config.json')) as fh:
		saved = json.load(fh)
	assert saved['cdf_dir'] == 'E:/PlasmaCoop/pydata'
	assert saved['download'] == {'source': 'cdaweb', 'concurrent': 8}
