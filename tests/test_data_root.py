"""data_root: the app's working-files folder comes from the settings file (Graham, 2026-11-25, for the installer:
other PCs have no E: drive). The default keeps every path exactly as it was when data_paths fixed it."""
import json
import os
import subprocess
import sys

import app_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PRINT = ('import json, data_paths as d; print(json.dumps([d.DATA_ROOT, d.PICKLE_ROOT, d.CORRUPTED_LOG, '
          'd.INTERVAL_MANIFEST, d.BINNING_DIR, d.BATCH_DIR, d.DATA_DIR]))')


def _paths(config_path):
	"""data_paths as a fresh process sees it with this settings file (data_paths reads it once, at import)."""
	env = dict(os.environ, PLASMACOOP_CONFIG=str(config_path))
	out = subprocess.run([sys.executable, '-c', _PRINT], cwd=ROOT, env=env, capture_output=True, text=True,
	                     check=True).stdout
	return json.loads(out)


def test_the_default_is_the_old_fixed_root():
	assert app_config.DEFAULTS['data_root'] == 'E:/PlasmaCoop'
	assert app_config.data_root() == app_config.native_path('E:/PlasmaCoop')


def test_no_settings_file_gives_the_paths_data_paths_had_before(tmp_path):
	old = 'E:/PlasmaCoop' if os.name == 'nt' else '/mnt/e/PlasmaCoop'
	got = _paths(tmp_path / 'none.json')
	assert got[:6] == [old, f'{old}/.serialized', f'{old}/corrupted.txt', f'{old}/.manifest', f'{old}/.binning',
	                   f'{old}/batches']


def test_a_set_data_root_moves_every_working_folder_and_leaves_cdf_dir_alone(tmp_path):
	cfg = tmp_path / 'cfg.json'
	cfg.write_text(json.dumps({'data_root': 'D:/Lab/PlasmaData'}))
	root = app_config.native_path('D:/Lab/PlasmaData')
	got = _paths(cfg)
	assert got[:6] == [root, f'{root}/.serialized', f'{root}/corrupted.txt', f'{root}/.manifest', f'{root}/.binning',
	                   f'{root}/batches']
	assert got[6] == app_config.native_path('E:/PlasmaCoop/pydata')     # cdf_dir is its own setting


def test_data_root_is_named_the_way_this_platform_names_it():
	cfg = dict(app_config.DEFAULTS, data_root='D:/Lab/PlasmaData')
	assert app_config.data_root(cfg) == app_config.native_path('D:/Lab/PlasmaData')
	assert app_config.native_path('D:/Lab/PlasmaData', nt=False) == '/mnt/d/Lab/PlasmaData'
	assert app_config.native_path('D:/Lab/PlasmaData', nt=True) == 'D:/Lab/PlasmaData'
