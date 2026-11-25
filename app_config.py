"""The app's settings file.

Graham, 2026-10-06: "We should have a config file that holds the CDF download directory and default or
previously selected settings that should save every time the app closes and my config should say CDFs are
downloaded to that E: drive folder."

One JSON file, plasmacoop_config.json beside this module; the PLASMACOOP_CONFIG environment variable points
somewhere else (the tests do, so they never read or write the real one). Anything the file leaves out takes the
default below, so an older file keeps working as settings are added. Reading never raises: a missing or
unreadable file gives the defaults (an unreadable one with a printed warning). Saving writes a temporary file and
moves it into place, so a crash mid-save never leaves half a file. Each process reads the file once and keeps it in
memory (Graham, 2026-10-29); save() keeps that copy current, so a hand edit made while the app runs takes effect at
the next start.

  data_root  the folder the app keeps its working files in (FFT scratch, manifests, sidecars, the validation
             registry), written the Windows way ('E:/PlasmaCoop'); data_paths builds its folders from it. Graham,
             2026-11-25, for the installer: other PCs have no E: drive. The installer writes it.
  cdf_dir    the folder CDFs are downloaded to and read from, written the Windows way ('E:/PlasmaCoop/pydata').
             WSL reads the same folder as /mnt/e/PlasmaCoop/pydata: native_path() makes the swap.
  download   source: 'cdaweb' (NASA SPDF's CDAWeb, the default since 2026-10-06) or 'sdc' (the LASP SDC through
             pyspedas, as before); concurrent: how many files CDAWeb downloads at once.
  archive    the preprocessed archive: enabled (default off), delete_cdfs, verify_each_request, verify_on_launch
             (the endless background scan). preproc_archive reads them; the window's Settings tab edits them.
  event_ui   the event window's settings as they were when it last closed (event_window saves them).

Only stdlib here: data_paths imports this, and every FFT worker process imports data_paths.
"""
import copy
import json
import os
import re

VERSION = 1

DEFAULTS = {
	'version': VERSION,
	'data_root': 'E:/PlasmaCoop',
	'cdf_dir': 'E:/PlasmaCoop/pydata',
	'download': {'source': 'cdaweb', 'concurrent': 8},     # Graham: "CDAWeb at 8 at a time"
	# The preprocessed archive (preproc_archive), off unless turned on (wired 2026-11-03). Graham, 11-03: verify each
	# request's data before use, and scan the whole archive in the background without end, requests first.
	'archive': {'enabled': False, 'delete_cdfs': True, 'verify_each_request': False, 'verify_on_launch': True},
	# Post-process verification (post_verify.py; Graham, 2026-11-04): every request's CSVs checked, every row, by an
	# independent process with its own downloads; max_staged_gb: how far ahead its downloads may run.
	'post_verify': {'enabled': True, 'max_staged_gb': 20},
	'event_ui': {},
}

SOURCES = ('cdaweb', 'sdc')

_DRIVE = re.compile(r'^([A-Za-z]):[\\/](.*)$')        # 'E:/x' or 'E:\x'
_WSL = re.compile(r'^/mnt/([a-z])(?:/(.*))?$')         # '/mnt/e/x'


def path():
	"""Where the settings file is."""
	return os.environ.get('PLASMACOOP_CONFIG') or os.path.join(os.path.dirname(os.path.abspath(__file__)),
	                                                           'plasmacoop_config.json')


def _merge(base, over):
	out = copy.deepcopy(base)
	for key, value in over.items():
		if isinstance(value, dict) and isinstance(out.get(key), dict):
			out[key] = _merge(out[key], value)
		else:
			out[key] = value
	return out


_loaded = {}       # settings file path -> its settings, read once per process (Graham, 2026-10-29)


def load():
	"""The settings: the file over the defaults. Never raises. The file is read once per process and kept in memory;
	save() keeps that copy current, and each caller gets a copy of its own to change."""
	p = path()
	if p not in _loaded:
		_loaded[p] = _read(p)
	return copy.deepcopy(_loaded[p])


def _read(p):
	try:
		with open(p, encoding='utf-8') as fh:
			saved = json.load(fh)
		if not isinstance(saved, dict):
			raise ValueError('not a JSON object')
	except FileNotFoundError:
		return copy.deepcopy(DEFAULTS)
	except (OSError, ValueError) as exc:
		print(f'Settings file {p} could not be read ({exc}); using the defaults.')
		return copy.deepcopy(DEFAULTS)
	return _merge(DEFAULTS, saved)


def save(cfg):
	"""Write the settings: a temporary file, then a move into place."""
	p = path()
	tmp = p + '.tmp'
	with open(tmp, 'w', encoding='utf-8') as fh:
		json.dump(cfg, fh, indent=1, sort_keys=True)
		fh.write('\n')
	os.replace(tmp, p)
	_loaded[p] = _merge(DEFAULTS, cfg)


def update(section, values):
	"""Merge `values` into one section of the file and save it. Returns the new settings."""
	cfg = load()
	current = cfg.get(section)
	cfg[section] = _merge(current, values) if isinstance(current, dict) and isinstance(values, dict) else values
	save(cfg)
	return cfg


def native_path(p, nt=None):
	"""The same folder as this platform names it: 'E:/x' <-> '/mnt/e/x'. Other paths come back as they are."""
	nt = (os.name == 'nt') if nt is None else nt
	if nt:
		m = _WSL.match(p.replace('\\', '/'))
		return f'{m.group(1).upper()}:/{m.group(2) or ""}' if m else p
	m = _DRIVE.match(p)
	return f'/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), "/")}'.rstrip('/') if m else p


def data_root(cfg=None):
	"""The app's working-files folder, as this platform names it."""
	return native_path((cfg or load())['data_root'])


def cdf_dir(cfg=None):
	"""The CDF download folder, as this platform names it."""
	return native_path((cfg or load())['cdf_dir'])


def download_source(cfg=None):
	source = str((cfg or load())['download'].get('source', 'cdaweb')).lower()
	if source not in SOURCES:
		raise ValueError(f'download source {source!r} in {path()} is not one of {SOURCES}')
	return source


def download_concurrent(cfg=None):
	return max(1, int((cfg or load())['download'].get('concurrent', 8)))


def require_drive(folder):
	"""Raise unless the drive `folder` is on is there. On WSL an unmounted /mnt/<x> is a plain folder inside
	WSL's own disk image (on C:), so creating CDF folders under it would fill C: instead of failing."""
	m = _WSL.match(folder.replace('\\', '/'))
	if os.name != 'nt' and m:
		mount = f'/mnt/{m.group(1)}'
		if not os.path.ismount(mount):
			raise OSError(f'{mount} is not mounted; the CDF folder {folder} would land inside WSL (on C:)')
	m = _DRIVE.match(folder)
	if os.name == 'nt' and m and not os.path.exists(f'{m.group(1)}:/'):
		raise OSError(f'Drive {m.group(1)}: is not there; the CDF folder is {folder}')
