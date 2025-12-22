import os
from datetime import datetime

import pandas as pd

def gen_path(project='mms', satellite='1', inst='scm', mode='scb', process='fft', ext='pkl', date=datetime(2019,8,16,9,31,56), pad=0.5):
	pth = ''
	match process:
		case 'fft':
			pth = 'fft/'
		case _:
			pth = ''

	if date is not None:
		dt = f'{date.year}{date.month:02d}{date.day:02d}{date.hour:02d}{date.minute:02d}{date.second:02d}.{ext}'
		if process == 'fft' and ext == 'pkl':
			dt = f'{pad:0.1f}p' + dt
	else:
		dt = ''

	return os.path.join(
		'./.cache',
		f'{project}/{satellite}/{inst}/{pth}{mode}/{dt}'
	)


def clear_fft(project='mms', satellite='1', inst='scm', mode='scb', ext='pkl', date=datetime(2019,8,16,9,31,56)):
	clear_cache(gen_path(project, satellite, inst, mode, 'fft', date=date), 'y', ext)


def clear_xlsx(project='mms', satellite='1', inst='scm', mode='scb', ext='xlsx', date=datetime(2019,8,16,9,31,56), pad=0.5):
	clear_cache(gen_path(project, satellite, inst, mode, date=date, pad=pad), 'y', ext)


def clear_pkl(project='mms', satellite='1', inst='scm', mode='scb', ext='pkl', date=datetime(2019,8,16,9,31,56), pad=0.5):
	clear_cache(gen_path(project, satellite, inst, mode, date=date, pad=pad), 'y', ext)


def clear_cache(path, *args):
	if args[0] is not None:
		a = 'y' if args[0].lower() == 'y' else str(input("Are you sure you want to clear this cache? (Y/n)\t"))
	else:
		a = 'n'
	
	ext = args[1]
	
	if a.lower() == 'y':
		for file in os.listdir(os.path.dirname(path)):
			if file.endswith(ext):
				os.remove(os.path.join(os.path.dirname(path), file))


def load_pickle_safe(pkl_path, cdf_path, key, createfunc, *args):
	"""Load pickle with fallback to regenerate from CDF if incompatible."""
	try:
		if os.path.exists(pkl_path):
			return pd.read_pickle(pkl_path)
	except (ModuleNotFoundError, AttributeError, ImportError) as e:
		print(f"Warning: Pickle file incompatible ({e}). Regenerating from CDF...")
	
	return createfunc(pkl_path, cdf_path, key, args)


def regen_cdf(pickle_path, cdf_path, key, *args):
	try:
		from spacepy import pycdf
		with pycdf.CDF(cdf_path) as cdf:
			var = cdf['mms1_scm_acb_gse_scb_brst_l2'][:]
			time = pd.to_datetime(cdf['Epoch'])

		f = key[0]
		df = pd.DataFrame({'time': time, f'{f}x': var[:, 0], f'{f}y': var[:, 1], f'{f}z': var[:, 2]})

		# Save the regenerated pickle
		os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
		df.to_pickle(pickle_path)
		print(f"Successfully regenerated pickle file: {pickle_path}")

		return df
	except ImportError:
		print("Error: spacepy not available. Cannot regenerate from CDF.")
		print("Please install spacepy or delete the .cache directory and regenerate pickle files.")
		raise

