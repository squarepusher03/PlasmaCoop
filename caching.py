import os
from datetime import datetime

import pandas as pd

def load_pickle_safe(pkl_path, cdf_path, key, createfunc, *args):
	"""Load pickle with fallback to regenerate from CDF if incompatible."""
	try:
		if os.path.exists(pkl_path):
			return pd.read_pickle(pkl_path)
	except (ModuleNotFoundError, AttributeError, ImportError) as e:
		print(f"Warning: Pickle file incompatible ({e}). Regenerating from CDF...")
	
	return createfunc(pkl_path, cdf_path, key, args)


def regen_cdf(pickle_path, cdf_path, key, *args):
	print('Regenerating from CDF...')
	try:
		from spacepy import pycdf
		print(cdf_path)
		with pycdf.CDF(cdf_path) as cdf:
			var = cdf['mms1_scm_acb_gse_scb_brst_l2'][:]
			time = pd.to_datetime(cdf['Epoch'])

		f = key[0]
		df = pd.DataFrame({'time': time, f'{f}x': var[:, 0], f'{f}y': var[:, 1], f'{f}z': var[:, 2]})

		# Save the regenerated pickle
		os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
		df.to_pickle(pickle_path)
		print(f"Successfully regenerated CDF pickle file: {pickle_path}")

		return df
	except ImportError:
		print("Error: spacepy not available. Cannot regenerate from CDF.")
		print("Please install spacepy or delete the .cache directory and regenerate pickle files.")
		raise

