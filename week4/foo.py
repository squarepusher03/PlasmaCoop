
import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator
from scipy.interpolate import interp1d
from spacepy import pycdf

import draw

RE = 6300 # km
THRESHOLD = 1e4

fgm_files = glob.glob('../themis_data/tha/l2/fgm/2014/*')
state_files = glob.glob('../themis_data/tha/l1/state/2014/*')

dbz6s = pd.Series()

rows = []

for i, f in enumerate(fgm_files):
	with pycdf.CDF(state_files[i]) as cdf:
		var = cdf['tha_pos_gsm']
		postime = pd.to_datetime(cdf['tha_state_time'], unit='s')
		x = pd.Series(var[:, 0])
		y = pd.Series(var[:, 1])
	
	with pycdf.CDF(f) as cdf:
		var = cdf['tha_fgs_gsm']
		time = pd.to_datetime(cdf['tha_fgs_time'], unit='s')
		bx = pd.Series(var[:, 0])
		by = pd.Series(var[:, 1])
		bz = pd.Series(var[:, 2])
	
	row = pd.DataFrame({'time': time, 'Bx': bx, 'By': by, 'Bz': bz, 'x': x, 'y': y})
	
	# Linearly interpolate spacecraft position (x,y) from state times onto FGM times
	# Convert times to int64 nanoseconds for interpolation domain
	postime_i64 = postime.astype('int64')
	time_i64 = time.astype('int64')
	fx = interp1d(postime_i64, x.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
	fy = interp1d(postime_i64, y.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
	row['x'] = fx(time_i64)
	row['y'] = fy(time_i64)
	
	# Apply spatial filters using interpolated positions
	row = row[row['x'] < -6 * RE]
	row = row[row['y'].abs() < (row['x'].abs() / 2)]
	row = row[(row['Bx']**2 + row['By']**2)**0.5 < 15]

	# Drop unused magnetic components to save memory
	del row['Bx']
	del row['By']
	del row['x']
	del row['y']
	del fx
	del fy
	
	rows.append(row)
	
df = None
for row in rows:
	df = pd.concat([df, row], ignore_index=True)

df['dBz6s'] = df['Bz'].shift(-1) - df['Bz'].shift(1)

x = df['Bz'][1:-1]
y = df['dBz6s'][1:-1]
draw.bushit(dfx=x, dfy=y, fill=True)
draw.bushit(dfx=x, dfy=y, fill=True, zoom=((-10, 105), (-1, 1)))
