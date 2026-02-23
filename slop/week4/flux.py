
import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy.interpolate import interp1d
from spacepy import pycdf

RE = 6300 # km
THRESHOLD = 1e4

fgm_files = glob.glob('../themis_data/tha/l2/fgm/2014/*')
sst_files = glob.glob('../themis_data/tha/l2/sst/2014/*')
state_files = glob.glob('../themis_data/tha/l1/state/2014/*')

dbz6s = pd.Series(dtype=float)

rows = []
def yformat(val, pos):
    # This rounds the exponent to avoid tiny fp errors
    if val >= THRESHOLD:
        exp = int(np.round(np.log10(val)))
        return r"$10^{%d}$" % exp
    elif val <= -THRESHOLD:
        exp = int(np.round(np.log10(np.abs(val))))
        return r"$-10^{%d}$" % exp
    else:
        return '%d' % int(val)   # empty string => no label shown

for i, f in enumerate(fgm_files):
    with pycdf.CDF(f) as cdf:
        var = cdf['tha_fgs_gsm']
        time = pd.to_datetime(cdf['tha_fgs_time'], unit='s')
        bx = pd.Series(var[:, 0])
        by = pd.Series(var[:, 1])
        bz = pd.Series(var[:, 2])
    
    with pycdf.CDF(state_files[i]) as cdf:
        var = cdf['tha_pos_gsm']
        postime = pd.to_datetime(cdf['tha_state_time'], unit='s')
        x = pd.Series(var[:, 0])
        y = pd.Series(var[:, 1])
    
    # Linearly interpolate spacecraft position (x,y) from state times onto FGM times
    # Convert times to int64 nanoseconds for interpolation domain
    postime_i64 = postime.to_numpy(dtype='datetime64[ns]').astype('int64')
    time_i64 = time.to_numpy(dtype='datetime64[ns]').astype('int64')
    fx = interp1d(postime_i64, x.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
    fy = interp1d(postime_i64, y.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
    
    interx = fx(time_i64)
    intery = fy(time_i64)
    
    with pycdf.CDF(sst_files[i]) as cdf:
        ifluxtime = pd.to_datetime(cdf['tha_psif_time'], unit='s').to_numpy(dtype='datetime64[ns]').astype('int64')
        efluxtime = pd.to_datetime(cdf['tha_psef_time'], unit='s').to_numpy(dtype='datetime64[ns]').astype('int64')
        Fi = cdf['tha_psif_en_eflux'][:, 3]
        Fe = cdf['tha_psef_en_eflux'][:, 5]
        
    fi = interp1d(ifluxtime, Fi, kind='linear', bounds_error=False, fill_value=np.nan)
    fe = interp1d(efluxtime, Fe, kind='linear', bounds_error=False, fill_value=np.nan)
    
    interfi = fi(time_i64)
    interfe = fe(time_i64)
    
    row = pd.DataFrame({'time': time, 'Bx': bx, 'By': by, 'Bz': bz,
                        'x': interx, 'y': intery, 'Fi': interfi, 'Fe': interfe})
    
    del fx
    del fy
    del fi
    del fe

    rows.append(row)
    
filtered_rows = []
for row in rows:
    # Apply spatial filters using interpolated positions
    row = row[(row['Bx'] ** 2 + row['By'] ** 2) ** 0.5 < 15]
    
    row = row[row['x'] < -6 * RE]
    row = row[row['y'].abs() < (row['x'].abs() / 2)]
    
    # Drop unused magnetic components to save memory
    del row['Bx']
    del row['By']
    del row['x']
    del row['y']
    
    filtered_rows.append(row)

# Concatenate all filtered rows at once to avoid deprecated/ambiguous concat patterns
if filtered_rows:
    df = pd.concat(filtered_rows, ignore_index=True)
else:
    df = pd.DataFrame(columns=['time', 'Bz', 'Fi', 'Fe'])

df['dBz6s'] = df['Bz'].shift(-1) - df['Bz'].shift(1)
df['dFi'] = df['Fi'].shift(-1) - df['Fi'].shift(1)
df['dFe'] = df['Fe'].shift(-1) - df['Fe'].shift(1)
df['pFi'] = (df['Fi'].shift(-1) + df['Fi'].shift(1)) / 2
df['pFe'] = (df['Fe'].shift(-1) + df['Fe'].shift(1)) / 2
df['rFi'] = df['dFi'] / df['pFi']
df['rFe'] = df['dFe'] / df['pFe']

fig, ax = plt.subplots(figsize=(8,5))
fig.suptitle('Tha SST ion fluxes vs electron fluxes')

ax.set_xlabel(r'$\Delta F_i$ ion flux $(^{\text{kEv}}\!\!/_{6\text{ sec}})$ normalized by their mean between ion fluxes')
ax.set_ylabel(r'$\Delta F_e$ $\text{e}^-$ flux $(^{\text{kEv}}\!\!/_{6\text{ sec}})$ normalized by their mean between $\text{e}^-$ fluxes')

# Replace inf values with NaN in rFi and rFe, then drop any rows where either is NaN
# Note: dropping NaNs on a single Series does not remove corresponding rows from the DataFrame.
# Use dropna with subset to remove rows where rFi or rFe are missing.
df[['rFi', 'rFe']] = df[['rFi', 'rFe']].replace([np.inf, -np.inf], np.nan)
# Also remove the first/last rows affected by shift if they are NaN; subset ensures row-wise drop
df.dropna(subset=['rFi', 'rFe'], inplace=True)

#ax.set_yscale('log')
h, xedges, yedges, im = ax.hist2d(df['rFe'].to_numpy(), df['rFi'].to_numpy(), range=((-1, 1), (-0.3, 0.3)),
                                  norm=LogNorm(), bins=100)
# Ensure no empty bins: assign a minimum count of 1 to all bins for coloring
h_filled = h.copy()
h_filled[h_filled == 0] = 1
im.set_array(h_filled.ravel())
im.set_clim(1, h_filled.max())

# Force scientific notation on both axes tick labels (no offset text)
xfmt = ScalarFormatter(useMathText=True)
xfmt.set_scientific(True)
xfmt.set_powerlimits((0, 0))
xfmt.set_useOffset(False)
ax.xaxis.set_major_formatter(xfmt)

yfmt = ScalarFormatter(useMathText=True)
yfmt.set_scientific(True)
yfmt.set_powerlimits((0, 0))
yfmt.set_useOffset(False)
ax.yaxis.set_major_formatter(yfmt)

cbar = fig.colorbar(im, ax=ax)
cbar.ax.yaxis.set_major_formatter(FuncFormatter(yformat))
cbar.set_label('Counts (log scale)')

ax.grid(True, which='major', color='grey', linestyle='-', linewidth=1, alpha=0.5)

# Use a linear, readable scale focused on [-100, 100] with 4 ticks above and below 0

plt.tight_layout()
#plt.show()
plt.savefig('fluxlogcbfill')
