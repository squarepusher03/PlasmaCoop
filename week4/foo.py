import glob

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d

RE = 6300  # km
THRESHOLD = 1e4


def read_data(fileperinst):
    from spacepy import pycdf

    state_file = fileperinst[0]
    fgm_file = fileperinst[1]
    sst_file = fileperinst[2]

    with pycdf.CDF(state_file) as cdf:
        var = cdf['tha_pos_gsm']
        postime = pd.to_datetime(cdf['tha_state_time'], unit='s')
        x = pd.Series(var[:, 0])
        y = pd.Series(var[:, 1])

    with pycdf.CDF(fgm_file) as cdf:
        var = cdf['tha_fgs_gsm']
        time = pd.to_datetime(cdf['tha_fgs_time'], unit='s')
        bx = pd.Series(var[:, 0])
        by = pd.Series(var[:, 1])
        bz = pd.Series(var[:, 2])

    with pycdf.CDF(sst_file) as cdf:
        ifluxtime = pd.to_datetime(cdf['tha_psif_time'], unit='s').astype('int64', copy=False)
        efluxtime = pd.to_datetime(cdf['tha_psef_time'], unit='s').astype('int64', copy=False)
        Fi = cdf['tha_psif_en_eflux'][:, 3]
        Fe = cdf['tha_psef_en_eflux'][:, 5]

    row = pd.DataFrame({'time': time, 'Bx': bx, 'By': by, 'Bz': bz, 'x': x, 'y': y})
    Fi[Fi == 0] = 1
    Fe[Fe == 0] = 1

    # Linearly interpolate spacecraft position (x,y) from state times onto FGM times
    # Convert times to int64 nanoseconds for interpolation domain
    postime_i64 = postime.astype('int64', copy=False)
    time_i64 = time.astype('int64', copy=False)

    fx = interp1d(postime_i64, x.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
    fy = interp1d(postime_i64, y.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
    fi = interp1d(ifluxtime, Fi, kind='linear', bounds_error=False, fill_value=np.nan)
    fe = interp1d(efluxtime, Fe, kind='linear', bounds_error=False, fill_value=np.nan)

    row['x'] = fx(time_i64)
    row['y'] = fy(time_i64)
    row['Fi'] = fi(time_i64)
    row['Fe'] = fe(time_i64)

    # Apply spatial filters using interpolated positions
    row = row[row['x'] < -6 * RE]
    row = row[row['y'].abs() < (row['x'].abs() / 2)]
    row = row[(row['Bx'] ** 2 + row['By'] ** 2) ** 0.5 < 15]

    row.drop(columns=[c for c in row.columns if c not in ['time', 'Bz', 'Fi', 'Fe']], inplace=True)

    del fx
    del fy
    del fi
    del fe

    return row


if __name__ == '__main__':
    import draw
    from spacepy import pycdf

    process_data = False

    if process_data:
        fgm_files = sorted(glob.glob('../themis_data/tha/l2/fgm/2014/*'))
        state_files = sorted(glob.glob('../themis_data/tha/l1/state/2014/*'))
        sst_files = sorted(glob.glob('../themis_data/tha/l2/sst/2014/*'))

        file_pairs = list(zip(state_files, fgm_files, sst_files))

        max_workers = max(1, (os.cpu_count() or 1) - 2)

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            rows = list(ex.map(read_data, file_pairs))

        df = pd.concat(rows, ignore_index=True)

        df = df.sort_values('time').reset_index(drop=True)

        df['dBz6s'] = df['Bz'].shift(-1) - df['Bz'].shift(1)

        df.to_pickle('processed.pkl')
    else:
        df = pd.read_pickle('processed.pkl')

    x = df['Bz'][1:-1]
    y = df['dBz6s'][1:-1]
    z1 = df['Fi'][1:-1]
    z2 = df['Fe'][1:-1]

    draw.title = "THA FGS vs PSEF"
    draw.name = r'$\forall \Delta B_z, dB_z = 6$ seconds (nT) vs $B_z$ (nT) vs $\bar{F_i}$ (kEv)'

    #draw.draw_profile3v(x, y, z1, 'ion fluxes')
    #draw.draw_profile3v(x, y, z1, 'ion fluxes', zoom=((60, 120), (-5, 5)))
    draw.draw_profile3v(x, y, z2, 'electron fluxes', zoom=((20, 120), (-10, 10)))

