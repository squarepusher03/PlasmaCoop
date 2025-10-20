import glob

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
from itertools import repeat

RE = 6300  # km
THRESHOLD = 1e4

def cache_path(file):
    """Create a unique pickle path for a given fgm file name."""
    base = os.path.basename(file)
    name, _ = os.path.splitext(base)
    return f'./.cache/{name[-8:-4]}.pkl'

def read_data(fileperinst):
    from spacepy import pycdf

    fgm_file = fileperinst[0]
    state_file = fileperinst[1]
    mom_file = fileperinst[2]

    pkl = cache_path(fgm_file)

    if os.path.exists(pkl):
        row = pd.read_pickle(pkl)
    else:
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

        with pycdf.CDF(mom_file) as cdf:
            itime = pd.to_datetime(cdf['tha_peim_time'], unit='s')
            ivel = cdf['tha_peim_velocity_gsm'][:]
            iden = pd.Series(cdf['tha_peim_density'])  # cm^-3
            itemp = cdf['tha_peim_t3_mag'][:]  # eV, shape (N, 3)

        row = pd.DataFrame({'time': time, 'Bx': bx, 'By': by, 'Bz': bz, 'x': x, 'y': y})
        # Linearly interpolate spacecraft position (x,y) from state times onto FGM times
        # Convert times to int64 nanoseconds for interpolation domain
        postime_i64 = postime.astype('int64', copy=False)
        time_i64 = time.astype('int64', copy=False)
        itime_i64 = itime.astype('int64', copy=False)

        iprp1 = pd.Series(itemp[:, 0])
        iprp2 = pd.Series(itemp[:, 1])
        iprp = (iprp1 + iprp2) / 2
        ipar = pd.Series(itemp[:, 2])
        ivx = pd.Series(ivel[:, 0])
        ivy = pd.Series(ivel[:, 1])

        m = np.isfinite(ipar) & np.isfinite(iprp) & np.isfinite(iden)
        itime_i64 = itime_i64[m]
        ivx, ivy, iden, ipar, iprp = [a[m] for a in (ivx, ivy, iden, ipar, iprp)]

        fx = interp1d(postime_i64, x.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
        fy = interp1d(postime_i64, y.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
        fivx = interp1d(itime_i64, ivx.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
        fivy = interp1d(itime_i64, ivy.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
        fiden = interp1d(itime_i64, iden.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
        fipar = interp1d(itime_i64, ipar.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)
        fiprp = interp1d(itime_i64, iprp.to_numpy(), kind='linear', bounds_error=False, fill_value=np.nan)

        vx = fivx(time_i64)
        vy = fivy(time_i64)
        par = fipar(time_i64)
        prp = fiprp(time_i64)
        ae = (prp / par) - 1
        vxy = (vx ** 2 + vy ** 2) ** 0.5

        row['x'] = fx(time_i64)
        row['y'] = fy(time_i64)
        row['Vxy'] = vxy
        row['density'] = fiden(time_i64)
        row['Ae'] = ae

        del fx
        del fy
        del fivx
        del fivy
        del fiden
        del fipar
        del fiprp

        # Apply spatial filters using interpolated positions
        row = row[row['x'] < -6 * RE]
        row = row[row['y'].abs() < (row['x'].abs() / 2)]
        row = row[(row['Bx'] ** 2 + row['By'] ** 2) ** 0.5 < 15]

        row.drop(columns=['x','y','Bx','By'], inplace=True)

        row.drop(columns=[c for c in row.columns if c not in ['time', 'Bz', 'density', 'Ae', 'Vxy']],
                 inplace=True)

    row.to_pickle(cache_path(fgm_file))

    return row

if __name__ == '__main__':
    import draw
    from spacepy import pycdf

    pycdf.lib.set_backward(False)

    process_data = False

    if process_data:
        state_files = sorted(glob.glob('./themis_data/tha/l1/state/2014/*'))
        fgm_files = sorted(glob.glob('./themis_data/tha/l2/fgm/2014/*'))
        mom_files = sorted(glob.glob('./themis_data/tha/l2/mom/2014/*'))

        file_pairs = zip(fgm_files, state_files, mom_files)

        max_workers = max(1, (os.cpu_count() or 1) - 2)

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            rows = list(ex.map(read_data, file_pairs))

        df = pd.concat(rows, ignore_index=True)

        df = df.sort_values('time').reset_index(drop=True)

        df.to_pickle('processed.pkl')
    else:
        df = pd.read_pickle('processed.pkl')

    draw.title = 'THA FGS vs PEIM'
    draw.name = r'$\forall \Delta B_z,\hspace{0.3}dB_z = 6$ seconds (nT) vs $B_z$ (nT)'

