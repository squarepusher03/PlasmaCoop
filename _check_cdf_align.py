"""Verify CDF Epoch alignment across MMS probes for three burst windows."""
from datetime import datetime
import numpy as np
import pandas as pd
import glob
import re
import os
from spacepy import pycdf

from data_paths import DATA_DIR as DATA  # per-platform E:/ vs /mnt/e


def find_file(probe, start_dt):
    """Find the burst CDF on `probe` that brackets start_dt (rightmost with file_time <= start_dt)."""
    pattern = f'{DATA}/mms{probe}/scm/brst/l2/scb/**/*.cdf'
    cands = []
    for p in glob.glob(pattern, recursive=True):
        m = re.search(r'_(\d{14})_', p)
        if m:
            ft = datetime.strptime(m.group(1), '%Y%m%d%H%M%S')
            cands.append((ft, p))
    cands.sort()
    chosen = None
    for ft, p in cands:
        if ft <= start_dt:
            chosen = (ft, p)
        else:
            break
    return chosen


def load(probe, path):
    with pycdf.CDF(path) as cdf:
        ep  = pd.Series(pd.to_datetime(cdf['Epoch'][:]))
        var = cdf[f'mms{probe}_scm_acb_gse_scb_brst_l2'][:].copy()
    return ep, var


def check_window(label, probes, start, end):
    print(f'\n=== {label} : {start} -> {end} ===')
    loaded = {}
    for p in probes:
        hit = find_file(p, start)
        if hit is None:
            print(f'  MMS{p}: NO FILE bracketing this start')
            continue
        ft, path = hit
        ep, var = load(p, path)
        print(f'  MMS{p}: file_ts={ft}  n={len(ep)}  first={ep.iloc[0]}  last={ep.iloc[-1]}')
        if ep.iloc[0] > end or ep.iloc[-1] < start:
            print(f'         (file does not actually cover the requested window)')
        loaded[p] = (ft, ep, var, path)

    # Pairwise alignment within the window
    probes_with_data = sorted(loaded.keys())
    for i in range(len(probes_with_data)):
        for j in range(i+1, len(probes_with_data)):
            pa, pb = probes_with_data[i], probes_with_data[j]
            fta, ea, va, _ = loaded[pa]
            ftb, eb, vb, _ = loaded[pb]
            # restrict to requested window
            sa = np.searchsorted(ea.values, np.datetime64(start, 'ns'), side='left')
            ha = np.searchsorted(ea.values, np.datetime64(end,   'ns'), side='right')
            sb = np.searchsorted(eb.values, np.datetime64(start, 'ns'), side='left')
            hb = np.searchsorted(eb.values, np.datetime64(end,   'ns'), side='right')
            na, nb = ha - sa, hb - sb
            print(f'  MMS{pa} vs MMS{pb} in window: n_a={na}  n_b={nb}  diff={abs(na-nb)}')
            if na == 0 or nb == 0:
                continue
            n = min(na, nb)
            t_a = ea.iloc[sa:sa+n].values.astype('datetime64[ns]').astype(np.int64)
            t_b = eb.iloc[sb:sb+n].values.astype('datetime64[ns]').astype(np.int64)
            d   = t_a - t_b
            print(f'    first sample MMS{pa}={ea.iloc[sa]}  MMS{pb}={eb.iloc[sb]}  start_delta_us={(t_a[0]-t_b[0])/1e3:.3f}')
            print(f'    Epoch_a - Epoch_b ns: min={d.min()} max={d.max()} mean={d.mean():.1f} std={d.std():.1f}')
            # sample interval consistency
            dta = np.diff(t_a)
            dtb = np.diff(t_b)
            print(f'    MMS{pa} sample dt ns: median={int(np.median(dta))} (={1e9/np.median(dta):.1f} Hz)')
            print(f'    MMS{pb} sample dt ns: median={int(np.median(dtb))} (={1e9/np.median(dtb):.1f} Hz)')


# Window 1: paper event — MMS1 ∩ MMS2 (MMS3 has no data this far back)
check_window('A: 2019-08-16 paper event',
             probes=[1, 2, 3],
             start=datetime(2019, 8, 16, 9, 31, 45),
             end  =datetime(2019, 8, 16, 9, 32, 15))

# Window 2: pick a 30s window inside MMS3's 2022-01-01 02:29:53 burst
check_window('B: 2022-01-01 burst (MMS2/MMS3)',
             probes=[1, 2, 3],
             start=datetime(2022, 1, 1, 2, 30, 0),
             end  =datetime(2022, 1, 1, 2, 30, 30))

# Window 3: 2021-10-01 19:59:33 burst (MMS3 first available file)
check_window('C: 2021-10-01 burst (MMS2/MMS3)',
             probes=[1, 2, 3],
             start=datetime(2021, 10, 1, 19, 59, 45),
             end  =datetime(2021, 10, 1, 20, 0, 15))
