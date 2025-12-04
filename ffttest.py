import glob
from datetime import datetime, timedelta

import os
from time import strftime

import numpy as np
from matplotlib.patches import Patch
from matplotlib.text import Text
from scipy import fft
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator, MultipleLocator, LogLocator, FixedLocator
from matplotlib import rcParams
from scipy.signal.windows import hann
from matplotlib.widgets import Slider
from scipy.stats import binned_statistic_2d

SCHB_FS = 16_384


def timegraph(ax_time):
    # Top: time-domain Ez with x-limits from start to end and ticks every 0.2 s
    ax_time.set_xlim(start, end)
    ax_time.axvspan(xmin=fft_start, xmax=fft_end, color='maroon', alpha=0.5)

    def dformat(x, pos):
        """Format tick labels as seconds since `start`.

        We avoid timezone/naive-datetime issues by working in Matplotlib's
        float date units (days) and converting the delta to seconds.
        """
        secs = (x - mdates.date2num(start)) * 86400.0
        return f"{secs:.1f}"

    ax_time.xaxis.set_major_formatter(FuncFormatter(dformat))

    #		# Tick every 0.2 seconds, labels show time in seconds only
    ax_time.set_xticks(pd.date_range(start, end, freq='500ms')[:-1])
    ax_time.set_xticks(pd.date_range(start, end, freq='050ms'), minor=True)
    ax_time.tick_params(axis='x', labelrotation=90)
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Ez (nT)')
    ax_time.set_title(f'Ez vs Time starting @ {datetime.strftime(start, "%H:%M:%S.%f")[:-4]} s\n')
    ax_time.grid(True)


def load_pickle_safe(pickle_path, cdf_path, key):
    """Load pickle with fallback to regenerate from CDF if incompatible."""
    try:
        if os.path.exists(pickle_path):
            return pd.read_pickle(pickle_path)
    except (ModuleNotFoundError, AttributeError, ImportError) as e:
        print(f"Warning: Pickle file incompatible ({e}). Regenerating from CDF...")

    # Regenerate from CDF
    try:
        from spacepy import pycdf
        with pycdf.CDF(cdf_path) as cdf:
            var = cdf['mms1_scm_acb_gse_schb_brst_l2'][:]
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


sfn = 'mms_data/mms1/edp/brst/l2/dce/2020/09/02/mms1_edp_brst_l2_dce_20200902062933_v3.0.1.cdf'  # edp source
pkl_path_e = '.cache/edp/dce/20200902062933.pkl'

sfc = 'mms_data/mms1/scm/brst/l2/schb/2020/09/02/mms1_scm_brst_l2_schb_20200902062933_v2.2.0.cdf'  # scm source
pkl_path_s = '.cache/scm/schb/20200902062933.pkl'

paper = 'mms_data/mms1/scm/brst/l2/schb/2019/08/16/mms1_scm_brst_l2_schb_20190816093103_v2.2.0.cdf'
pkl_path_p = '.cache/scm/schb/20190816093145.pkl'
if __name__ == "__main__":
    rcParams['path.simplify'] = True
    rcParams['path.simplify_threshold'] = 0.2
    #rcParams['savefig.format'] = 'svg'
    mpl.use('Qt5Agg')

    lpf_lim = 1000
    hpf_lim = 10
    num_bins = 24
    bin_factor = (lpf_lim) ** (1 / num_bins)

    # all scm caches for graphing all files
    # sfc = glob.glob('mms_data/mms1/scm/brst/l2/schb/2020/09/02/*')
    # pkl_paths = glob.glob('.cache/scm/schb/*')

    sig_key = 'Bz'
    units = r'\frac{\text{nT}^2}{\text{Hz}}' if sig_key[0] == 'B' else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'

    cache_path = paper
    pkl_path = pkl_path_p

    df = load_pickle_safe(pkl_path, cache_path, sig_key)
    df = df.dropna().reset_index(drop=True)

    datefmt = '%m/%d/%Y-%H:%M:%S'
    start = datetime.strptime('08/16/2019-09:31:45', datefmt)
    end = datetime.strptime('08/16/2019-09:32:15', datefmt)

    # Create two axes: top for Ez vs time, bottom for FFT/PSD
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 8), sharex=False, constrained_layout=True
    )

    st = start
    ts = end
    times = [np.arange((st - start).total_seconds(), (ts - start).total_seconds(), 0.05)]

    fdf = pd.DataFrame(columns=['time', 'frequency', 'power'])

    for i in np.arange(0, (ts - st).total_seconds(), 0.05):
        center = st + timedelta(seconds=(float(i)))
        fft_start = center - timedelta(seconds=(float(0.5)))
        fft_end = center + timedelta(seconds=(float(0.5)))

        tdf = df[(df['time'] >= fft_start) & (df['time'] <= fft_end)].reset_index(drop=True)
        if tdf.empty:
            continue
        #signal = tdf[sig_key]
        signal = np.sqrt(tdf[sig_key[0] + 'z'] ** 2 + tdf[sig_key[0] + 'y'] ** 2 + tdf[sig_key[0] + 'x'] ** 2)
        signal = signal - signal.mean()

        n = len(signal)
        w = hann(n, sym=False)
        signal_win = signal * w

        # FFT and frequency axis (one-sided)
        signal_fft = fft.rfft(signal_win)
        signal_fft_freq = np.fft.rfftfreq(n, d=1 / SCHB_FS)  # Hz

        # Hann window power normalization for PSD (units: nT^2/Hz)
        U = np.sum(w ** 2)  # window power of Hann
        signal_psd = (np.abs(signal_fft) ** 2) / (SCHB_FS * U)

        # One-sided correction (conserve variance)
        if n % 2 == 0:
            # even n: DC at 0, Nyquist present at last index
            if signal_psd.size > 2:
                signal_psd[1:-1] *= 2
        else:
            # odd n: no Nyquist bin
            if signal_psd.size > 1:
                signal_psd[1:] *= 2

        mask = (signal_fft_freq > 0) & (signal_fft_freq <= lpf_lim)

        f_sel = signal_fft_freq[mask]
        P_sel = signal_psd[mask]
        fdf = pd.concat([fdf, pd.DataFrame({
            'time': [(center - start).total_seconds()] * len(f_sel),
            'frequency': f_sel,
            'power': P_sel})], ignore_index=True)

    i = lpf_lim
    bins = list()
    while i > 2: # TODO: count how many bins are in the graph and update this
        bins.append(int(i))
        i = i / bin_factor
    bins.append(1)
    bins.reverse()

    def channelize(x):
        if x == lpf_lim:
            return len(bins) - 1
        for i in range(1, len(bins)):
            if bins[i - 1] <= x < bins[i]:
                return i
        return None

    vfunc = np.vectorize(channelize)
    fdf['channel'] = vfunc(fdf['frequency'])

    # Bottom: PSD (FFT) plot
    time_bins = np.arange(fdf['time'].min(), fdf['time'].max() + 0.06, 0.05)
    ax.yaxis.set_major_locator(NullLocator())
    ax.xaxis.set_major_locator(MultipleLocator(1, 1))

    stat, xe, ye, _ = binned_statistic_2d(fdf['time'], fdf['channel'], fdf['power'],
                                          statistic='median', bins=[time_bins, list(range(1, len(bins) + 1))])

    mappable = plt.pcolormesh(xe, ye, stat.T,
                   norm=mpl.colors.LogNorm(), cmap='jet')
    cbar = plt.colorbar(mappable=mappable, ax=ax)
    cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')

    #ax.set_ylim(bottom=7)
    ax.set_xlim(right=30.5)

    lg_lbls = [f'Channel {i}: {bins[i-1]} Hz - {bins[i]} Hz' for i in range(1, len(bins))]
    proxies = [Patch(color='none') for _ in range(len(lg_lbls))]

    ax.legend(proxies, lg_lbls, handlelength=0, handletextpad=0)

    #ax.hist2d(fdf['time'], fdf['frequency'], weights=fdf['power'], bins=[time_bins, freq_bins],
    #         cmap='viridis', norm=mpl.colors.LogNorm(), cmin=10e-12, cmax=10e-6)

    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(fr'$|{sig_key[0]}|$ Frequency vs. Time vs. $|{sig_key[0]}|$ Power ${units}$')
    ax.set_xlabel('Time (s)')

#    ytx = ax.get_yticks()
#    ytl = [str(int(i)) for i in ytx[:-2]] + ['']
    #ax.set_yticklabels(ytl)

    #plt.show()
    plt.savefig('./out/scm/fft/20bins.png')