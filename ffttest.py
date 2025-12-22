import glob
from datetime import datetime, timedelta

import os

import numpy as np
from matplotlib.patches import Patch
from scipy import fft
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import log
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator, MultipleLocator, LogLocator, FixedLocator
from matplotlib import rcParams
from scipy.signal.windows import hann
from scipy.stats import binned_statistic_2d

from caching import load_pickle_safe, regen_cdf, gen_path, clear_fft

SCB_FS = 8_192
SCHB_FS = 16_384


def gen_fft(pickle_path, cdf_path, key, *args):
    st = args[0][1]
    ts = args[0][2]
    inst = args[0][3].lower()
    pad = args[0][0]
    
    df = load_pickle_safe(gen_path(inst=inst, process='cdf', date=st, pad=pad), cdf_path, key, regen_cdf)
    
    vr = np.vectorize(lambda x: round(x))
    Df = 1 / pad
    
    match inst:
        case 'schb':
            fs = SCHB_FS
        case 'scb':
            fs = SCB_FS
        case _:
            fs = SCB_FS
        
    m = 0
    fdf = pd.DataFrame(columns=['time', 'frequency', 'power'])
    
    for i in np.arange(0, (ts - st).total_seconds() + 0.01, 0.05):
        center = st + timedelta(seconds=(float(i)))
        fft_start = center - timedelta(seconds=(float(int(pad))))
        fft_end = center + timedelta(seconds=(float(pad)))
        
        tdf = df[(df['time'] >= fft_start) & (df['time'] <= fft_end)].reset_index(drop=True)
        # signal = tdf[sig_key]
        signal = np.sqrt(tdf[key[0] + 'z'] ** 2 + tdf[key[0] + 'y'] ** 2 + tdf[key[0] + 'x'] ** 2)
        signal = signal - signal.mean()
        
        n = len(signal)
        m += n
        w = hann(n, sym=False)
        signal_win = signal * w
        
        # FFT and frequency axis (one-sided)
        signal_fft = fft.rfft(signal_win)
        signal_fft_freq = np.fft.rfftfreq(n, d=1 / fs)  # Hz
        signal_fft_freq = vr(signal_fft_freq / Df) * Df
        
        # Hann window power normalization for PSD (units: nT^2/Hz) ONLY FOR POWER SPECTRA
        U = np.sum(w ** 2) / n  # window power of Hann DON'T SQUARE FOR AMPLITUDE
        signal_psd = (np.abs(signal_fft) ** 2) / (n * U * fs)
        
        # One-sided correction (conserve variance)
        if n % 2 == 0:
            # even n: DC at 0, Nyquist present at last index
            if signal_psd.size > 2:
                signal_psd[1:-1] *= 2
        else:
            # odd n: no Nyquist bin
            if signal_psd.size > 1:
                signal_psd[1:] *= 2
        
        mask = (signal_fft_freq >= 0) & (signal_fft_freq <= lpf_lim)
        
        f_sel = signal_fft_freq[mask]
        P_sel = signal_psd[mask]
        fdf = pd.concat([fdf, pd.DataFrame({
            'time': [(center - st).total_seconds()] * len(f_sel),
            'frequency': f_sel,
            'power': P_sel})], ignore_index=True)
    
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    fdf.to_pickle(pickle_path)
    
    print(f"Successfully generated pickle file: {pickle_path}")
    
    return fdf


paper = 'mms_data/mms1/scm/brst/l2/schb/2019/08/16/mms1_scm_brst_l2_schb_20190816093103_v2.2.0.cdf'
pkl_path_p = '.cache/mms/1/scm/schb/20190816093145.pkl'

paperb = 'mms_data/mms1/scm/brst/l2/scb/2019/08/16/mms1_scm_brst_l2_scb_20190816093103_v2.2.1.cdf'
pkl_path_b = '.cache/mms/1/scm/scb/20190816093145.pkl'

pkl_path_pr = '.cache/mms/1/scm/fft/scb/20190816093145.pkl'
if __name__ == "__main__":
#	rcParams['path.simplify'] = True
#	rcParams['path.simplify_threshold'] = 0.2
    #rcParams['savefig.format'] = 'svg'
    mpl.use('Qt5Agg')
    
    lpf_lim = 1000
    hpf_lim = 10
    num_bins = 40
    bin_factor = (lpf_lim) ** (1 / num_bins)

    cdf_path = paperb
    
    # all scm caches for graphing all files
    # sfc = glob.glob('mms_data/mms1/scm/brst/l2/schb/2020/09/02/*')
    # pkl_paths = glob.glob('.cache/scm/schb/*')
    
    sig_key = 'Bz'
    units = r'\frac{\text{nT}^2}{\text{Hz}}' if sig_key[0] == 'B' else r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}'
    
    datefmt = '%m/%d/%Y-%H:%M:%S'
    start = datetime.strptime('08/16/2019-09:31:56', datefmt)
    end = datetime.strptime('08/16/2019-09:32:00', datefmt)
    
    # Create two axes: top for Ez vs time, bottom for FFT/PSD
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 12), sharex=True, constrained_layout=True
    )
    
    #ax = axes  #[0]
    #ax1 = axes[1]

    i = lpf_lim
    bins = list()
    j = 0
    while i > 2:
        ti = int(i)
        try:
            if (len(bins) > 0 and bins[len(bins) - 1] - ti > 0) or len(bins) == 0:
                bins.append(ti)
        except Exception as e:
            print(f'! {e}: [i: {i}, ti: {ti}, j: {j}]\n\t[bins: {bins}]')

        i = i / bin_factor
        j += 1
    bins.append(0)
    bins.reverse()

    def channelize(x):
        if x == bins[0]:
            return 0
        else:
            for i in range(1, len(bins)):
                if bins[i - 1] < x <= bins[i]:
                    return i
        
        if x == lpf_lim + 1:
            return len(bins)

        print(f'wtf: {x}')
        return None

    vfunc = np.vectorize(channelize)

    #clear_fft(date=start) # deletes all pkl files for start date, must be adjusted for other

    name = gen_path(ext='xlsx')
    if os.path.exists(name):
        writer = pd.ExcelWriter(name, mode='a', if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(name, mode='w')

    f = lambda x: int(x)
    vf = np.vectorize(f)
    logbf = lambda x: log(x, bin_factor)
    llogbf = lambda x: [logbf(i) for i in x]
    vlogbf = np.vectorize(logbf)

    time_bins = np.arange(0, (end - start).total_seconds() + 0.06, 0.05)

    fig.suptitle(
        fr'$|{sig_key[0]}|$ Frequency vs. Time vs. $|{sig_key[0]}|$ Power ${units}$ starting from {start.hour:02d}:{start.minute:02d}:{start.second:02d}')

    for j, i in enumerate([0.1, 0.25, 0.5]):
        ax = axes[j]
        pkl_path_real = gen_path(date=start, pad=i)
        fdf = load_pickle_safe(pkl_path_real, cdf_path, sig_key, gen_fft, i, start, end, 'scb')
        fdf.loc[fdf['frequency'] == 0, 'frequency'] = 1
        #fdf['frequency'] = vf(fdf['frequency'])
        fdf['channel'] = vfunc(fdf['frequency'])

        tdf = fdf[(fdf['time'] == 0.95) | (fdf['time'] == 1.00)].reset_index(drop=True)
        tdf.to_excel(writer, sheet_name=f'{i * 2} sec interval')
        
        # Bottom: PSD (FFT) plot
        #fdf['frequency'] = vlogbf(fdf['frequency'])
        fdf['time'] += 1e-12

        #ax.yaxis.set_major_locator(NullLocator())
        #ax.yaxis.set_major_locator(MultipleLocator(1))
        #ax.xaxis.set_major_locator(MultipleLocator(0.05, 0))
        #ax.tick_params('x', rotation=90)

        ax.xaxis.set_major_locator(MultipleLocator(0.5, 1))
        ax.yaxis.set_major_locator(FixedLocator([10, 100, 1001]))
        ax.set_yticklabels([r'$10^1$', r'$10^2$', r'$10^3$'])

        stat, xe, ye, bn = binned_statistic_2d(fdf['time'], fdf['frequency'], fdf['power'],
                                               statistic='mean', bins=[time_bins, fdf['frequency'].unique()])

        mappable = ax.pcolormesh(xe, ye, stat.T,
                                 norm=mpl.colors.LogNorm(vmin=1e-6, vmax=1e-2), cmap='jet')
        cbar = plt.colorbar(mappable=mappable, ax=ax)
        cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')

        ax.set_ylim(bottom=10, top=lpf_lim + 1)
        ba = ax.get_yticklabels()
        ba[-1].set_text(r'$10^3$')
        ba[-2].set_text('')

        lg_lbls = [f'Channel {i}: [{bins[i - 1]} - {bins[i]}) Hz' for i in range(1, len(bins) - 1)]
        lg_lbls.append(f'Channel {len(bins) - 1}: [{bins[len(bins) - 2]} - {lpf_lim}] Hz')
        proxies = [Patch(color='none') for _ in range(len(lg_lbls))]

        #ax.grid()
        #ax.legend(proxies, lg_lbls, handlelength=0, handletextpad=0)
        ax.set_title(rf'FFT taken with {i * 2} sec window / $\Delta f = {int(i ** -1)}$')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')

        #    ytx = ax.get_yticks()
        #    ytl = [str(int(i)) for i in ytx[:-2]] + ['']
        #ax.set_yticklabels(ytl)

        #	fdf = tdf[fdf['time'] == 2.45].drop(columns='time')
        #	ax1.plot(fdf['frequency'], fdf['power'], color='red')

        #	fdf = load_pickle_safe(pkl_path_real, cache_path, sig_key, gen_fft, 0.25, start, end)
        #	fdf = tdf[tdf['time'] == 2.45].drop(columns='time')
        #	ax1.plot(fdf['frequency'], fdf['power'], color='blue')
        #
        #	ax1.set_xticks([10,] + list(range(100, 1001, 100)))
        #	ax1.set_xlim(left=10, right=1000)
        #	ax1.set_title(fr'$|{sig_key[0]}|$ Power Spectra at 09:31:58.45')
        #	ax1.set_xlabel('Frequency (Hz)')
        #	ax1.set_ylabel(rf'Power $\left({units}\right)$')
        #	ax1.set_yscale('log')
        #	ax1.legend(labels=['0.2 second window', '0.5 second window'])

    plt.show()
    out = './out/scm/fft/scb20.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    #plt.savefig(out)

    writer.close()

