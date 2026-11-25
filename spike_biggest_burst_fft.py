"""
SPIKE / TEMP — stacked FFT spectrograms of the SCM |B| and/or EDP |E| field.

Set FIELDS in the config below: ['B'] (SCM magnetic, scb mode), ['E'] (EDP electric,
dce mode), or ['B', 'E'] to render BOTH side by side — B in the left column, E in
the right. Each entry becomes a figure COLUMN. Both fields are sampled at the SCB
rate (SAMPLE_RATE, 8192 Hz) by gen_fft, so the window/Df/log-bins and the whole pipeline
are identical for either; only the per-field block in field_spec() (key, units, color
range, instrument label) changes. B and E have different units/ranges, so each column
carries its own LogNorm + colorbar; the f_ce overlay is field-independent and drawn
on every column.

Produces one power-spectrogram row per spacecraft (MMS1 top -> MMS4 bottom) per field
column, sharing a time axis. The y-axis uses Graham's log-binning logic (make_log_bins
/ channelize, copied verbatim from ffttest.py) with fine log bins up to
lpf_lim = 3000 Hz. Each FFT point is filed into a discrete log channel and drawn
as discrete pcolormesh cells.

The window is set via FIXED_WINDOW; with FIXED_WINDOW = None the "biggest burst"
(largest RMS of the detrended |B|/|E| fluctuation on the reference probe) is
selected instead. The same time window is plotted for all four spacecraft. If a
probe is missing data for the window, its panel is drawn empty + labeled. Output
PDFs are named spectro_{FIELDS}_... (e.g. spectro_B_, spectro_E_, spectro_BE_) so
different field selections never collide.

Runs on Windows or WSL/Linux: the E:/ data drive (E:/ on Windows, /mnt/e on WSL)
is resolved per-platform in the config below. Needs matplotlib, spacepy/pycdf.
"""

from datetime import datetime, timedelta

import os
import numpy as np
import pandas as pd
import matplotlib as mpl

from ffttest import (find_cdf_files, find_data_pickles, ensure_data_pickle, gen_fft,
                     SAMPLE_RATE, field_burst, _compute_and_save_fft, _fft_pickle_path,
                     check_cdfs, cdf_check, find_local_cdfs, read_burst_data,
                     read_burst_signal, _archived_row)
from acquisition import acquire

# ---------------------------------------------------------------- config
# Data paths resolve per-platform (E:/ on Windows, /mnt/e on WSL); see data_paths.
import data_paths
data_dir      = data_paths.DATA_DIR
pickle_root   = data_paths.PICKLE_ROOT
corrupted_log = data_paths.CORRUPTED_LOG

DAY        = datetime(2021, 2, 23)
# Optional explicit (start, end). If set, plots this exact window and skips the
# biggest-burst search. Set to None to auto-select the day's largest-RMS segment.
# Deliberately set to a long (multi-day) span here on purpose: this is also used
# to exercise the long-interval calculation path itself (perf/capability testing),
# not just to plot an actual burst.
FIXED_WINDOW = (datetime(2021, 2, 23, 11, 7, 33),
                datetime(2021, 3, 2, 11, 7, 33))
PROBES     = [1, 2, 3]
REF_PROBE  = PROBES[0]            # segment timestamps are identical across probes; MMS1 is representative

# Fields to render, left-to-right as figure COLUMNS. 'B' = SCM magnetic (scb),
# 'E' = EDP electric (dce). Both are sampled at the SCB rate (SAMPLE_RATE, 8192 Hz) by
# gen_fft, so the window/Df/log-bins and the whole pipeline are identical for either
# — only the per-field block in field_spec() differs. One entry -> a single-column
# figure (the original behaviour); two -> B in the left column and E in the right,
# sharing the time axis and the (field-independent) f_ce overlay.
FIELDS = ['E']


def field_spec(field):
    """Per-field display settings: signal key, instrument label, RMS unit, power
    units, and LogNorm range. key[0] selects the instrument in ffttest
    ('B' -> scm/scb, 'E' -> edp/dce). Each field carries its OWN color range, so a
    two-field figure needs one norm + colorbar per column (B is nT^2/Hz, E is
    mV^2/m^2/Hz — they cannot share a scale)."""
    if field == 'B':
        return {'sig_key': 'Bz', 'inst': 'SCM/SCB', 'rms_unit': 'nT',
                'units': r'\frac{\text{nT}^2}{\text{Hz}}',
                'vmin': 1e-6, 'vmax': 1e-2}   # LogNorm power range (nT^2/Hz)
    return {'sig_key': 'Ez', 'inst': 'EDP/DCE', 'rms_unit': 'mV/m',
            'units': r'\frac{\text{mV}^2}{\text{m}^2 \cdot \text{Hz}}',
            'vmin': 1e-4, 'vmax': 1e1}        # LogNorm power range (mV^2/m^2/Hz)

PAD        = 0.25         # half-window (s); full window = 0.5 s, Df = 2 Hz
STEP       = 0.10         # window-center stride (s); 0.5 s window -> 80% overlap
LPF_LIM    = 3000         # Hz, top of the frequency range / top cell
NUM_BINS   = 40           # sets log-bin density via BIN_FACTOR (actual channel count ~30)
BIN_FACTOR = LPF_LIM ** (1 / NUM_BINS)  # ~1.22; fine log bins (ffttest scheme), not octaves
DISPLAY_DT = 0.10         # spectrogram COLUMN width (s); == STEP -> one FFT result per
                          # column. This is the time-AXIS cell, NOT the FFT window length
                          # (= 2*PAD); the transform length stays driven by PAD.
# Cap on display columns. At DISPLAY_DT the column count is span/0.1, so a week is
# ~6M columns x ~32 channels = 193M cells: each grid (p_sum/p_cnt/mean) is ~1.44 GiB
# and matplotlib needs another ~1.44 GiB to color them at savefig -> OOM. It is also
# pointless — an 11-inch figure resolves only a few thousand columns. When the window
# would exceed this, the COLUMN width is widened so the grid stays bounded; binning
# already averages every FFT that lands in a column, so wider columns just fold more
# DISPLAY_DT cells (FFT steps) together (the right behaviour for a multi-day overview).
# This only changes the display time axis — the FFT window (2*PAD) is unaffected.
# 200k cols x 32 chan keeps the three grids ~150 MB total (vs ~4.3 GiB for a week
# at 0.1 s) while preserving native 0.1 s cells for any window up to ~5.5 h; only
# longer windows coarsen (a week -> ~3 s cells). Rasterized rendering then composites
# the cells to the figure's pixel grid, so the count never bloats the PDF.
MAX_DISPLAY_COLS = 200_000

SAVE_DIR   = './out/fft'  # output filename is built per-window in main()

# Burst-interval manifest: when FFTs are calculated, gen_fft writes one row per
# contiguous recorded burst (start/end read from each CDF's Epoch, clipped to the
# request) to {pickle_root}/mms/{probe}/{inst}/{mode}/intervals/. ON by default;
# set False to skip the up-front per-file Epoch reads.
WRITE_MANIFEST = True

# Optional, fully self-contained: also dump the binned FFTs to an Excel workbook
# (one sheet per probe, columns = window timestamps, rows = log-bin index) plus a
# regenerated bin README. OFF by default — set True to enable. Requires openpyxl.
EXPORT_EXCEL = True
EXPORT_DIR   = SAVE_DIR    # where the export + bin README are written

# Export format. Each display time-cell is a COLUMN, so a long, fine-time window
# blows past Excel's ~16k-column cap (a week is millions of columns -> crash). CSV
# uses the SAME layout (bins as rows, timestamps as columns) but has no column cap,
# so it's the long-window fallback. One file per probe.
#   'auto'  -> Excel for short windows, CSV once the window exceeds CSV_OVER_MIN
#   'excel' -> always Excel (explicit override)
#   'csv'   -> always CSV (explicit override)
EXPORT_FORMAT = 'auto'
CSV_OVER_MIN  = 30         # in 'auto', windows longer than this (minutes) -> CSV

# Export layout, applied to BOTH the Excel sheets and the CSVs:
#   'wide' -> bin index as rows, timestamps as columns (the original sheet shape)
#   'tall' -> timestamps as rows, bin_0..bin_N as columns (pandas' fast direction)
# Wide chokes when a long window has millions of time COLUMNS (to_csv ~7 min/week);
# tall writes the same data in ~1/3 the time. DROP_EMPTY omits time slots with no
# data in any channel — for burst data most of a multi-day window is empty, so this
# is the dominant speed/size win (a week's CSV ~25x faster) and makes even wide fast.
# The time axis then carries only real samples; gaps show as missing timestamps.
EXPORT_LAYOUT = 'wide'
DROP_EMPTY    = True

# Optional electron cyclotron frequency (f_ce) from FGM background |B|. Three
# independent gates (all OFF by default):
#   CALC_FCE   — load FGM, compute f_ce(t). Required by the two below.
#   PLOT_FCE   — overlay the f_ce curve on each spectrogram (needs CALC_FCE).
#   EXPORT_FCE — add f_ce as MMS<p>_fce sheet(s) to the FFT workbook (needs
#                CALC_FCE and EXPORT_EXCEL; no longer a separate file).
# f_ce comes from FGM (DC field); the spectrogram is SCM (AC) — different cadence,
# hence its own sheet. See fce.py.
CALC_FCE   = True
PLOT_FCE   = True
EXPORT_FCE = True

# global bins list consumed by channelize(), exactly as in ffttest
bins = []


# ---------------------------------------------------------------- log-binning (copied from ffttest)
def make_log_bins(freqs, period):
    h = len(freqs) - 1
    df = 1 / period

    b = list()
    while freqs[h] >= df and h > 0:
        b.append(int(freqs[h]))
        h = int(h // BIN_FACTOR)

    b.append(int(freqs[0]))
    b.reverse()
    return b


def channelize(x):
    if x == bins[0]:
        return 0
    else:
        for i in range(1, len(bins)):
            if bins[i - 1] < x <= bins[i]:
                return i

    if x == bins[-1]:
        return len(bins)

    print(f'wtf: {x}')
    return None


# ---------------------------------------------------------------- display grid sizing
def effective_display_dt(st, ts):
    """Time-cell width actually used for the grid and plot.

    DISPLAY_DT (== STEP) is the finest column — one FFT result per column. For long
    spans that count blows up (a week at 0.1 s = ~6M columns), exhausting RAM and
    swamping the figure's pixel width. Cap the column count at MAX_DISPLAY_COLS by
    widening the column; the binning averages every FFT that lands in a column, so a
    wider column just folds more DISPLAY_DT cells together. Short spans are unaffected
    (the cap never binds, so the returned dt stays DISPLAY_DT). Only the display time
    axis changes here — the FFT window length (2*PAD) is never involved."""
    span_s = (ts - st).total_seconds()
    if span_s <= 0:
        return DISPLAY_DT
    return max(DISPLAY_DT, span_s / MAX_DISPLAY_COLS)


def downsample_matrix(mtx, dt_target, vmin):
    """Aggregate a full-resolution (t_bins, mean, p_cnt) grid into wider time cells
    (dt_target) for plotting, WITHOUT recomputing any FFTs.

    The full-res grid is what the CSV/Excel export keeps (the data of record); only the
    plot uses this coarser copy, because a week at 0.1 s is ~6M columns — far past the
    figure's pixels and enough that matplotlib's per-cell color array at savefig OOMs.
    Re-sums power and counts over the DISPLAY_DT columns that fall in each wider column
    — the same averaging the binning already does — so a plotted column is the true mean
    power of every FFT inside it. Summed one channel at a time so the only temporary is
    one column (~48 MB for a week), never a second full-grid copy. dt_target is a
    display-axis width only; the FFT window length (2*PAD) is not touched here.

    Returns the input unchanged when no coarsening is needed (short spans, where
    dt_target == DISPLAY_DT) or for a missing probe (None)."""
    if mtx is None:
        return None
    t_bins, mean, p_cnt = mtx
    n_full, n_chan = mean.shape
    dt0 = t_bins[1] - t_bins[0]
    if dt_target <= dt0 * (1 + 1e-9):
        return mtx

    # Each full column's CENTER decides which wide column it lands in (monotonic, no
    # gaps). The +0.5 is half a column (center vs left edge), not a duration.
    centers = (np.arange(n_full) + 0.5) * dt0
    jx = np.floor(centers / dt_target + 1e-9).astype(int)
    n_cap = int(jx[-1]) + 1

    new_sum = np.zeros((n_cap, n_chan))
    new_cnt = np.zeros((n_cap, n_chan))
    for c in range(n_chan):
        col_cnt = p_cnt[:, c].astype(float)
        col_sum = mean[:, c] * col_cnt          # empty cells: cnt==0 -> 0 contribution
        new_sum[:, c] = np.bincount(jx, weights=col_sum, minlength=n_cap)
        new_cnt[:, c] = np.bincount(jx, weights=col_cnt, minlength=n_cap)

    new_mean = np.full((n_cap, n_chan), vmin)
    nz = new_cnt > 0
    new_mean[nz] = new_sum[nz] / new_cnt[nz]
    t_bins_new = np.arange(n_cap + 1) * dt_target
    return t_bins_new, new_mean, new_cnt.astype(np.int64)


# ---------------------------------------------------------------- burst selection
def ensure_range_pickles(probe, key, start, end, download=True):
    """Sorted [(time, cdf_path)] of the probe's local burst CDFs covering [start, end).

    Named for what it used to do, convert each CDF to a data pickle first. The
    CDFs are read directly now: this checks them (ffttest.check_cdfs deletes and
    logs an unreadable one, as conversion did) and returns the verified list
    (ffttest.find_local_cdfs).

    download=False: skip find_cdf_files' own network check entirely -- for
    callers (see find_whistler_spike) that run after a dedicated prewarm/
    download pass has already guaranteed the data is local, so this never
    triggers a surprise real SDC network call of its own."""
    inst, mode = field_burst('B' if key[0] == 'B' else 'E')[:2]   # B: scb or schb
    # Validated as discovered (ffttest.cdf_check), with pickle_root so a failure
    # counts toward the 5 strikes -- without it the database refused the write.
    check = cdf_check(key, probe, pickle_root=pickle_root, corrupted_log=corrupted_log)
    find_cdf_files(data_dir, inst, mode, start, end, probe=probe, download=download, check=check)
    return find_local_cdfs(data_dir, inst, mode, start, end, probe=probe, pickle_root=pickle_root,
                           check=check)


def ensure_day_pickles(probe, key, day, download=True):
    """Sorted [(time, cdf_path)] of the probe's local burst CDFs for `day` -- see ensure_range_pickles."""
    return ensure_range_pickles(probe, key, day, day + timedelta(days=1), download=download)


def segment_rms(pkl_path, key):
    """RMS of one burst file's mean-removed |field|. pkl_path is a burst CDF now
    (a data pickle still works); the parameter keeps its old name for callers."""
    archived = _archived_row(pkl_path, key, signal=True)
    if archived is not None:
        # The archive keeps |field| itself, its squares added z, y, x (the FFT's order): at most 1 ulp per sample
        # from the x, y, z sum below, ~1e-7 relative in the RMS -- nothing an argmax over files can see.
        sig = archived.signal
        if sig.size == 0:
            return 0.0
    else:
        df = read_burst_data(pkl_path, key)
        if df.empty:
            return 0.0
        f = key[0]
        sig = np.sqrt(df[f + 'x'].values ** 2 + df[f + 'y'].values ** 2 + df[f + 'z'].values ** 2)
    sig = sig - sig.mean()
    return float(np.sqrt(np.mean(sig ** 2)))


def find_biggest_burst(probe, key, day, spec):
    """Return (start_dt, end_dt, rms) for the highest-RMS segment of `day` on `probe`."""
    day_pkls = ensure_day_pickles(probe, key, day, download=False)   # acquired up front in main()
    if not day_pkls:
        raise FileNotFoundError(f'No {spec["inst"]} burst files for MMS{probe} on {day:%Y-%m-%d}.')

    scored = [(segment_rms(path, key), ftime, path) for ftime, path in day_pkls]
    rms, ftime, path = max(scored, key=lambda r: r[0])

    t, _sig = read_burst_signal(path, key)            # the archive's row when it holds the file
    st = pd.Timestamp(int(t.min())).to_pydatetime()
    ts = pd.Timestamp(int(t.max())).to_pydatetime()
    print(f'Biggest burst (MMS{probe}): {st:%H:%M:%S}-{ts:%H:%M:%S}  '
          f'RMS|{key[0]}|={rms:.3g} {spec["rms_unit"]}')
    return st, ts, rms


def find_whistler_spike(probe, key, start, end, spec, band=(150, 500), pad=0.25, step=0.05,
                        window_s=10.0, lpf_lim=1000, download=True):
    """Slide a `window_s`-second window across every burst file in [start, end)
    and return the (start, end) of the window with the highest total FFT power
    in `band` Hz — where whistler-mode tones typically show up (~150-500 Hz) —
    the argmax over every candidate window in the requested range.

    [start, end) can be anything from a few seconds to a week+ — a few seconds
    just clips down to (at most) one file's worth of candidate windows; a long
    span walks every burst file it covers, same as ensure_range_pickles.

    Unlike find_biggest_burst (broadband RMS of the raw time-domain signal),
    this scores actual FFT power restricted to the target band, so the picked
    window is centered on where the spectrogram actually lights up in that
    band, not just wherever the field is noisiest overall.

    FFTs are computed (and cached, same on-disk layout as gen_fft) per file at
    the given pad/step/lpf_lim; make sure lpf_lim covers `band`'s upper edge.

    download=False: forwarded to ensure_range_pickles -- set by callers (see
    plot_biggest_spikes.py) that already ran their own dedicated download
    pass, so this never triggers an extra, uncontrolled real network call.

    Returns (start_dt, end_dt, band_power) for the winning window.
    """
    inst, mode, fs = field_burst('B' if key[0] == 'B' else 'E')   # B: scb or schb; E never follows it
    n_expected = int(fs * pad * 2)
    Df = 1 / (pad * 2)

    range_pkls = ensure_range_pickles(probe, key, start, end, download=download)
    if not range_pkls:
        raise FileNotFoundError(
            f'No {spec["inst"]} burst files for MMS{probe} in {start:%Y-%m-%d %H:%M:%S}'
            f'-{end:%Y-%m-%d %H:%M:%S}.'
        )

    start_ns = np.datetime64(start, 'ns')
    end_ns   = np.datetime64(end, 'ns')

    best = None  # (band_power_sum, start_dt, end_dt)
    for i, (file_time, curr_path) in enumerate(range_pkls):
        next_path      = range_pkls[i + 1][1] if i + 1 < len(range_pkls) else None
        next_file_time = range_pkls[i + 1][0] if i + 1 < len(range_pkls) else None

        fft_pkl = _fft_pickle_path(pickle_root, inst, mode, pad, file_time, probe=probe, step=step)
        fft_pkl = _compute_and_save_fft(
            (curr_path, next_path, file_time, next_file_time, fft_pkl),
            key, inst, mode, fs, pad, n_expected, Df, step,
        )
        if not fft_pkl:
            continue

        fdf = pd.read_pickle(fft_pkl)
        # Clip to the requested range so a tight (e.g. few-second) request only
        # searches its own vicinity, not the rest of whatever file it anchors to.
        fdf = fdf[(fdf['time'] >= start_ns) & (fdf['time'] <= end_ns)]
        band_df = fdf[(fdf['frequency'] >= band[0]) & (fdf['frequency'] <= band[1])]
        if band_df.empty:
            continue

        # One value per FFT time-step: total power inside the whistler band.
        power_by_t = band_df.groupby('time')['power'].sum().sort_index()
        times = power_by_t.index.values
        power = power_by_t.values

        win_n = max(1, int(round(window_s / step)))
        win_n = min(win_n, len(power))

        # Rolling sum over win_n consecutive FFT steps via cumulative sum (O(n)).
        cumsum  = np.concatenate(([0.0], np.cumsum(power)))
        rolling = cumsum[win_n:] - cumsum[:-win_n]
        idx     = int(np.argmax(rolling))

        win_start = pd.Timestamp(times[idx]).to_pydatetime()
        win_end   = pd.Timestamp(times[min(idx + win_n - 1, len(times) - 1)]).to_pydatetime()
        win_power = float(rolling[idx])

        if best is None or win_power > best[0]:
            best = (win_power, win_start, win_end)

    if best is None:
        raise FileNotFoundError(
            f'No FFT power found in the {band[0]}-{band[1]} Hz band for MMS{probe} in '
            f'{start:%Y-%m-%d %H:%M:%S}-{end:%Y-%m-%d %H:%M:%S}.'
        )

    band_power, st, ts = best
    print(f'Whistler-band spike (MMS{probe}): {st:%Y-%m-%d %H:%M:%S}-{ts:%Y-%m-%d %H:%M:%S}  '
          f'{band[0]}-{band[1]} Hz power sum={band_power:.3g}')
    return st, ts, band_power


# ---------------------------------------------------------------- per-probe spectrogram
def stream_probe_matrix(probe, key, st, ts, n_chan, vmin):
    """Stream a probe's FFTs straight into the (time x channel) grid the plot draws.

    gen_fft yields one DataFrame per raw file; we bin each into a shared accumulator
    and drop it, so peak memory is ONE file's FFT rows plus the grid — flat in the
    length of [st, ts] (a week bins the same as a second). Replaces the old
    load_probe_fft + probe_matrix pair, which pulled the whole interval into a list,
    concatenated it (a second full copy), then copied it again per call.

    Returns (t_bins, mean, p_cnt) or None when the probe has no data for the window:
      t_bins  – (n_time+1,) display-column edges in seconds since `st`
      mean    – (n_time, n_chan) mean power per cell; empty cells = VMIN
      p_cnt   – (n_time, n_chan) FFT-point count per cell (0 => no data there)
    Both the plot (draw_probe) and the optional Excel export consume this one grid,
    so the two can never diverge.

    The time grid is anchored at `st` and sized from the window length, NOT from the
    data's own min/max as before: streaming can't know the data extent up front, and
    `st` is exactly what the x-axis label ('seconds since st') already claims. Gaps
    inside the window — including any lead/trail where data doesn't reach the edges —
    draw as the VMIN background, same as an interior gap did before.
    """
    span_s = (ts - st).total_seconds()
    dt = DISPLAY_DT   # full-resolution grid (one column per FFT step) — feeds the CSV; the
                      # plot downsamples a copy separately (see downsample_matrix)
    n_time = int(np.floor(span_s / dt + 1e-9)) + 1
    if n_time < 1:
        return None

    st_ns = np.datetime64(st, 'ns').astype(np.int64)
    p_sum = np.zeros((n_time, n_chan), dtype=np.float64)
    p_cnt = np.zeros((n_time, n_chan), dtype=np.int64)
    saw_data = False

    try:
        chunks = gen_fft(data_dir, pickle_root, key, PAD, st, ts, LPF_LIM,
                         corrupted_log=corrupted_log, step=STEP, probe=probe,
                         write_manifest=WRITE_MANIFEST, manifest_dir=data_paths.INTERVAL_MANIFEST,
                         download=False)       # acquired up front in main()
    except FileNotFoundError:
        return None

    for fdf in chunks:
        if fdf.empty:
            continue
        # DC (0 Hz) carries no real power; fold it into the lowest channel. Mutate the
        # chunk in place — gen_fft just built it for us and nobody else holds it, so the
        # old defensive .copy() of the whole interval was pure waste.
        fdf.loc[fdf['frequency'] == 0, 'frequency'] = 1

        channel_map = {f: channelize(f) for f in fdf['frequency'].unique()}
        iy = fdf['frequency'].map(channel_map).values.astype(int)

        # Direct floor-division into display columns (NOT np.digitize against arange
        # edges): arange edges carry float error (0.1*3 -> 0.30000000000000004) that
        # files a 0.2999999... center one column early; the 1e-9 nudge absorbs it.
        t_vals = (fdf['time'].values.astype('datetime64[ns]').astype(np.int64) - st_ns) / 1e9
        ix = np.floor(t_vals / dt + 1e-9).astype(int)

        valid = (ix >= 0) & (ix < n_time) & (iy >= 0) & (iy < n_chan)
        ixv, iyv = ix[valid], iy[valid]
        if ixv.size == 0:
            continue
        pw = fdf['power'].values[valid]

        # Bincount only over this file's column span, then add into the grid slice, so
        # the temporary never grows to the full interval — only the grid does.
        lo, hi = int(ixv.min()), int(ixv.max())
        w = hi - lo + 1
        flat = (ixv - lo) * n_chan + iyv
        p_sum[lo:hi + 1] += np.bincount(flat, weights=pw,
                                        minlength=w * n_chan).reshape(w, n_chan)
        p_cnt[lo:hi + 1] += np.bincount(flat,
                                        minlength=w * n_chan).reshape(w, n_chan)
        saw_data = True

    if not saw_data:
        return None

    mean = np.full((n_time, n_chan), vmin)
    nz = p_cnt > 0
    mean[nz] = p_sum[nz] / p_cnt[nz]
    t_bins = np.arange(n_time + 1) * dt
    return t_bins, mean, p_cnt


def time_axis_unit(span_s):
    """Pick a readable x-axis time unit from the window length.

    Returns (divisor_seconds, label). t_bins are in seconds; dividing by the divisor
    keeps the axis legible across scales — a 15 s burst reads in 's', a week in 'days'
    instead of '604800 s'."""
    if span_s < 180:          # < 3 min
        return 1.0, 's'
    if span_s < 3 * 3600:     # < 3 h
        return 60.0, 'min'
    if span_s < 3 * 86400:    # < 3 days
        return 3600.0, 'hours'
    return 86400.0, 'days'    # >= 3 days


def window_stamp(st, ts):
    """Filename-safe start–end stamp. Includes the end DATE when the window spans more
    than one calendar day, so multi-day outputs (e.g. a week of FFTs / f_ce) are
    unambiguous; same-day windows keep the compact start-date + end-time form.
        same day : 20210223_110733-110748
        multi-day: 20210223_110733-20210302_110733
    """
    start = f'{st:%Y%m%d_%H%M%S}'
    end   = f'{ts:%Y%m%d_%H%M%S}' if st.date() != ts.date() else f'{ts:%H%M%S}'
    return f'{start}-{end}'


def draw_probe(ax, probe, mtx, st, norm, n_chan, x_div, inst, vmin):
    """Render one probe's discrete-cell spectrogram into ax from its precomputed
    (t_bins, mean, p_cnt) grid. x_div rescales the (seconds) time axis into the unit
    chosen by time_axis_unit. `inst`/`vmin` are the field's instrument label and
    LogNorm floor (empty-cell color). Returns the mappable (or None)."""
    if mtx is None:
        ax.set_facecolor('0.9')
        ax.text(0.5, 0.5, f'MMS{probe}: no {inst} burst data for {DAY:%Y-%m-%d}',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        return None

    t_bins, mean, _ = mtx

    ax.set_facecolor(mpl.cm.jet(norm(vmin)))
    ye = np.arange(n_chan + 1)
    # rasterized: draw the cells as a single image in the PDF instead of one vector
    # quad each, so the file stays small (and matplotlib never builds a per-quad path).
    return ax.pcolormesh(t_bins / x_div, ye, mean.T, norm=norm, cmap='jet',
                         rasterized=True)


# ---------------------------------------------------------------- main
def acquire_inputs():
    """The acquisition phase (acquisition.acquire), before anything reads a CDF:
    every file this run can open -- the fixed window, or the whole DAY the biggest
    burst is picked from, for each field and probe, plus FGM for f_ce -- is
    downloaded, validated and, if corrupt, fetched again. Everything after reads
    with download=False and never validates (Graham, 2026-09-27)."""
    span = FIXED_WINDOW if FIXED_WINDOW is not None else (DAY, DAY + timedelta(days=1))
    for letter in list(FIELDS) + (['F'] if CALC_FCE else []):
        acquire(letter, PROBES, *span, data_dir=data_dir, pickle_root=pickle_root)


def main():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, FixedLocator

    global bins

    # Window selection keys off the FIRST field (segment timestamps are identical
    # across instruments at the SCB rate, so either field picks the same burst).
    ref_spec = field_spec(FIELDS[0])
    acquire_inputs()
    if FIXED_WINDOW is not None:
        st, ts = FIXED_WINDOW
        print(f'Fixed window: {st:%Y-%m-%d %H:%M:%S} – {ts:%Y-%m-%d %H:%M:%S}')
    else:
        st, ts, _ = find_biggest_burst(REF_PROBE, ref_spec['sig_key'], DAY, ref_spec)

    # Frequency channels are deterministic from the config and FIELD-INDEPENDENT (both
    # fields run the identical SCB-rate pipeline): the FFT only ever emits multiples of
    # Df (= 1/(2*PAD)) from 0 up to LPF_LIM, so the log-bin edges are built once up
    # front WITHOUT materializing any FFT data. That is what lets each probe be
    # streamed straight into its binned grid instead of concatenating the whole
    # interval in RAM first just to learn which frequencies appear.
    Df         = 1.0 / (2 * PAD)
    grid_freqs = (np.round(np.arange(0, LPF_LIM / Df + 1)) * Df).astype(int)
    grid_freqs = grid_freqs[(grid_freqs >= 0) & (grid_freqs <= LPF_LIM)]
    grid_freqs[grid_freqs == 0] = 1   # DC folds into the lowest channel (no power at 0 Hz)
    bins = make_log_bins(np.unique(grid_freqs), PAD * 2)
    n_chan = len(bins)
    print(f'octave bin edges (Hz): {bins}')

    # Optional f_ce from FGM background |B|, streamed per FGM file. f_ce is
    # field-INDEPENDENT (it comes from the FGM DC field, not the plotted spectrogram),
    # so it is collected ONCE here and reused for every field's overlay and export.
    fce_data = {}
    if CALC_FCE:
        from fce import collect_fce, overlay_fce
        for p in PROBES:
            times_ns, fce_hz = collect_fce(data_dir, pickle_root, st, ts,
                                           corrupted_log=corrupted_log, probe=p, download=False)
            fce_data[p] = (times_ns, fce_hz)
            print(f'f_ce (MMS{p}): {fce_hz.size} FGM samples'
                  + (f', {fce_hz.min():.0f}-{fce_hz.max():.0f} Hz' if fce_hz.size else ''))
            # Coverage diagnostic: if the f_ce line stops short of the window end,
            # this shows whether FGM data actually reaches ts. lead/tail gaps are
            # the seconds of [st, ts] with NO FGM sample at each edge.
            if times_ns.size:
                first = pd.Timestamp(times_ns.min())
                last  = pd.Timestamp(times_ns.max())
                lead  = (first - pd.Timestamp(st)).total_seconds()
                tail  = (pd.Timestamp(ts) - last).total_seconds()
                print(f'  FGM coverage: {first:%H:%M:%S.%f}–{last:%H:%M:%S.%f} '
                      f'(lead gap {lead:+.3f}s, tail gap {tail:+.3f}s vs window)')

    # Per-field display sizing (field-independent): the readable time-axis unit and
    # the downsampled column width depend only on the window length, so compute them
    # once and reuse for every column.
    plot_dt       = effective_display_dt(st, ts)
    x_div, x_unit = time_axis_unit((ts - st).total_seconds())

    # Build, export, and downsample each field IN TURN. Doing one field end-to-end
    # before the next keeps peak memory at a SINGLE field's full-resolution grids
    # (a week is ~1.4 GiB each): the big grids are released the moment their small
    # plot copy is made — exactly the free-after-downsample discipline of the old
    # single-field path. Only the small downsampled plot grids (one set per field)
    # and the per-field norms survive into the drawing loop below.
    plot_mtx = {}   # field -> {probe -> downsampled grid (or None)}
    norms    = {}   # field -> LogNorm
    for field in FIELDS:
        spec = field_spec(field)
        print(f'--- {field}-field ({spec["inst"]}) ---')

        # Stream each probe's FFTs into its binned (time x channel) grid. Peak memory
        # is one raw FFT file plus the grid, independent of how long [st, ts] is.
        pm = {p: stream_probe_matrix(p, spec['sig_key'], st, ts, n_chan, spec['vmin'])
              for p in PROBES}
        field_empty = all(m is None for m in pm.values())
        if field_empty:
            print(f'No {spec["inst"]} FFT data for any probe — {field} column drawn empty.')

        # SCM (the FFT) and FGM (f_ce) are independent sources. Only when BOTH are
        # absent for a probe is its required data genuinely missing — flag that
        # explicitly (the SCM-only gap is already labeled by draw_probe).
        if CALC_FCE:
            for p in PROBES:
                scm_missing = pm[p] is None
                fgm_missing = not (p in fce_data and fce_data[p][0].size > 0)
                if scm_missing and fgm_missing:
                    print(f'MMS{p}: required data MISSING — no {spec["inst"]} FFT and '
                          f'no FGM f_ce for {st:%H:%M:%S}-{ts:%H:%M:%S}.')

        # Optional export + README for THIS field, stamped with the field letter
        # (fft_B_…, fft_E_…) so the two never collide. f_ce (when CALC_FCE +
        # EXPORT_FCE) rides along in each field's workbook/CSV, for probes that have
        # both an SCM FFT and FGM samples. Skipped when the field is entirely empty.
        if EXPORT_EXCEL and not field_empty:
            from fft_excel_export import (export_ffts_to_excel, export_ffts_to_csv,
                                          write_bins_readme)
            meta = {'field': field, 'inst': spec['inst'], 'units': spec['units'],
                    'day': DAY, 'start': st, 'end': ts, 'pad': PAD,
                    'df': 1.0 / (2 * PAD), 'lpf_lim': LPF_LIM, 'num_bins': NUM_BINS,
                    'display_dt': DISPLAY_DT}
            stamp        = f'{field}_{window_stamp(st, ts)}'
            readme_path  = os.path.join(EXPORT_DIR, f'fft_{stamp}_bins_README.md')
            os.makedirs(EXPORT_DIR, exist_ok=True)
            write_bins_readme(readme_path, bins, n_chan, meta)
            print(f'Wrote bin README -> {readme_path}')

            window_min = (ts - st).total_seconds() / 60.0
            use_csv = EXPORT_FORMAT == 'csv' or (
                EXPORT_FORMAT == 'auto' and window_min > CSV_OVER_MIN)
            fce_probes = [p for p in PROBES if pm[p] is not None]

            if use_csv:
                why = 'forced' if EXPORT_FORMAT == 'csv' else \
                      f'{window_min:.1f} min > {CSV_OVER_MIN} min'
                paths = export_ffts_to_csv(EXPORT_DIR, stamp, pm, st, n_chan, PROBES,
                                           layout=EXPORT_LAYOUT, drop_empty=DROP_EMPTY)
                print(f'Exported CSV ({why}) -> {len(paths)} file(s) in {EXPORT_DIR}')
                if CALC_FCE and EXPORT_FCE:
                    from fce import export_fce_to_csv
                    written = export_fce_to_csv(EXPORT_DIR, stamp, fce_data, fce_probes)
                    if written:
                        tags = ', '.join(f'MMS{p}' for p in written)
                        print(f'Exported f_ce CSV ({tags}) -> {EXPORT_DIR}')
                    else:
                        print('No f_ce CSV written (no probe had both SCM FFT and FGM f_ce).')
            else:
                xlsx_path = os.path.join(EXPORT_DIR, f'fft_{stamp}.xlsx')
                export_ffts_to_excel(xlsx_path, pm, st, n_chan, PROBES,
                                      layout=EXPORT_LAYOUT, drop_empty=DROP_EMPTY)
                print(f'Exported Excel -> {xlsx_path}')
                if CALC_FCE and EXPORT_FCE:
                    from fce import append_fce_sheets
                    written = append_fce_sheets(xlsx_path, fce_data, fce_probes)
                    if written:
                        tags = ', '.join(f'MMS{p}' for p in written)
                        print(f'Appended f_ce sheets ({tags}) -> {xlsx_path}')
                    else:
                        print('No f_ce sheets written (no probe had both SCM FFT and FGM f_ce).')
        elif EXPORT_EXCEL and field_empty:
            print(f'No {field} data — export skipped.')
        elif CALC_FCE and EXPORT_FCE:
            print('EXPORT_FCE is set but EXPORT_EXCEL is off; f_ce ships inside the FFT '
                  'export, so nothing was written. Enable EXPORT_EXCEL.')

        # The export above consumed the full-res grids; the plot only needs a copy
        # downsampled to a sane column count. Build the small copies, then drop the
        # big grids (del pm) before the next field so peak memory stays at one field's
        # worth, not all FIELDS at once.
        plot_mtx[field] = {p: downsample_matrix(pm[p], plot_dt, spec['vmin'])
                           for p in PROBES}
        norms[field]    = mpl.colors.LogNorm(vmin=spec['vmin'], vmax=spec['vmax'])
        del pm

    if plot_dt > DISPLAY_DT:
        print(f'Plot downsampled: {DISPLAY_DT:g} s -> {plot_dt:.3g} s cells '
              f'({MAX_DISPLAY_COLS} col cap); CSV/Excel keep full {DISPLAY_DT:g} s.')
    if all(plot_mtx[f][p] is None for f in FIELDS for p in PROBES):
        raise RuntimeError('No FFT data for any field/probe in the window — nothing to plot.')

    # Figure grid: one ROW per probe (MMS1 top -> MMS4 bottom), one COLUMN per field
    # (B left, E right). squeeze=False keeps `axes` a 2-D array even for a single
    # probe or single field, so axes[row][col] indexing is uniform.
    n_rows, n_cols = len(PROBES), len(FIELDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11 * n_cols, 6 * n_rows),
                             sharex=True, squeeze=False, constrained_layout=True)

    for j, field in enumerate(FIELDS):
        spec = field_spec(field)
        norm = norms[field]
        for i, p in enumerate(PROBES):
            ax = axes[i][j]
            m = draw_probe(ax, p, plot_mtx[field][p], st, norm, n_chan, x_div,
                           spec['inst'], spec['vmin'])

            # Overlay f_ce (same field-independent FGM curve on every column). Drawn
            # on the channelized y-axis via fce._frac_channel. Skip when there is no
            # spectrogram to annotate (m is None) or the probe has no FGM samples.
            if PLOT_FCE and m is not None and p in fce_data and fce_data[p][0].size:
                times_ns, fce_hz = fce_data[p]
                overlay_fce(ax, st, bins, times_ns, fce_hz, time_div=x_div)

            # Probe label only on the leftmost column; freq ticks on every column.
            ax.set_ylabel(f'MMS{p}\nFreq (Hz)' if j == 0 else 'Freq (Hz)')
            # Top edge is n_chan: pcolormesh cell edges run 0..n_chan, so the top bin
            # (the 2454-3000 Hz channel) and its 3000 Hz tick at channelize(3000)+1 ==
            # n_chan only show when ylim reaches n_chan.
            ax.set_ylim(channelize(10), n_chan)
            # decade ticks only (10/100/1000); ~40 fine bins make per-bin labels unreadable.
            ax.yaxis.set_major_locator(FixedLocator(
                [channelize(10), channelize(100), channelize(1000), channelize(3000) + 1]))
            ax.set_yticklabels(['10', '100', '1000', '3000'])
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(rf'$|{field}|$ ({spec["inst"]})')

        # Include the date in the anchor for multi-hour/day spans where it is ambiguous.
        anchor = f'{st:%Y-%m-%d %H:%M:%S}' if x_div >= 3600.0 else f'{st:%H:%M:%S}'
        axes[-1][j].set_xlabel(f'Time ({x_unit}) since {anchor} UTC')

        # One colorbar per COLUMN — B and E have different units/ranges, so each field
        # gets its own log-power scale beside its column.
        sm = mpl.cm.ScalarMappable(norm=norm, cmap='jet')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[axes[i][j] for i in range(n_rows)], pad=0.01)
        cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')
        cbar.update_ticks()
        cbar.set_label(rf'$\log_{{10}}$ $|{field}|$ power $\;{spec["units"]}$')

    fields_lbl = ' & '.join(f'$|{f}|$' for f in FIELDS)
    fig.suptitle(
        rf'{fields_lbl} power spectrograms — {DAY:%Y-%m-%d} '
        rf'({st:%H:%M:%S}–{ts:%H:%M:%S} UTC),  {PAD * 2:g}s window / '
        rf'$\Delta f = {int((PAD * 2) ** -1)}$ Hz')

    save_path = os.path.join(SAVE_DIR,
                             f'spectro_{"".join(FIELDS)}_{window_stamp(st, ts)}.pdf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf')
    print(f'Saved {save_path}')
    plt.show()


if __name__ == '__main__':
    mpl.use('Qt5Agg')
    main()
