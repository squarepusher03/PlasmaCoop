"""
Electron cyclotron frequency (f_ce) overlay + export for the SCM spectrograms.

Optional companion to spike_biggest_burst_fft.py: imported only when CALC_FCE is
True, so the plotting path never depends on it.

Physics
-------
f_ce = (e / 2π m_e) · |B|  ≈  27.99 Hz/nT · |B[nT]|

The spectrogram is built from SCM (search-coil, AC-coupled) data, which carries
only the wave *fluctuations* and is mean-subtracted — it has no DC content. The
gyrofrequency depends on the *background* field magnitude |B|, which only the
fluxgate magnetometer (FGM) measures. So f_ce is sourced from FGM burst data, on
FGM's own native time grid (~128 Hz), which differs from the FFT window grid —
hence its own sheet (append_fce_sheets) rather than sharing the FFT columns.

Streaming
---------
stream_fgm_bmag() yields one FGM file's (time, |B|) at a time, mirroring
ffttest.gen_fft, so peak memory is a single FGM file. f_ce is far smaller than
the FFTs, so collect_fce() concatenates the (tiny) result for the overlay/export.

Requires openpyxl only when export_fce_to_excel is used.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from ffttest import (find_cdf_files, find_data_pickles, ensure_data_pickle,
                     check_cdfs, cdf_check, find_local_cdfs, read_burst_data)

# e / (2π m_e):  1.602176634e-19 / (2π · 9.1093837015e-31)
#   = 2.79924902e10 Hz/T  =  27.9924902 Hz per nT of |B|
FCE_HZ_PER_NT = 27.992_490_2

# FGM key 'F' routes ffttest._cdf_to_df (through read_burst_data) to the FGM
# branch, which keeps a single 'Bmag' column. FGM CDFs have no mode subdir, so
# discovery uses mode=''. _FGM_PKL_MODE is the mode the old data-pickle cache
# used ('brst', set inside ensure_data_pickle); nothing reads that cache now.
_FGM_INST = 'fgm'
_FGM_PKL_MODE = 'brst'
_FGM_KEY = 'F'


def fce_from_bmag(bmag_nt):
    """Electron cyclotron frequency (Hz) from |B| in nT. Scalar or array."""
    return FCE_HZ_PER_NT * np.asarray(bmag_nt, dtype=float)


def stream_fgm_bmag(data_dir, pickle_root, st, ts, corrupted_log=None, probe=1, download=True):
    """Yield (times_ns, bmag) per FGM file over [st, ts], one file at a time.

    Read straight from the FGM CDFs (ffttest.read_burst_data), the way the
    SCM/EDP path reads its files -- no per-file pickle. A CDF that cannot be read
    is deleted and logged by ffttest.check_cdfs, as conversion used to do. Each
    yielded chunk is filtered to the [st, ts] window.

    download=False: forwarded to find_cdf_files -- for callers (e.g.
    plot_biggest_spikes.py) that already ran their own dedicated download
    pass and need this to never trigger a surprise network call.
    """
    # Validated as discovered (ffttest.cdf_check). pickle_root goes to the check
    # too: without it the database refuses the write, so a bad FGM file was never
    # counted toward the 5 strikes and a good one was revalidated every process.
    check = cdf_check(_FGM_KEY, probe, pickle_root=pickle_root, corrupted_log=corrupted_log)
    find_cdf_files(data_dir, _FGM_INST, '', st, ts, probe=probe, download=download, check=check)

    st64 = np.datetime64(st, 'ns')
    ts64 = np.datetime64(ts, 'ns')
    for ftime, path in find_local_cdfs(data_dir, _FGM_INST, '', st, ts, probe=probe,
                                       pickle_root=pickle_root, check=check):
        df = read_burst_data(path, _FGM_KEY, probe=probe)
        if df.empty:
            continue
        t = df['time'].values.astype('datetime64[ns]')
        m = (t >= st64) & (t <= ts64)
        if not m.any():
            continue
        yield t[m], df['Bmag'].values[m]


def collect_fce(data_dir, pickle_root, st, ts, corrupted_log=None, probe=1, download=True):
    """Drain stream_fgm_bmag into (times_ns, fce_hz) arrays. Empty -> (empty, empty).

    download=False: forwarded to stream_fgm_bmag -- see its docstring."""
    times, mags = [], []
    for t_chunk, b_chunk in stream_fgm_bmag(data_dir, pickle_root, st, ts,
                                            corrupted_log=corrupted_log, probe=probe,
                                            download=download):
        times.append(t_chunk)
        mags.append(b_chunk)
    if not times:
        empty_t = np.array([], dtype='datetime64[ns]')
        return empty_t, np.array([], dtype=float)
    times_ns = np.concatenate(times)
    fce_hz = fce_from_bmag(np.concatenate(mags))
    return times_ns, fce_hz


def _frac_channel(fce_hz, bins):
    """Map frequencies (Hz) to fractional log-bin channel positions.

    The spectrogram y-axis is the channelize() integer index; channel i has
    upper edge bins[i]. Interpolating in log10-frequency against channel index
    places f_ce smoothly *between* bin centers so the overlay is a continuous
    curve rather than a staircase. Values outside the bin range clamp to the
    nearest edge (np.interp behavior).
    """
    edges = np.asarray(bins, dtype=float)
    log_edges = np.log10(edges)
    chan_idx = np.arange(len(edges), dtype=float)
    return np.interp(np.log10(np.asarray(fce_hz, dtype=float)), log_edges, chan_idx)


def _gap_segments(x, gap_factor):
    """Yield (start, stop) index ranges of contiguous f_ce runs, split at FGM gaps.

    FGM bursts are separate intervals; collect_fce concatenates them, so one plotted
    line would bridge the data-less gap between bursts with a straight segment. Instead
    of inserting NaNs into the full (a week = millions of samples) array — which forces
    a full-size copy just to mark a handful of breaks — this only finds where the gaps
    are and yields the slice bounds. The caller plots each run as its own segment (the
    slices are views, no copy, no NaN materialized), so the pen lifts across gaps while
    peak memory stays the size of the data. A gap is spacing above gap_factor x the
    median cadence (~8 ms at 128 Hz); intra-burst jitter stays well under it."""
    n = x.size
    if n <= 1:
        yield 0, n
        return
    dt = np.diff(x)
    pos = dt[dt > 0]
    if pos.size == 0:
        yield 0, n
        return
    cuts = np.flatnonzero(dt > gap_factor * np.median(pos)) + 1   # run boundaries
    prev = 0
    for c in cuts:
        yield prev, int(c)
        prev = int(c)
    yield prev, n


def overlay_fce(ax, st, bins, times_ns, fce_hz, color='white',
                lw=4.0, ls='-', label=r'$f_{ce}$', gap_factor=8.0, time_div=1.0):
    """Draw f_ce as a curve on a channelized spectrogram Axes.

    x = (seconds since st) / time_div — time_div rescales to the spectrogram's x-axis
    unit (s/min/hours/days) so the curve stays aligned with the pcolormesh. y =
    fractional log-bin channel (matches the pcolormesh y-axis). The line is drawn with
    a dark stroke so it reads on the jet colormap,
    and `label` is placed inline just above the curve's midpoint (no legend).
    Draws onto the given Axes; never creates a figure. Returns the first Line2D.

    The curve is drawn as one stroked segment per contiguous FGM run (see
    _gap_segments), so it breaks across intervals with no FGM data instead of
    interpolating a straight line over them — without ever building a NaN-padded
    full-size copy of the (week = millions of samples) array.
    """
    if times_ns.size == 0:
        return None
    import matplotlib.patheffects as pe

    st_ns = np.datetime64(st, 'ns').astype('int64')
    x = (times_ns.astype('int64') - st_ns) / 1e9 / time_div
    y = _frac_channel(fce_hz, bins)

    stroke = [pe.withStroke(linewidth=lw + 1.8, foreground='black')]
    first = None
    for a, b in _gap_segments(x, gap_factor):
        if b - a < 1:
            continue
        # x[a:b] / y[a:b] are views — no copy, no NaN. matplotlib copies only this
        # run's points into the Line2D, so total memory matches the data, not 2x it.
        (line,) = ax.plot(x[a:b], y[a:b], color=color, lw=lw, ls=ls, zorder=5,
                          path_effects=stroke)
        if first is None:
            first = line

    if label and x.size:
        # Inline label riding just above the curve at its midpoint, with its own
        # dark stroke so it reads on the jet colormap. A small fixed offset keeps
        # it ON the line (annotating it) instead of floating up like a title;
        # clipping to the Axes prevents it escaping the plot.
        mid = x.size // 2
        text_stroke = [pe.withStroke(linewidth=3, foreground='black')]
        ax.annotate(label, xy=(x[mid], y[mid]), xytext=(0, 8),
                    textcoords='offset points', va='bottom', ha='center',
                    color=color, fontsize=13, fontweight='bold', zorder=6,
                    path_effects=text_stroke, clip_on=True, annotation_clip=True)
    return first


# Excel datetime format with milliseconds. FGM is ~128 Hz (~8 ms between
# samples), so the default Excel datetime format — which hides the fraction —
# makes every sample in the same second look identical, i.e. the times appear
# missing. '.000' keeps each sample's time distinct.
_FCE_TIME_FMT = 'yyyy-mm-dd hh:mm:ss.000'


def _write_fce_sheet(writer, sheet_name, times_ns, fce_hz):
    """Write a time_utc + f_ce_Hz sheet into an open ExcelWriter, with the time
    column formatted to millisecond precision and both columns widened to fit."""
    df = pd.DataFrame({
        'time_utc': pd.to_datetime(times_ns),
        'f_ce_Hz':  np.asarray(fce_hz, dtype=float),
    })
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    for cell in ws['A'][1:]:                 # column A = time_utc, skip header
        cell.number_format = _FCE_TIME_FMT
    for cell in ws['B'][1:]:                 # column B = f_ce_Hz, skip header
        cell.number_format = '0.00'          # display at most 2 decimals
    # Width must exceed the rendered string length or Excel shows '######'. The
    # formatted timestamp 'yyyy-mm-dd hh:mm:ss.000' is 23 chars; +1 for margin.
    ws.column_dimensions['A'].width = len(_FCE_TIME_FMT) + 1
    ws.column_dimensions['B'].width = 12
    return ws


def append_fce_sheets(xlsx_path, fce_data, probe_order):
    """Append one MMS<p>_fce sheet per probe to an existing FFT workbook.

    f_ce lives on FGM's native ~128 Hz time grid, not the FFT window grid, so it
    gets its own sheet (time_utc + f_ce_Hz) rather than sharing the FFT columns —
    but in the SAME workbook as the FFTs instead of a separate file.

    fce_data: {probe: (times_ns, fce_hz)}. Probes absent from fce_data OR with no
    samples are skipped (no empty sheet). Returns the list of probes written; if
    that is empty the workbook is left untouched.
    """
    # Match fft_excel_export: strip pandas' default header styling so old pandas
    # (Windows 2.3.3) and pandas 3.0 produce the same plain sheet.
    from pandas.io.formats.excel import ExcelFormatter
    ExcelFormatter.header_style = None

    to_write = [p for p in probe_order
                if p in fce_data and fce_data[p][0].size > 0]
    if not to_write:
        return []

    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a',
                            if_sheet_exists='replace') as writer:
            for probe in to_write:
                times_ns, fce_hz = fce_data[probe]
                _write_fce_sheet(writer, f'MMS{probe}_fce', times_ns, fce_hz)
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            'f_ce Excel export needs openpyxl — install it with `pip install openpyxl`.'
        ) from exc
    return to_write


def export_fce_to_csv(csv_dir, stamp, fce_data, probe_order):
    """Write one tall CSV per probe (time_utc, f_ce_Hz) — the CSV companion to
    append_fce_sheets, used when the FFT export switches to CSV for long windows.

    Same skip rule as append_fce_sheets (probes absent or with no samples are
    skipped). f_ce is rounded to 2 decimals to match the spreadsheet display. No
    openpyxl needed. Returns the list of probes written.

    time_utc is written as 'YYYY-MM-DD HH:MM:SS.mmm' (date then time, millisecond
    precision) — the same precision as the Excel f_ce sheet. Without explicit
    formatting, to_csv prints the raw datetime64[ns], i.e. 9 fractional-second digits
    (e.g. 11:07:33.131269289), which reads like garbage/overflow; FGM is ~128 Hz so
    milliseconds already separate every sample.

    All probes from this call land in the SAME make_batch_dir(csv_dir, stamp)
    subfolder export_ffts_to_csv uses for this stamp, so the FFT CSVs and their
    f_ce companions from one run stay grouped together.
    """
    import os

    from fft_excel_export import make_batch_dir
    batch_dir = make_batch_dir(csv_dir, stamp)
    written = []
    for probe in probe_order:
        if probe not in fce_data or fce_data[probe][0].size == 0:
            continue
        times_ns, fce_hz = fce_data[probe]
        # %f is microseconds (6 digits); drop the last 3 to get milliseconds.
        time_utc = pd.to_datetime(times_ns).strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
        df = pd.DataFrame({
            'time_utc': time_utc,
            'f_ce_Hz':  np.round(np.asarray(fce_hz, dtype=float), 2),
        })
        path = os.path.join(batch_dir, f'fce_{stamp}_MMS{probe}.csv')
        df.to_csv(path, index=False)
        written.append(probe)
    return written
