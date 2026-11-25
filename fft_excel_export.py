"""
Optional Excel + README export for spike_biggest_burst_fft.py.

This module is completely optional: spike_biggest_burst_fft only imports it when
its EXPORT_EXCEL flag is True, so the plotting path never depends on it (and never
needs openpyxl installed). It consumes the exact (t_bins, mean, p_cnt) grids that
probe_matrix() produces for the plot, so the spreadsheet and the spectrogram are
guaranteed to show the same binned power — there is no second DSP path here.

Layout produced:
  fft_<field>_<window>.xlsx          one workbook, one sheet per probe (MMS1..MMS4)
                                     columns = window timestamps (absolute UTC),
                                     rows    = log-bin (channel) index,
                                     cells   = mean power; blank where no FFT data
  fft_<field>_<window>_bins_README.md   bin-index -> frequency-range map, regenerated
                                         each run because binning depends on NUM_BINS.

Requires: openpyxl (pandas' Excel engine). pip install openpyxl
"""

import os
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.io.formats.excel import ExcelFormatter

# pandas' to_excel bolds and box-borders the header row + index column by default
# (visible on pandas < 3.0, e.g. the Windows 2.3.3 env). We only want fitted column
# widths, not styling, so write headers as plain text. Harmless where the default
# was already removed (pandas 3.0).
ExcelFormatter.header_style = None


def _fmt_ts(dt, with_date=False):
    """Absolute UTC timestamp, light precision (HH:MM:SS.d, deciseconds).
    with_date=True prefixes the calendar date so multi-day windows are
    unambiguous; both the Excel and CSV exporters set it."""
    stem = f'{dt:%Y-%m-%d %H:%M:%S}' if with_date else f'{dt:%H:%M:%S}'
    return f'{stem}.{dt.microsecond // 100000}'


def _probe_dataframe(t_bins, mean, p_cnt, st, n_chan, with_date=False,
                     layout='wide', drop_empty=True):
    """Build the power frame shared by the Excel and CSV exporters, so both stay
    identical.

    layout='wide' (default): rows = bin_index 0..n_chan-1, columns = window
        timestamps + trailing max_power / min_power. Matches the original sheet.
    layout='tall': rows = window timestamps (a time_utc column), columns = bin_0..
        bin_{n_chan-1}, with two trailing max_power / min_power summary rows. This is
        pandas' fast direction — wide chokes when a long window has millions of
        time COLUMNS (to_csv ~7 min/week), tall writes the same data in ~1/3 the time.

    drop_empty=True: omit time slots where no channel recorded a point (p_cnt all 0).
        For burst data most of a multi-day window is empty, so this is the dominant
        speed/size win (a week's CSV: ~25x faster, far smaller) in EITHER layout. The
        time axis then carries only real samples — gaps show as missing timestamps.

    cells = mean power; blank (NaN) where no FFT points landed (p_cnt == 0).
    """
    n_time = mean.shape[0]
    left   = t_bins[:n_time]             # t_bins holds n_time+1 edges; use left edges
    keep = (p_cnt > 0).any(axis=1) if drop_empty else np.ones(n_time, dtype=bool)
    cnt  = p_cnt[keep]
    data = mean[keep].astype(float)      # (n_kept, n_chan): row=time, col=bin
    data[cnt == 0] = np.nan              # blank cells with no measurement
    times = [_fmt_ts(st + timedelta(seconds=float(t)), with_date) for t in left[keep]]

    # Per-bin extremes over the kept window, ignoring blank cells. An all-empty bin is
    # all-NaN -> blank (nanmax/nanmin warn and return NaN on an all-NaN slice).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        bin_max = np.nanmax(data, axis=0) if data.shape[0] else np.full(n_chan, np.nan)
        bin_min = np.nanmin(data, axis=0) if data.shape[0] else np.full(n_chan, np.nan)

    if layout == 'tall':
        df = pd.DataFrame(data, columns=[f'bin_{i}' for i in range(n_chan)])
        df.insert(0, 'time_utc', times)
        # per-bin extremes as two labeled trailing rows (parallels wide's trailing cols)
        summary = pd.DataFrame([['max_power', *bin_max], ['min_power', *bin_min]],
                               columns=df.columns)
        return pd.concat([df, summary], ignore_index=True)

    # wide: transpose to (bin_index x timestamp), append per-bin extreme columns
    df = pd.DataFrame(data.T, index=np.arange(n_chan), columns=times)
    df.index.name = 'bin_index'
    df['max_power'] = bin_max
    df['min_power'] = bin_min
    return df


def _bin_index_ranges(bins, n_chan):
    """Map each grid bin index -> (lo_hz, hi_hz, human label), mirroring channelize().

    channelize() files a frequency f as:
      index 0                  when f == bins[0]            (the lowest edge)
      index i (1..len(bins)-1) when bins[i-1] < f <= bins[i]
    n_chan == len(bins): every channel maps to a real frequency range, so the
    fallback loop below is a no-op unless a caller passes a padded n_chan.
    """
    ranges = {0: (bins[0], bins[0], f'= {bins[0]} Hz')}
    for i in range(1, len(bins)):
        ranges[i] = (bins[i - 1], bins[i], f'({bins[i - 1]}, {bins[i]}] Hz')
    for i in range(len(bins), n_chan):
        ranges[i] = (None, None, 'unused / overflow channel')
    return ranges


def _fit_column_widths(ws):
    """Set each column to the minimum width that holds its widest displayed cell.

    openpyxl has no real autofit, so width is the longest rendered string in the
    column. Scientific-formatted cells render as e.g. '1.85E-02' (the displayed
    form), not the long underlying float, so columns hug the timestamp headers.
    """
    for col in ws.columns:
        widest = 0
        for cell in col:
            if cell.value is None:
                continue
            if isinstance(cell.value, (int, float)) and cell.number_format == '0.00E+00':
                shown = f'{cell.value:.2E}'
            else:
                shown = str(cell.value)
            widest = max(widest, len(shown))
        ws.column_dimensions[col[0].column_letter].width = widest + 1


def export_ffts_to_excel(xlsx_path, probe_matrices, st, n_chan, probe_order,
                         layout='wide', drop_empty=True):
    """Write one workbook with a sheet per probe.

    probe_matrices: {probe: (t_bins, mean, p_cnt)} or {probe: None} when a probe
    had no data for the window. Cells with no FFT points (p_cnt == 0) are left
    blank rather than written as the VMIN plot floor, so the sheet reflects only
    real measurements. layout / drop_empty are passed to _probe_dataframe.
    """
    try:
        writer = pd.ExcelWriter(xlsx_path, engine='openpyxl')
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            'Excel export needs openpyxl — install it with `pip install openpyxl`.'
        ) from exc

    with writer:
        for probe in probe_order:
            sheet = f'MMS{probe}'
            mtx = probe_matrices.get(probe)
            if mtx is None:
                pd.DataFrame({'note': ['no burst data for this window']}).to_excel(
                    writer, sheet_name=sheet, index=False)
                _fit_column_widths(writer.sheets[sheet])
                continue

            t_bins, mean, p_cnt = mtx
            df = _probe_dataframe(t_bins, mean, p_cnt, st, n_chan, with_date=True,
                                  layout=layout, drop_empty=drop_empty)
            # wide carries a meaningful bin_index; tall's index is a throwaway RangeIndex.
            df.to_excel(writer, sheet_name=sheet, index=(layout == 'wide'))

            # Display every power cell in scientific notation with a 2-decimal
            # mantissa (e.g. 1.85E-02); the full-precision value is kept underneath.
            ws = writer.sheets[sheet]
            for cells in ws.iter_rows(min_row=2, min_col=2):
                for cell in cells:
                    cell.number_format = '0.00E+00'
            _fit_column_widths(ws)


def make_batch_dir(csv_dir, stamp):
    """Create (if needed) and return csv_dir/stamp -- the per-batch subfolder
    every CSV writer in this project uses so CSVs from ONE export call land
    together, named by that batch's own start-end stamp. Without this, CSVs
    from different (possibly interleaved, slow) runs mix flat in csv_dir with
    nothing but a filename prefix to tell batches apart."""
    path = os.path.join(csv_dir, stamp)
    os.makedirs(path, exist_ok=True)
    return path


def export_cell_to_csv(batch_csv_dir, key, probe, start, end, time_bins, mean_g, vmi, n_chan,
                       layout='wide'):
    """Write one power CSV for a single (interval, probe) spectrogram cell --
    the CSV-writing half of ShortIntervalPlotter.plot_from_fft_data's
    export_csv_dir feature. batch_csv_dir is already the batch-grouped folder
    (see make_batch_dir); this function only builds the cell's own filename
    and writes it, keeping the actual export mechanics out of the plotter.

    Reuses _probe_dataframe, so the CSV matches export_ffts_to_csv's layout.
    That function expects a (t_bins, mean, p_cnt) triple; the plotter's cells
    only keep mean_g (p_cnt is discarded once mean_g is built), so p_cnt is
    reconstructed here as `mean_g != vmi` -- vmi is the floor value the
    plotter's binning fills EVERY empty cell with, so this is an exact,
    lossless stand-in for "did any FFT row land in this cell", not an
    approximation. Returns the path written.

    layout: 'wide' (default, unchanged from before this param existed) or
    'tall' -- see _probe_dataframe's own docstring for the shape/speed
    tradeoff (wide chokes at ~7 min/week on a long window; tall is ~1/3 the
    time). Exposed here but left defaulting to 'wide' so existing callers'
    CSV shape doesn't change under them -- a caller with a wide 'full'
    column can opt into 'tall' explicitly.
    """
    p_cnt = (mean_g != vmi).astype(int)
    start_stamp = f'{start:%Y%m%dT%H%M%S}'
    end_stamp   = f'{end:%Y%m%dT%H%M%S}' if start.date() != end.date() else f'{end:%H%M%S}'
    fname = f'fft_{key}_{start_stamp}-{end_stamp}_MMS{probe}.csv'
    path = os.path.join(batch_csv_dir, fname)
    df = _probe_dataframe(time_bins, mean_g, p_cnt, start, n_chan, with_date=True, layout=layout)
    df.to_csv(path, float_format='%.6e', index=(layout == 'wide'))
    return path


def split_rows_by_day(start, t_left_s, mean_g, p_cnt):
    """Split one spectrogram block's rows into per-CALENDAR-DAY groups.

    Yields (day_midnight, offsets_from_midnight_s, mean_block, cnt_block), one
    entry per UTC day the block touches, in ascending day order. A block that
    starts on one day and ends on the next yields two entries whose rows,
    concatenated, are exactly the input rows -- nothing is dropped, duplicated,
    or re-binned at the boundary, so day N+1's file picks up precisely where
    day N's left off.

    Offsets are re-expressed relative to that day's own midnight so each day's
    CSV stands alone: _probe_dataframe(st=day_midnight) then reconstructs the
    same absolute UTC timestamps the un-split block would have produced.
    """
    n_time = mean_g.shape[0]
    left = np.asarray(t_left_s[:n_time], dtype=float)
    if n_time == 0:
        return

    abs_times = pd.DatetimeIndex(pd.to_datetime(start) + pd.to_timedelta(left, unit='s'))
    day_keys = abs_times.normalize()
    for day in day_keys.unique().sort_values():
        m = np.asarray(day_keys == day)
        offsets = np.asarray((abs_times[m] - day).total_seconds(), dtype=float)
        yield day, offsets, mean_g[m], p_cnt[m]


def day_csv_path(request_csv_dir, key, day, probe):
    """Canonical per-day CSV path inside one request's own folder."""
    return os.path.join(request_csv_dir, f'fft_{key}_{day:%Y%m%d}_MMS{probe}.csv')


def export_day_to_csv(request_csv_dir, key, probe, day, offsets_s, mean_block,
                      cnt_block, n_chan, layout='wide'):
    """Write ONE calendar day's rows for one probe, as a single complete file.

    Same _probe_dataframe shaping as every other exporter here, so a per-day
    CSV is column-for-column identical to the equivalent slice of a single-file
    export -- only the row range differs. The trailing max_power/min_power
    summary _probe_dataframe adds are therefore that DAY's extremes.

    Deliberately a whole-file write, never an append: the summary rows (and, in
    'wide', the timestamp column headers) are properties of the complete row
    set, so a day has to be assembled before it is written. Callers streaming a
    day in across several blocks must buffer that day and flush it once -- see
    ShortIntervalPlotter.plot_good_chunks, which does exactly that on the day
    rollover the chunk stream's ascending order guarantees.
    """
    os.makedirs(request_csv_dir, exist_ok=True)
    path = day_csv_path(request_csv_dir, key, day, probe)
    df = _probe_dataframe(offsets_s, mean_block, cnt_block, day, n_chan,
                          with_date=True, layout=layout)
    df.to_csv(path, float_format='%.6e', index=(layout == 'wide'))
    return path


def export_ffts_to_csv(csv_dir, stamp, probe_matrices, st, n_chan, probe_order,
                       layout='wide', drop_empty=True):
    """Write one CSV per probe in the SAME layout as the Excel sheet — the
    long-window alternative to the workbook.

    Built from _probe_dataframe, so layout / drop_empty mean the same as there:
    'wide' = bin rows x timestamp columns (the original sheet shape); 'tall' =
    timestamp rows x bin columns (pandas' fast direction). The CSV exists because
    Excel caps at ~16k columns, so a long fine-time window (a week of 0.1 s cells =
    millions of columns) can't be a workbook; a CSV has no such cap. drop_empty omits
    timestamps with no data, which is what keeps even the wide CSV fast at week scale.
    Power is written in scientific notation (%.6e).

    All probes from this call land in one make_batch_dir(csv_dir, stamp)
    subfolder, so CSVs from different runs don't mix flat in csv_dir. Returns
    the paths written.
    """
    batch_dir = make_batch_dir(csv_dir, stamp)
    paths = []
    for probe in probe_order:
        path = os.path.join(batch_dir, f'fft_{stamp}_MMS{probe}.csv')
        mtx = probe_matrices.get(probe)
        if mtx is None:
            pd.DataFrame({'note': ['no burst data for this window']}).to_csv(path, index=False)
            paths.append(path)
            continue

        t_bins, mean, p_cnt = mtx
        df = _probe_dataframe(t_bins, mean, p_cnt, st, n_chan, with_date=True,
                              layout=layout, drop_empty=drop_empty)
        # wide's index is the meaningful bin_index; tall's is a throwaway RangeIndex.
        df.to_csv(path, float_format='%.6e', index=(layout == 'wide'))
        paths.append(path)
    return paths


def write_bins_readme(path, bins, n_chan, meta):
    """Regenerate the bin-index -> frequency-range README for this run's binning.

    meta['start']/meta['end'] may be None -- the per-request CSV folders are
    already named for their span, so the writer there omits it rather than
    repeating it. Everything else in this file is a property of the binning
    (pad/lpf_lim/num_bins), not of the data, which is why the README can be
    written for a request whose probes had no data at all.
    """
    ranges = _bin_index_ranges(bins, n_chan)

    if meta.get('start') is not None and meta.get('end') is not None:
        window_line = (f"- Window: {meta['start']:%Y-%m-%d %H:%M:%S}"
                       f"–{meta['end']:%Y-%m-%d %H:%M:%S} UTC")
    else:
        window_line = '- Window: (see the containing folder name)'

    lines = [
        f"# FFT bin index map — |{meta['field']}| ({meta['inst']})",
        '',
        window_line,
        f"- Window length: {2 * meta['pad']:g} s  ->  Δf = {meta['df']:g} Hz",
        f"- Time cell (column) width: {meta['display_dt']:g} s",
        f"- LPF_LIM: {meta['lpf_lim']} Hz   NUM_BINS: {meta['num_bins']}",
        f"- Channels: {n_chan} (bin indices 0–{n_chan - 1})",
        f"- Bin edges (Hz): {bins}",
        '',
        'Each spreadsheet row index is a log-frequency channel; the table below '
        'gives the frequency range that channel covers. This file is regenerated '
        'every run, so it always matches the current NUM_BINS / LPF_LIM.',
        '',
        '## What the number in each cell is',
        '',
        'The **arithmetic mean of the linear PSD values that fall inside the cell** '
        '— not a median, not a peak, and not a mean of logarithms (averaging in '
        'linear power and displaying on a log scale are different things, and this '
        'is the former).',
        '',
        'The mean pools two axes at once:',
        '',
        f"- every Δf = {meta['df']:g} Hz FFT row whose frequency lands in that "
        'channel. Channels are log-spaced, so a high channel averages many more '
        'rows than a low one — see the table below for how wide each is;',
        f"- every FFT window whose centre lands in that {meta['display_dt']:g} s "
        'time cell.',
        '',
        'Power and counts are summed separately and divided only at the end, so the '
        'result does not depend on how the rows happened to be split across files.',
        '',
        'Two things that are easy to misread:',
        '',
        '- A cell no FFT row landed in is **not** blank. It is filled with the '
        "field's floor value (`vmi` in short_interval._FftGrid, which is also the "
        'bottom of the colour scale), so a real measurement that happens to equal '
        'the floor cannot be told apart from missing data.',
        '- DC (0 Hz) is folded into the 1 Hz channel before binning.',
        '',
        '| bin_index | frequency range |',
        '| --- | --- |',
    ]
    for i in range(n_chan):
        _, _, label = ranges[i]
        lines.append(f'| {i} | {label} |')
    lines.append('')

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
