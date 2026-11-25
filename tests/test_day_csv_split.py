"""
Per-day CSV splitting tests for the 'chunks' export path.

A chunks request spans weeks, so its power CSVs are written one file per
calendar day per probe inside that request's own folder. The load-bearing claim
is CONTINUITY: a burst that crosses midnight is split at the boundary and
contributes to both days' files, with day N+1 starting exactly where day N
stopped -- no row dropped, duplicated, or re-binned across the seam.

Two levels:
  1. split_rows_by_day -- the split itself, against a block straddling midnight.
  2. plot_good_chunks   -- the full streamed path, including the buffer/flush
                           that assembles a day from several chunks, plus the
                           bin README written from config even with no data.

Run (needs numpy/pandas/matplotlib -- the `plasma` conda env):

    python tests/test_day_csv_split.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

from fft_excel_export import split_rows_by_day, export_day_to_csv, day_csv_path
from short_interval import BinSpec, ShortIntervalPlotter


def _blocks(n_time, n_chan, seed=0):
    rng = np.random.default_rng(seed)
    mean_g = rng.uniform(1e-6, 1e-2, size=(n_time, n_chan))
    p_cnt = np.ones((n_time, n_chan), dtype=int)
    return mean_g, p_cnt


def test_split_at_midnight_is_lossless():
    """A block starting 30 s before midnight, 0.5 s cells, running 60 s."""
    start = datetime(2021, 2, 23, 23, 59, 30)
    n_time, n_chan, dt = 120, 8, 0.5
    t_left = np.arange(n_time) * dt
    mean_g, p_cnt = _blocks(n_time, n_chan)

    parts = list(split_rows_by_day(start, t_left, mean_g, p_cnt))
    assert len(parts) == 2, f'expected 2 days, got {len(parts)}'

    (d1, off1, m1, c1), (d2, off2, m2, c2) = parts
    assert d1 == pd.Timestamp('2021-02-23'), d1
    assert d2 == pd.Timestamp('2021-02-24'), d2

    # Every row lands in exactly one day, in order, unmodified.
    assert len(off1) + len(off2) == n_time
    assert np.array_equal(np.vstack([m1, m2]), mean_g), 'power rows altered by split'
    assert np.array_equal(np.vstack([c1, c2]), p_cnt), 'count rows altered by split'

    # 30 s of cells before midnight at 0.5 s each = 60 rows.
    assert len(off1) == 60, len(off1)
    # Day 1's offsets are from ITS midnight (so near 86400); day 2's from its
    # own (so from 0). Reconstructed absolute times must be contiguous across
    # the seam -- day 2 picks up exactly one cell after day 1's last.
    last_abs = d1 + pd.to_timedelta(off1[-1], unit='s')
    first_abs = d2 + pd.to_timedelta(off2[0], unit='s')
    assert abs((first_abs - last_abs).total_seconds() - dt) < 1e-9, (
        f'seam gap {(first_abs - last_abs).total_seconds()} != cell width {dt}')
    print('  ok  split_rows_by_day: lossless, contiguous across midnight')


def test_split_single_day_is_one_part():
    start = datetime(2021, 2, 23, 11, 7, 33)
    n_time, n_chan = 40, 6
    t_left = np.arange(n_time) * 0.25
    mean_g, p_cnt = _blocks(n_time, n_chan)
    parts = list(split_rows_by_day(start, t_left, mean_g, p_cnt))
    assert len(parts) == 1
    assert parts[0][0] == pd.Timestamp('2021-02-23')
    assert np.array_equal(parts[0][2], mean_g)
    print('  ok  split_rows_by_day: single-day block stays one part')


def test_day_csv_files_written_and_continuous():
    """Write both days out and check the CSV timestamps continue across files."""
    start = datetime(2021, 2, 23, 23, 59, 30)
    n_time, n_chan, dt = 120, 8, 0.5
    t_left = np.arange(n_time) * dt
    mean_g, p_cnt = _blocks(n_time, n_chan)

    with tempfile.TemporaryDirectory() as d:
        written = []
        for day, off, mb, cb in split_rows_by_day(start, t_left, mean_g, p_cnt):
            written.append(export_day_to_csv(d, 'B', 1, day, off, mb, cb, n_chan,
                                             layout='tall'))
        assert len(written) == 2
        assert os.path.basename(written[0]) == 'fft_B_20210223_MMS1.csv'
        assert os.path.basename(written[1]) == 'fft_B_20210224_MMS1.csv'

        d1 = pd.read_csv(written[0])
        d2 = pd.read_csv(written[1])
        # Drop the two trailing max_power/min_power summary rows.
        t1 = pd.to_datetime(d1['time_utc'][~d1['time_utc'].isin(['max_power', 'min_power'])])
        t2 = pd.to_datetime(d2['time_utc'][~d2['time_utc'].isin(['max_power', 'min_power'])])
        assert t1.iloc[0].date() == pd.Timestamp('2021-02-23').date()
        assert t2.iloc[0].date() == pd.Timestamp('2021-02-24').date()
        # Day 2 begins one cell after day 1 ends -- the continuity guarantee.
        seam = (t2.iloc[0] - t1.iloc[-1]).total_seconds()
        assert abs(seam - dt) < 0.15, f'seam {seam}s across day files'
        assert len(t1) + len(t2) == n_time
    print('  ok  export_day_to_csv: per-day files continue across midnight')


def _fft_pickles(tmpdir, tag, start, span_s, probes, n_parts=2, seed=0):
    """Write each probe's frame out as FFT pickles, as the producer would.

    Split across n_parts files so the path-based consumer is exercised the way
    a real chunk hits it: several per-file pickles folded into one grid.
    """
    out = {}
    part_s = span_s / n_parts
    for p in probes:
        paths = []
        for i in range(n_parts):
            f = _fft_frame(start + timedelta(seconds=i * part_s), part_s,
                           n_rows=2000, seed=seed * 10 + p * 3 + i)
            path = os.path.join(tmpdir, f'{tag}_p{p}_{i}.pkl')
            f.to_pickle(path)
            paths.append(path)
        out[p] = paths
    return out


def _fft_frame(start, span_s, n_rows=4000, seed=0, freqs=None):
    rng = np.random.default_rng(seed)
    if freqs is None:
        freqs = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
    offs = np.sort(rng.uniform(0.0, span_s, size=n_rows))
    return pd.DataFrame({
        'time': np.datetime64(start, 'ns') + (offs * 1e9).astype('timedelta64[ns]'),
        'frequency': rng.choice(freqs, size=n_rows),
        'power': rng.uniform(1e-6, 1e-2, size=n_rows),
    })


def test_plot_good_chunks_writes_per_day_csvs():
    """Full streamed path: three chunks, the middle one crossing midnight."""
    plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
    probes = [1, 2]

    c1 = datetime(2021, 2, 23, 10, 0, 0)
    c2 = datetime(2021, 2, 23, 23, 59, 40)     # crosses into the 24th
    c3 = datetime(2021, 2, 24, 8, 0, 0)
    spans = [(c1, 40.0), (c2, 40.0), (c3, 40.0)]

    with tempfile.TemporaryDirectory() as d:
        pkl_dir = os.path.join(d, 'pkl')
        os.makedirs(pkl_dir)

        def stream():
            for i, (cs, span) in enumerate(spans):
                paths = _fft_pickles(pkl_dir, f'c{i}', cs, span, probes, seed=i)
                yield cs, cs + timedelta(seconds=span), paths, len(probes)

        csv_dir = os.path.join(d, 'csv', 'B', 'chunks_req')
        plotter.plot_good_chunks(
            stream(), key='B', probes=probes, pad=0.25,
            lpf_lim=256, num_bins=10, save_dir=os.path.join(d, 'img'),
            export_csv_dir=csv_dir, csv_layout='tall',
            bins_readme_dir=csv_dir,
        )
        files = sorted(os.listdir(csv_dir))
        csvs = [f for f in files if f.endswith('.csv')]
        # 2 days x 2 probes = 4 day-files (the midnight chunk feeds both days).
        assert len(csvs) == 4, csvs
        for probe in probes:
            for day in ('20210223', '20210224'):
                assert f'fft_B_{day}_MMS{probe}.csv' in csvs, (day, probe, csvs)

        # Day 1 for probe 1 must contain rows from BOTH chunk 1 (10:00) and the
        # pre-midnight part of chunk 2 -- i.e. the buffer really accumulated
        # across chunks before flushing.
        df = pd.read_csv(os.path.join(csv_dir, 'fft_B_20210223_MMS1.csv'))
        t = pd.to_datetime(df['time_utc'][~df['time_utc'].isin(['max_power', 'min_power'])])
        assert (t.dt.hour == 10).any(), 'chunk 1 rows missing from day 1'
        assert (t.dt.hour == 23).any(), 'pre-midnight chunk 2 rows missing from day 1'
        assert t.is_monotonic_increasing, 'day file rows not in time order'

        # Day 2 for probe 1 holds the post-midnight remainder plus chunk 3.
        df2 = pd.read_csv(os.path.join(csv_dir, 'fft_B_20210224_MMS1.csv'))
        t2 = pd.to_datetime(df2['time_utc'][~df2['time_utc'].isin(['max_power', 'min_power'])])
        assert (t2.dt.hour == 0).any(), 'post-midnight chunk 2 rows missing from day 2'
        assert (t2.dt.hour == 8).any(), 'chunk 3 rows missing from day 2'

        assert 'B_bins_README.md' in files, files
    print('  ok  plot_good_chunks: per-day CSVs across chunks + bins README')


def test_bins_readme_written_without_any_data():
    """The E-field regression: no cell had data, README must still appear."""
    plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
    with tempfile.TemporaryDirectory() as d:
        # Empty stream == a field whose probes had no good data at all.
        plotter.plot_good_chunks(
            iter(()), key='E', probes=[1, 2, 3, 4], pad=0.25,
            lpf_lim=3000, num_bins=25, save_dir=None,
            export_csv_dir=os.path.join(d, 'csv'),
            bins_readme_dir=os.path.join(d, 'csv'),
        )
        assert 'E_bins_README.md' in os.listdir(os.path.join(d, 'csv'))
        body = open(os.path.join(d, 'csv', 'E_bins_README.md')).read()
        assert '| bin_index | frequency range |' in body
        assert 'edp' in body
    print('  ok  bins README written for a field with zero data')


def test_analytic_bins_match_computed_bins():
    """The config-only bins must equal what a real binning pass produces.

    _bin_fft_dataframe derives its edges from the frequencies actually present
    in the frame, while _analytic_log_bins builds them from the Df grid the
    FFT is known to land on. Those agree exactly when the frame covers that
    grid -- which real _compute_file_ffts output does, since it snaps every
    frequency to the same grid. So the frame here is generated FROM the grid
    rather than from a handful of arbitrary tones.
    """
    plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
    pad, lpf_lim, num_bins = 0.25, 256, 10
    start = datetime(2021, 2, 23, 10, 0, 0)

    bins_analytic, n_chan_analytic = plotter._analytic_log_bins(pad, lpf_lim, num_bins)

    Df = 1.0 / (2 * pad)
    grid = np.unique((np.round(np.arange(0, lpf_lim / Df + 1)) * Df).astype(int))
    grid = grid[(grid >= 0) & (grid <= lpf_lim)]
    grid[grid == 0] = 1
    grid = np.unique(grid)

    fdf = _fft_frame(start, 30.0, n_rows=40000, freqs=grid)
    result = plotter._bin_fft_dataframe(
        fdf, 'B', pad, start, start + timedelta(seconds=30),
        BinSpec(lpf_lim=lpf_lim, num_bins=num_bins, display_dt=0.05), 0.05)
    assert result is not None
    _, _, _, bins_computed, n_chan_computed, *_ = result

    assert bins_analytic == bins_computed, (bins_analytic, bins_computed)
    assert n_chan_analytic == n_chan_computed
    print(f'  ok  analytic bins == computed bins ({n_chan_analytic} channels)')


def test_empty_chunks_are_not_drawn():
    """A chunk no probe has data for produces no image and no list entry."""
    plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
    probes = [1, 2]
    good = datetime(2021, 2, 23, 10, 0, 0)
    empty = datetime(2021, 2, 23, 12, 0, 0)

    with tempfile.TemporaryDirectory() as d:
        pkl_dir = os.path.join(d, 'pkl')
        os.makedirs(pkl_dir)
        blank_path = os.path.join(pkl_dir, 'blank.pkl')
        pd.DataFrame({'time': pd.to_datetime([]), 'frequency': [], 'power': []}
                     ).to_pickle(blank_path)

        def stream():
            # Real chunk.
            yield good, good + timedelta(seconds=40), \
                _fft_pickles(pkl_dir, 'good', good, 40.0, probes), 2
            # Probe present but its pickle has no rows -> bins to nothing.
            yield empty, empty + timedelta(seconds=40), \
                {p: [blank_path] for p in probes}, 2
            # No probes at all in the chunk.
            yield empty + timedelta(hours=1), empty + timedelta(hours=1, seconds=40), {}, 0

        img = os.path.join(d, 'img')
        saved = plotter.plot_good_chunks(
            stream(), key='B', probes=probes, pad=0.25,
            lpf_lim=256, num_bins=10, save_dir=img,
        )
        assert len(saved) == 1, f'expected only the one real chunk, got {saved}'
        pngs = []
        for root, _dirs, files in os.walk(img):
            pngs += [f for f in files if f.endswith('.png')]
        assert len(pngs) == 1, pngs
        assert '100000' in pngs[0], pngs
    print('  ok  empty chunks produce no image and no list entry')


def test_empty_intervals_are_not_drawn():
    """plot_from_fft_data drops intervals with no data on any probe."""
    plotter = ShortIntervalPlotter(data_dir='/nonexistent', pickle_root='/nonexistent')
    probes = [1, 2]
    a = datetime(2021, 2, 23, 10, 0, 0)
    b = datetime(2021, 2, 23, 12, 0, 0)
    intervals = [
        (a, a + timedelta(seconds=30), 'stream', False),   # has data
        (b, b + timedelta(seconds=30), 'stream', False),   # none
    ]
    # fft_by_probe covers only interval A's time range, so interval B bins empty.
    fft_by_probe = {p: _fft_frame(a, 30.0, seed=p) for p in probes}

    with tempfile.TemporaryDirectory() as d:
        saved = plotter.plot_from_fft_data(
            fft_by_probe=fft_by_probe, intervals=intervals, key='B', probes=probes,
            pad=0.25, lpf_lim=256, num_bins=10, save_dir=d, separate_files=True,
        )
        saved = [p for p in saved if p is not None]
        assert len(saved) == 1, f'expected 1 drawn interval, got {saved}'
        assert '100000' in os.path.basename(saved[0]), saved
        pngs = [f for f in os.listdir(d) if f.endswith('.png')]
        assert len(pngs) == 1, pngs
    print('  ok  empty intervals produce no figure')


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\n{len(tests)} passed')


if __name__ == '__main__':
    _run()
