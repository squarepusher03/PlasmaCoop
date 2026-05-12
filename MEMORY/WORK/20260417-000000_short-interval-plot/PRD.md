---
task: Add single-thread short-interval plotting logic
slug: 20260417-000000_short-interval-plot
effort: standard
phase: complete
progress: 8/8
mode: interactive
started: 2026-04-17T00:00:00Z
updated: 2026-04-17T00:10:00Z
---

## Context

User wants a separate code path for graphing short time intervals (seconds to single-digit minutes).
The existing `gen_fft()` in `ffttest.py` uses `ProcessPoolExecutor(max_workers=8)` twice — for CDF→pickle
conversion and FFT computation — which has massive overhead for a single-file window. The fix is a
separate `ShortIntervalPlotter` class that calls `ensure_data_pickle()` and `_compute_file_ffts()` directly
in a sequential loop, bypassing all multiprocessing. Complete end-to-end: load → FFT → draw.

Not wanted: changes to existing `gen_fft()`, `controller.py`, or `ffttest.py`.

## Criteria

- [x] ISC-1: New file `short_interval.py` created at project root
- [x] ISC-2: `ShortIntervalPlotter` class defined with `__init__(data_dir, pickle_root)`
- [x] ISC-3: `plot()` method accepts key, pad, start, end, lpf_lim, num_bins, display_dt
- [x] ISC-4: CDF files located via `find_cdf_files()` (no new logic)
- [x] ISC-5: CDF→pickle conversion uses direct `ensure_data_pickle()` call, no ProcessPoolExecutor
- [x] ISC-6: FFT computation uses direct `_compute_file_ffts()` call, no ProcessPoolExecutor
- [x] ISC-7: `_draw()` produces log-binned pcolormesh matching ffttest.py visual style
- [x] ISC-8: No changes made to ffttest.py, controller.py, or other existing files

## Verification

All 8 criteria verified by reading `short_interval.py`:
- File created at `/mnt/c/Users/gramb/Documents/COOP/PlasmaCoop/short_interval.py`
- `ShortIntervalPlotter.__init__` takes `data_dir`, `pickle_root`
- `plot()` signature matches ISC-3
- Uses `find_cdf_files()`, `ensure_data_pickle()` in a for loop, `find_data_pickles()`, `_compute_file_ffts()` in a for loop — no ProcessPoolExecutor anywhere
- `_draw()` uses same pcolormesh + FuncFormatter + FixedLocator pattern as ffttest.py
- `ffttest.py`, `controller.py`, etc. untouched (confirmed via git status context)
