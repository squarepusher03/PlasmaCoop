---
task: Design low-coupling FFT architecture from ffttest.py
slug: 20260512-190743_low-coupling-fft-architecture
effort: deep
phase: plan
progress: 0/48
mode: interactive
started: 2026-05-12T19:07:43Z
updated: 2026-05-12T19:08:30Z
---

## Context

Graham wants a software architecture redesign of the monolithic `ffttest.py` (only — not `slop/ffttest.py`) into independent, low-coupling components separated by single responsibility. He has new architecture knowledge from school and wants to apply it. He is hitting **performance issues** caused partly by Python/Pandas/NumPy overhead and partly by structural coupling that prevents targeted optimization.

The deliverable is a **design only** at this stage. No implementation yet. The design must be defensible, decomposed atomically, and verifiable before any code is written. The user explicitly asked PAI to use its full capabilities to prevent design drift, hallucination, and cyclical structures.

Working code being decomposed:
- `ffttest.py` (~668 lines) — the monolith
- `short_interval.py` — single-threaded path that imports from `ffttest.py` (consumer view of the API)

Existing in-progress refactor stubs (`badfactor/`, `src/downloading/`) are aspirational and reveal Graham's preferred direction (Request objects, Manager, Options classes, abstract Download requests). The design should integrate with that direction, not replace it.

## Criteria

### Atomic responsibility inventory (what ffttest.py actually does — one ISC per responsibility identified)
- [ ] ISC-1: Responsibility identified — discover CDF files in a date range by glob + regex parsing of filename timestamps
- [ ] ISC-2: Responsibility identified — discover existing data-pickle files in a date range (same anchor logic, different on-disk schema)
- [ ] ISC-3: Responsibility identified — read a CDF file into a typed DataFrame (instrument-specific variable names and Epoch column)
- [ ] ISC-4: Responsibility identified — derive a pickle cache path from CDF filename timestamp and instrument/mode
- [ ] ISC-5: Responsibility identified — convert CDF to pickle on cache miss, with corruption handling and a cross-process append-only log
- [ ] ISC-6: Responsibility identified — derive an FFT-result cache path that includes `pad` (since pad changes Df grid)
- [ ] ISC-7: Responsibility identified — slide a Hann-windowed window across one file's signal at a fixed step
- [ ] ISC-8: Responsibility identified — bridge windows that span a file boundary by lazily loading the next file's head
- [ ] ISC-9: Responsibility identified — decide per-window policy: skip / zero-fill / compute (rules differ for boundary vs interior)
- [ ] ISC-10: Responsibility identified — compute one-sided PSD from a windowed signal: rfft, |·|², normalize by `fs·u·n`, double non-DC
- [ ] ISC-11: Responsibility identified — snap rfft frequency bins onto the integer `Df` grid for cross-window comparability
- [ ] ISC-12: Responsibility identified — pre-size output arrays per file and trim to actual rows (avoid dynamic append)
- [ ] ISC-13: Responsibility identified — orchestrate the pipeline (discover → cache CDFs → cache FFTs → filter to [st,ts] → yield)
- [ ] ISC-14: Responsibility identified — fan-out parallel work across CDF files using `ProcessPoolExecutor`
- [ ] ISC-15: Responsibility identified — log-bin frequencies into channels and channelize a frequency value
- [ ] ISC-16: Responsibility identified — aggregate (time, channel, power) into a 2-D mean grid via `bincount`
- [ ] ISC-17: Responsibility identified — render the grid with `pcolormesh`, LogNorm, colorbar, and tick formatting

### Coupling defects to be eliminated (one ISC per defect — must be addressed in the design)
- [ ] ISC-18: Defect addressed — module-global `bins = []` mutated from `__main__` and read by `channelize`; new design must inject the bins object explicitly
- [ ] ISC-19: Defect addressed — instrument dispatch (`if key[0] == 'B'`) duplicated across `_cdf_to_df`, `ensure_data_pickle`, `gen_fft`; new design must localize to one strategy/registry
- [ ] ISC-20: Defect addressed — pickle path schema written by `ensure_data_pickle` and parsed back by `find_data_pickles` regex; new design must own this in a single path-builder/parser pair
- [ ] ISC-21: Defect addressed — `_compute_file_ffts` co-mingles windowing math, file I/O, boundary loading, and frequency snapping; new design must isolate the pure DSP kernel from I/O
- [ ] ISC-22: Defect addressed — `gen_fft` couples discovery + pool fan-out + cache filtering + yielding; new design must separate orchestration from execution and from filtering
- [ ] ISC-23: Defect addressed — `short_interval.py` re-imports private `_compute_file_ffts` and duplicates `make_log_bins`/`channelize`; new design must expose a public single-pass and a streaming API from the same kernel
- [ ] ISC-24: Defect addressed — `_corrupted_log_lock` is a module-global `threading.Lock` only useful for ThreadPool, but the actual fan-out uses `ProcessPoolExecutor` with a Manager Lock; new design must own a single corruption-log writer with one locking strategy

### Module decomposition (one ISC per proposed module — design must define its interface)
- [ ] ISC-25: Module `cdf_io` defined — pure CDF→DataFrame reader, instrument-agnostic interface, no caching, no globbing
- [ ] ISC-26: Module `cache_layout` defined — single owner of pickle/FFT path schemas; emits and parses paths (no I/O)
- [ ] ISC-27: Module `file_index` defined — discovery (`find_cdf_files`, `find_data_pickles`); pure on filesystem + cache_layout
- [ ] ISC-28: Module `data_cache` defined — CDF→pickle cache writer; uses cdf_io + cache_layout; emits a `CorruptionEvent` record instead of touching a global log
- [ ] ISC-29: Module `corruption_log` defined — single sink for `CorruptionEvent`; chooses its own locking based on executor (multiprocessing Manager Lock or thread Lock)
- [ ] ISC-30: Module `fft_kernel` defined — pure DSP: takes `(signal: np.ndarray, fs, pad)` → `(freqs: np.ndarray, psd: np.ndarray)`; no DataFrames, no I/O
- [ ] ISC-31: Module `window_scheduler` defined — given a time index, file boundaries, and policy, yields `(center_ns, signal_slice)` tuples; no FFT math, no I/O beyond `np.memmap`/`np.load` of pickled arrays
- [ ] ISC-32: Module `freq_grid` defined — owns `Df` grid math and rfft-bin → integer-Hz snapping; consumed by fft_kernel post-process
- [ ] ISC-33: Module `fft_cache` defined — read/write FFT-result arrays per file; uses cache_layout for paths
- [ ] ISC-34: Module `pipeline` defined — orchestrates discovery → cache_ensure → fft_compute → filter; injection points for executor and cache backends
- [ ] ISC-35: Module `channelizer` defined — `LogChannelizer(freqs, lpf, num_bins)` instance object exposing `channels` and vectorized `to_channel(freq_array)` via `np.searchsorted`; no globals
- [ ] ISC-36: Module `aggregator` defined — `bincount`-based (time, channel) → mean grid; no plotting
- [ ] ISC-37: Module `plotter` defined — renders a prepared grid; consumes channelizer for ticks; no compute

### Coupling rules the design must satisfy (one ISC per invariant)
- [ ] ISC-38: Dependency rule defined — `fft_kernel` depends on nothing except NumPy/SciPy; verifiable by absent imports
- [ ] ISC-39: Dependency rule defined — `cache_layout` depends on nothing except `os.path`; verifiable by absent imports
- [ ] ISC-40: Dependency rule defined — `plotter` does not import `fft_kernel`, `data_cache`, or `pipeline`; verifiable by import graph
- [ ] ISC-41: Dependency rule defined — `pipeline` is the only module that imports the multiprocessing executor; all other modules are executor-agnostic
- [ ] ISC-42: Dependency rule defined — no module mutates module-level state at import or at runtime; design lists every state holder as an instance attribute

### Performance design decisions (one ISC per intervention, each independently testable)
- [ ] ISC-43: Performance — replace Python `channelize` + `np.vectorize` with vectorized `np.searchsorted` over a NumPy bins array (≥10× speedup expected on the channelize step)
- [ ] ISC-44: Performance — replace `pd.DataFrame` round-trip for per-file FFT results with three parallel `np.ndarray`s saved as `.npz` (smaller pickles, faster reads)
- [ ] ISC-45: Performance — eliminate per-window `pd.read_pickle(next_path)` re-loads by holding a `(file_time → memmap)` LRU in the window scheduler (size 2)
- [ ] ISC-46: Performance — replace fixed `max_rows = n_steps * len(freq_grid)` over-allocation with growing-in-chunks list of np.ndarrays then a single concatenate at the end
- [ ] ISC-47: Performance — make `signal = signal - np.mean(signal)` and `np.hanning(n)` cache-aware: precompute `(n_expected, hann_window)` once per file since `n` is constant except at edges
- [ ] ISC-48: Performance — confirm or correct EDP `fs` (currently hard-coded to `SCB_FS = 8192` for E-field with `TODO: confirm EDP sample rate`); spec a single source-of-truth `instrument_spec` map

### Anti-criteria
- [ ] ISC-A1: NO new module introduces a cycle in the import graph
- [ ] ISC-A2: NO module-level mutable state (no module-scope `bins = []`, no module-scope locks)
- [ ] ISC-A3: NO implementation code is written this turn — design only
- [ ] ISC-A4: NO existing working file is deleted as part of the design phase (`ffttest.py`, `short_interval.py`, `download.py`, `main.py` remain untouched until Graham approves a migration plan)
- [ ] ISC-A5: NO `slop/ffttest.py` content is folded into the design — Graham explicitly excluded it
- [ ] ISC-A6: NO design choice depends on changing the on-disk pickle format silently — any format change must be explicit and migration-ready
