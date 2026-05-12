# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
python main.py          # run short-interval spectrogram plots
python ffttest.py       # run long-interval FFT generation (sets plot/save flags at bottom of file)
python download.py      # download MMS data via pyspedas and convert CDFs to pickles
```

No build step, no test suite. All paths are hardcoded to `E:/PlasmaCoop/` (Windows data drive).

## Architecture

This project generates **power spectrograms** from NASA MMS satellite CDF files, analyzing magnetic (SCM/SCB, 8192 Hz) and electric (EDP/DCE) field data around a magnetic reconnection event (2019-08-16).

### Data pipeline

```
CDF files (pydata/) → per-file data pickles (.cache/) → FFT pickles (.cache/…/fft/) → matplotlib plot
```

- `find_cdf_files` / `find_data_pickles` in `ffttest.py` do anchor-based file discovery: find the rightmost file whose timestamp ≤ start, include everything through end.
- `ensure_data_pickle` converts a CDF to a DataFrame pickle on first use; bad CDFs are deleted and logged to `corrupted.txt`.
- `_compute_file_ffts` is the core FFT worker: sliding Hann-windowed rfft, one-sided PSD, frequency snapped to `Df` grid, output as `(time, frequency, power)` DataFrame rows.

### Two execution paths

| Path | File | Use case |
|------|------|----------|
| `ShortIntervalPlotter` | `short_interval.py` | Seconds to ~10 min; single-threaded, all in one call |
| `gen_fft` | `ffttest.py` | Hours to months; parallel CDF conversion (`ProcessPoolExecutor`) then parallel FFT workers |

`main.py` is the entry point — currently uses `ShortIntervalPlotter`.

### In-progress MVC refactor (branch: MMS-REFACTOR)

The files `controller.py`, `drawer.py`, `request.py`, `options.py`, `fftcalculator.py` form an incomplete OOP wrapper around the working `ffttest.py` logic:

- `Request` / `BFieldRequest` / `EFieldRequest` — hold plot config options
- `Controller` — queues `Request` objects, calls calculator and drawer (currently `_calculator` is never assigned — stub)
- `FFTCalculator` — intended to wrap `gen_fft`/`_compute_file_ffts`; mostly stub
- `Drawer` — wraps matplotlib spectrogram drawing; mostly complete but wired to unfinished calculator
- `options.py` — `FrequencyOptions`, `PowerOptions`, `TimeOptions`, `SourceOptions` config classes

The working code is `ffttest.py` + `short_interval.py`. The MVC classes are aspirational structure, not yet functional.

## Key constants

- `SCB_FS = 8192` Hz — SCM burst mode sample rate
- `SCHB_FS = 16384` Hz — SCM high-burst sample rate (unused in current plots)
- FFT window: `pad` seconds on each side → full window `2*pad` s → frequency resolution `Df = 1/(2*pad)` Hz
- Log-bin channelization: `bin_factor = lpf_lim ** (1/num_bins)`, each channel mapped by `channelize()`

## Dependencies

`numpy`, `scipy`, `pandas`, `matplotlib`, `spacepy` (for `pycdf`), `pyspedas`, `pytplot`
