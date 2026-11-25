# PlasmaCoop

My Python work from my Plasma Research Data Analyst co-op. I study whistler-mode chorus waves with data from NASA's Magnetospheric Multiscale (MMS) mission, using FFT processing on the ground. The four spacecraft carry the high-precision instruments this needs, so there's a lot of repeated big-data processing, and much of the work is making it fast and provably correct.

## What it does

The project turns MMS burst-mode CDF files into power spectrograms and CSV exports. It uses the magnetic field from the search-coil magnetometer (SCM, |B|, 8192 or 16384 Hz) and the electric field from the double-probe instrument (EDP, |E|, 8192 Hz).

```
CDAWeb download → validated CDFs → preprocessed SQLite archive → windowed FFTs (in workers) → binned CSVs / spectrogram images
```

- **Acquisition** (`acquisition.py`, `cdaweb.py`): downloads from CDAWeb, 8 files at a time, and validates every file before any computation.
- **Preprocessed archive** (`preproc_archive.py`): one SQLite file holds each CDF's record times, quality bits and float32 |field|, at about a tenth of the CDF size.
- **FFT engine** (`ffttest.py`, `short_interval.py`): sliding Hann-windowed FFTs with seam-aware window rules. They are computed per request, with no persistent cache.
- **Event export** (`event_export.py`, `event_images.py`, `event_window.py`): finds events, then writes per-event CSVs and images from a Qt window.
- **Post-process verification** (`post_verify.py`, `post_verify_method.py`): an independent verifier recomputes every CSV from its own CDF copies and reports any difference.

## Running

```bash
python event_window.py   # the event window: find events, export CSVs and images
python main.py           # short-interval spectrogram plots
python download.py       # download MMS data
python -m pytest tests   # test suite
```

Data paths come from `plasmacoop_config.json` (`cdf_dir`). Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `spacepy`, `pyspedas`, `pytplot`, plus `PyQt5` for the event window and output explorer.

## Built with

This repository's history was restarted on 2026-11-25 from the then-current state of the code. The code is developed with an AI assistant:

| Tool | Version |
|------|---------|
| Claude model | Claude Opus 5.5 (`claude-opus-5-5`) |
| Claude Code | 2.1.332 |
| PAI (Personal AI Infrastructure) | 5.0.0 (Algorithm v6.3.0) |
