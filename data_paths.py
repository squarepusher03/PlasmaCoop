"""Per-platform data root for the PlasmaCoop E: drive.

The MMS data lives on the E: drive, which Windows addresses as 'E:/PlasmaCoop'
and WSL/Linux mounts at '/mnt/e/PlasmaCoop'. Resolve it once here so every entry
point (main, ffttest, download, burst_times, ...) runs on either platform
without editing paths. This is the single place the rule lives.

DATA_DIR, the folder CDFs are downloaded to and read from, comes from the
settings file (app_config: cdf_dir, Graham's is E:/PlasmaCoop/pydata), named the
way this platform names it.
"""

import os

import app_config

# The settings file's data_root (default E:/PlasmaCoop; was fixed here until 2026-11-25), named the way this
# platform names it: 'E:/...' on Windows ('nt'), '/mnt/e/...' on WSL/Linux.
DATA_ROOT = app_config.data_root()

DATA_DIR          = app_config.cdf_dir()          # the settings file's cdf_dir (was f'{DATA_ROOT}/pydata')
PICKLE_ROOT       = f'{DATA_ROOT}/.serialized'
CORRUPTED_LOG     = f'{DATA_ROOT}/corrupted.txt'
INTERVAL_MANIFEST = f'{DATA_ROOT}/.manifest'
BINNING_DIR       = f'{DATA_ROOT}/.binning'
# Per-run output bundle: each execution gets its own subfolder (named by wall-
# clock run time) holding that run's CSVs (field-separated), interval manifests,
# and bin READMEs together. See plot_biggest_spikes.make_batch_run_dir.
BATCH_DIR         = f'{DATA_ROOT}/batches'
