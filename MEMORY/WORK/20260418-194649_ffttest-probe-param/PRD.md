---
task: Parameterize MMS probe number in ffttest.py
slug: 20260418-194649_ffttest-probe-param
effort: standard
phase: complete
progress: 10/10
mode: interactive
started: 2026-04-18T19:46:49-04:00
updated: 2026-04-18T19:48:00-04:00
---

## Context

Added `probe: int = 1` parameter to probe-relevant functions in ffttest.py so MMS1/2/3/4 stacked plotting becomes possible. Default=1 preserves all existing caller behavior. Tab indentation preserved via Python-literal \t edits.

## Criteria

- [x] ISC-1: find_cdf_files signature gains probe: int = 1 parameter
- [x] ISC-2: find_cdf_files glob pattern uses f-string mms{probe}
- [x] ISC-3: find_data_pickles signature gains probe: int = 1 parameter
- [x] ISC-4: find_data_pickles glob uses f-string mms/{probe}
- [x] ISC-5: ensure_data_pickle gains probe param and uses it in pickle_path
- [x] ISC-6: _cdf_to_df gains probe and interpolates CDF var names as f-strings
- [x] ISC-7: _fft_pickle_path gains probe and uses it in path f-string
- [x] ISC-8: gen_fft gains probe and forwards to all helpers it calls
- [x] ISC-9: grep mms1 and mms/1/ returns only __main__ driver hits (out of scope)
- [x] ISC-10: python -c import ffttest succeeds and defaults show probe=1

## Decisions

- _compute_file_ffts and _compute_and_save_fft do NOT take probe: they receive paths as arguments and never build probe-dependent strings. Per user instruction ("only add where used").
- ensure_data_pickle forwards probe into _cdf_to_df (required because _cdf_to_df now uses probe for CDF variable name lookups).
- Module-level sample path variables (lines 410-416) and __main__ driver cdf_path strings (lines 433, 437, 447) left untouched — they are dead/driver code, outside surgical scope.

## Verification

- Import sanity: python3 -c "import ffttest; ..." succeeds with defaults (1,) for all parameterized functions.
- Tab indentation verified on lines 38, 65, 122, 155: tabs preserved, no space-conversion.
- All 24 string replacements applied exactly once each.
