# PlasmaCoop — current goals

_Written 2026-11-24, when Graham asked: "Write the current goals we have, then a handoff summarizing them". The
handoff that summarizes this file is `HANDOFF_bin-in-worker-and-cancel-rule.md`. The facts behind each line are in
`CLAUDE.md` and the project memory._

## The mission

Produce verified FFT spectrogram output (CSV, and graphs where asked) from NASA MMS burst data (SCM |B|, EDP |E|),
for any request in the event window and for the whole mission through the census. Every output must be provably
right: an independent verifier recomputes it from its own copy of the CDFs, and a wrong CSV must stop the run, not
pass silently.

## 1. Next: the whole-MMS CSV census

Graham, 2026-11-22: "After we answer these questions then we can think about the census." Those questions are now
answered: the exit cleanup, binning in the worker, Forge and Cato, and the VACUUM.

| | |
|---|---|
| **What it is** | Each chunk is downloaded, exported CSV-only, checked day by day by the independent verifier, then deleted. Exit 5 cancels the census on any differing CSV (`_wsl_scratch/whole_census_csv_run.sh`). |
| **Where it stopped** | 2026-10-09 08:46, exit 5, in chunk 2 on day 2015-09-14. A lone good EDP sample moved an event's edge; the values matched. |
| **Fixed since** | Short runs (< 10 samples) never count (11-03, `MANIFEST_VERSION` 6). The overlapping-seam data loss is fixed (11-07, v7). The census now uses post-process verification and the archive (11-04). No FFT cache, and since 11-23 no FFT files at all, so a day no longer needs 2.4–3.7x its CDF bytes in FFT scratch. |
| **Checked offline** | `census_export_offline.py` 25 of 25, `census_export_check.py` 25 of 25 (11-24) |
| **Who starts it** | Graham. The restart is his. |
| **To think about first** | the decisions below; E: has 185 GB free (11-24); chunk 1's pace (5.3 h per 117.9 GB, so ~53 days for the mission); the verifier downloads every day again |

**Decisions waiting on Graham before a restart:**
1. **Census STOP:** it deletes the day's output, finished events included, so the day is redone whole. Should the 11-24 cancel rule ("the rest should persist") apply to the census too?
2. **Verifier staging:** the verifier deletes its own copies of the CDFs when it ends, cancelled or not (up to `max_staged_gb` 20). They are duplicates nothing reads again. Keep deleting them?
3. **Chunk size and budget** against 185 GB free. It was 117.9 GB a chunk when E: had 471.6 GB free.

## 2. Keep: the event window is correct, safe and fast

Done. These hold now and must stay true.

| Goal | Status | Evidence |
|---|---|---|
| Output byte-identical across changes | holds | 208 of 208 files on five seam days against the 11-23 baseline, last rerun 11-24 |
| No FFT files on disk | done 11-23 | binned in the workers: 1.08–1.47 GB a day and 8.88 GB (2022-08-19) of scratch → 0 |
| Faster Runs | done 11-23 | 182.7 → 104.0 s over five days; 198–281 → 65–70 s on the heaviest day; memory 1.33 → 0.59 GB |
| Quit mid-Run waits, then closes; no orphan workers | done 11-23 | closes 1.4 s after the quit, rc 0; force quit offered on a second close |
| A cancel deletes only half-made files | done 11-24 | `.part` files of downloads under way and `.partial` files of the event or image being written; everything finished persists, and a re-run keeps its old files until the new ones are whole |
| Post-process verification on every request | holds (11-04) | `verification/report.json` per request |

## 3. Keep: disk under control

| Item | Now |
|---|---|
| E: free | 185 GB (11-24) |
| Archive | 11.84 GB, vacuumed 11-22; the pre-vacuum copy was deleted 11-23 once the checks passed |
| FFT cache | none; don't bring one back |
| Check outputs Graham may delete | ~2.9 GB in all (see the handoff §5); deleting is his call |

## 4. Open, Graham's call (no work pending)

- **Codex CLI for Forge and Cato:** not installed (`~/.bun/bin/codex` missing). Installing it is Graham's decision.
- **Whole months are slow:** downloads dominate, and the verifier downloads everything again.
- **Rerun 2017-03 whole, or 2016-01,** if wanted. The 2016-01 peak (~150 GB, estimated 11-19 with the FFT cache) no longer applies: the export writes no FFT files.

## 5. Not goals now

- The legacy tools' FFT scratch (`plot_biggest_spikes`, `gen_fft`, the `ShortIntervalPlotter` plots, `spike_biggest_burst_fft`). Graham, 11-24: "Forget about scratching performed in files that we don't use."
- The MVC refactor (`controller.py`, `drawer.py`, …), which is described in CLAUDE.md as aspirational and has had no work or ask.
