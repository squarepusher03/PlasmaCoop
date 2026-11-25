# Design: the acquisition phase (validate once, up front)

_2026-09-27. Follows `HANDOFF_acquisition-phase-and-validation-audit.md` §5. Nothing is built yet and no
existing file has changed. Four decisions (§6) are needed before any code is written._

Graham's directive (2026-09-27, verbatim in the handoff): acquire every CDF for a range → validate every
received file → retry only the corrupt ones, **5 downloads total** → still corrupt: delete + manifest note.
Everything after that never validates, never deletes, and never meets a corrupt file.

## 1. What pyspedas actually does (read from source, both environments)

The Windows venv has pyspedas 1.7.28 and `plasma` has 2.1.1. `projects/mms/mms_load_data.py` is the same in
both where it matters:

- **Skip rule (:166).** A file is skipped only if it exists **and** its size equals the SDC listing's
  `file_size`. Any other local copy is downloaded again to a temp file and `copy()`'d over the old one (:183–201).
  - An HTML 429 page (~75 bytes) or a truncated download is therefore **overwritten in place** by the next
    download. **No delete is needed to retry it.**
  - Only a **same-size** corrupt copy is skipped. That is the one case where the bad copy has to be moved
    out of the way first.
- **No status check on the file download (:182).** A 429 body is written under the CDF name. This is how the
  "corrupt" files seen so far were made (memory `project_sdc_rate_limit`).
- **The skip check looks only in `CONFIG['local_data_dir']`.** A download pointed at a staging folder
  re-downloads every file of the range, including the ones pydata already has.
- **A failed listing returns 0 files silently (:122–126).** "Nothing new arrived" can mean rate-limited.
- **`CONFIG['download_only'] = True`** returns the file list without loading it into tplot
  (`scm.py:142`, `fgm.py:147`). Today `_do_download` loads the whole range into tplot and then calls
  `del_data()`, a full read of every file that no one uses. Optional speed-up, and an edit to existing code.

## 2. The acquisition unit

`acquire(field, probes, start, end)`, with field B / E / F. It must cover every file the compute phase can open.
The audit showed that set reaches outside `[start, end]`. Per probe, from filename stamps alone (no opens), using
the live helpers `_scan_cdf_files`, `_anchor_index` and `_fall_forward`:

1. the naive anchor: the rightmost stamp ≤ `start`, or the first stamp after it (`_fall_forward`)
2. every stamp from the anchor through `end`, **all versions** of each stamp (a rejected v2.2.1 leaves v2.2.0
   to stand, so v2.2.0 must be valid too)
3. the on-disk **predecessor** of the anchor and **successor** of the last file (`pair_with_next_cdf`'s neighbours)

If a give-up removes a file, the set is recomputed, because a neighbour moves one file outward and may be a
legacy file no one has validated. This loops until nothing new appears. It is bounded, since each pass
validates or gives up at least one file.

## 3. The entry point: a new module, `acquisition.py`

```
acquire(field, probes, start, end, *, data_dir, pickle_root, download=True, log=print) -> Acquired
  1. download   attempt_download(inst, mode, probes, start - GAP_FILL_LOOKBACK, end, data_dir)   [if download]
  2. candidates §2, per probe
  3. validate   _check_cdf + cdf_validation (DB pass = free; library validation ON)   -- records, never deletes
  4. retry      failures only, rounds 2..5: get the bad copy out of pyspedas' way (§6 Q2), narrow download
                over just the failed files' span, validate again
  5. give up    5th failure: delete, count, name goes in the manifest's corrupted section
  6. return     per probe: the validated files; the given-up names; what was retried
```

It reuses `attempt_download` (subprocess, timeout, lock), `_check_cdf`, the `cdf_validation` DB and
`GIVE_UP_AFTER = 5` (which already means "first download + 4 retries"). `check_cdfs` is not reused, because it
deletes on the first failure. That is exactly the behaviour the directive removes.

## 4. What changes in existing code (each one needs your OK when we get there)

**Downloads move into acquisition.** Today they happen in six places:

| Call site | Today | After |
|---|---|---|
| `find_cdf_files` :504, :473 (anchor repair), :520 (end repair) | downloads inside every request | callers pass `download=False`; the repairs become acquisition's step 1 |
| `collect_fft_data` / `gen_fft` | `download=True` default | callers acquire first, then pass `download=False` |
| `event_export.download_missing` :281 | bare `attempt_download` | `acquire(...)` |
| `plot_biggest_spikes._download_request` :232 | bare `attempt_download` in the download thread | `acquire(...)` in the same thread |
| `download.py` | whole-day retry loop over `corrupted_log` | `acquire(...)` per day; the log loop retires |
| `short_interval._stream_binned` → `find_cdf_files(download=...)` | inside compute | `ShortIntervalPlotter.plot` acquires first |

**Validation and deletion leave the compute path.** These 9 sites go away or become no-ops:
`cdf_check` in `_plan_fft_jobs`, `collect_fft_data`, `gen_fft`, `_stream_binned`, `fce.stream_fgm_bmag`,
`spike.ensure_range_pickles`, `event_export.count_local_files`, `fgm_runs` and `_event_fft_paths`.
Keeping or removing the `check=` parameters themselves is your call (no-unilateral-removal).

**Where acquisition runs, per entry point:**

- `main.py`: `ShortIntervalPlotter.plot` calls `acquire` for its range, then computes.
- **Event UI**: *Download* → `acquire(download=True)`. *Don't download* and the plain coverage check →
  `acquire(download=False)`, which validates what is on disk before `count_local_files` reads a bound.
- `plot_biggest_spikes`: the download thread calls `acquire`. Compute for request *i* runs alongside the
  acquisition of request *i+1*. See the hazard in §5.
- `spike_biggest_burst_fft`, `fce`, `event_images` / `export_event`: acquire once up front. They already pass
  `download=False` in places.

## 5. Problems the directive's plain reading runs into

1. **Rate limiting looks like corruption.** Four immediate retries during a 429 return four more HTML pages,
   and a good file is given up. The throttle has outlasted its 300 s `Retry-After` twice in one session
   (memory, 2026-09-14). An HTML body is easy to tell apart without the CDF library: the first two bytes are
   not `\xcd\xf3` or `\xcd\xf2`. → §6 Q4.
2. **Download-disabled runs.** If a corrupt legacy file turns up in `acquire(download=False)`, it can't be
   retried, and deleting it breaks the 5-download rule. It still must not reach compute. The same mechanism
   as Q2 applies: move it out of pydata and count nothing, and the next downloading acquisition retries it.
3. **The pipelined download thread.** Today pyspedas writes new files straight into pydata with a
   non-atomic `copy()`. If request *i+1*'s range touches request *i*'s seam, compute for *i* can pick up a
   half-written or unvalidated neighbour. This hazard exists today; only staging (Q2 option B) closes it
   completely.
4. **A rate-limited run leaves holes.** Compute then runs on what is on disk, exactly as for data the SDC
   doesn't have. The manifest should say "pending (rate-limited)" so the hole isn't mistaken for
   "no burst data". The name keeps its attempt count unchanged, and the next acquisition fetches it.
5. **Quarantine details (advisor review).** The folder sits in the cache tree, outside pydata and outside
   anything pyspedas scans. Quarantined names must not end in `.cdf`. A download killed at the timeout leaves
   a partial file (pyspedas `copy()`s straight into the target), and the same run's validation catches it.
6. **Missing seam neighbours** (not corruption, noted only). A neighbour that was never downloaded is not
   fetched by a `[start, end]` download, so the last file's FFT drops its final windows. Whether acquisition
   should widen its download to fetch neighbours ties into the seam history question (memory: UNVERIFIED).

## 6. Decisions (Graham, 2026-09-27)

- **Q2: delete the corrupt copy in place, every time.** Whenever a download attempt leaves a corrupt file, it
  is deleted at once, whichever attempt it was. The number of attempts is unchanged: 5 downloads total, and
  the 5th failure ends in the manifest note. No quarantine and no staging ("moving files sounds slow"). §5.5
  no longer applies. A download-disabled run that finds a corrupt legacy file deletes it and counts it, and
  the next downloading acquisition retries it (§5.2).
- **Q3: validate a returning given-up name** like any new file. Intact: the record clears. Corrupt: deleted
  at once, with no further retries.
- **Q5: library validate-on-open OFF for every open after acquisition.** Acquisition's own checks stay ON.
- **Q4: an HTML 429 page does not count** toward the 5, **and acquisition paces its SDC calls** with a small
  buffer between them so that we don't get rate-limited in the first place. On a detected 429 page: delete
  it, leave the count unchanged, stop retrying this run, mark the manifest "pending (rate-limited)".
  **Buffer (Graham, 2026-09-28, superseding the 10 s / +5 s rule the same day):** "I would rather it start
  at 20s and go up 15s each time and cap out at 90s. It should drop back down to 20s after receiving a CDF
  or capping out." → `SDC_CALL_GAP_S = 20`, `SDC_GAP_STEP_S = 15`, `SDC_GAP_MAX_S = 90`.
  - **Refusals:** a run of refusals waits 35, 50, 65, 80, 90 s, then 20 again. That is 320 s per climb,
    which outlasts the 300 s Retry-After seen before.
  - **"Receiving a CDF":** a download delivered at least one real CDF, even one that later fails
    validation. A call that hit a 429 as well still raises the gap.
  - **"Capping out":** after the one call made at 90 s, the next gap is 20 s, whatever that call returned.
  - **Still no retry limit:** the gap is bounded, not the number of attempts.

### The original questions, for the record

- **Q2: how a retry gets past the bad copy.** A: move it to a quarantine folder in the cache tree (kept until
  the retry lands or give-up), then refetch into pydata. B: redirect pyspedas' final `copy()` into a staging
  folder inside our download subprocess, and move only validated files into pydata. C: overwrite in place only.
- **Q3: a given-up name that a later acquisition brings back** (pyspedas fetches it because it is absent):
  validate it like any file, or delete it on arrival.
- **Q5: the library's validate-on-open after acquisition**: off for every compute-phase open (~60 ms →
  ~0.4 ms each), with acquisition's own checks held ON under `cdf_validation.validation(True)`.
- **Q4 (new): does an HTML 429 page count as one of the 5 downloads?**

Proposed default, unless you say otherwise: the manifest's `# corrupted` section is scoped to the request's
probe and range. Today it lists every given-up file of the instrument (handoff §4 Q4).

## 7. Acceptance

1. Re-point `_wsl_scratch/open_audit.py`: any validation, deletion or download after `acquire` returns is a
   failure. Rerun the 72-run matrix and expect 0.
2. New tests: a corrupt file is retried and then passes; 5 failures → deleted + named in the manifest; a
   same-size corrupt copy is retried; an HTML page is handled per Q4; `download=False` keeps a corrupt file
   away from compute; tests never write to E:.
3. Output unchanged on good data: FFT digests + manifest text for all 12 entry points, seam harness
   163,907 rows, full suite (372 today).

## 8. Status (2026-09-28)

**Built, new files only:** `acquisition.py` and `tests/test_acquisition.py` (34 tests). Full suite:
**408 passed** (372 before + 34 new + 2 automatic call-signature guards). No existing file changed, and the
suite created no validation database on E:.

- **Mutation check (11 breaks).** Each rule was broken on purpose, one at a time: the 20 s start, the
  15 s step, the 90 s cap, the reset after the cap, the reset after a CDF, 429-page detection, validation
  off after acquire, superseded versions, the predecessor in the candidate set, legacy pages raising the
  gap, and validating arrivals. Every break failed at least one test.
- **Real child process.** The exit-code path was checked with a stubbed download (no SDC contact):
  clean → ok, listing 429 → rate_limited, listing 503 → failed, crash → failed.
- **One SDC call per selected probe** (Graham, 2026-09-28: "acquire should not call using all the probes
  at once; only request the probes the user has selected"). A batched call fires every probe's listing
  back to back with no gap, and a 429 on one probe repeats the whole call. The same applies to refetches.

### Wiring edit 1 (approved 2026-09-28): `ShortIntervalPlotter` (the path `main.py` uses)

- **What changed:** with `download=True`, `plot_probes` / `plot_intervals` call `_acquire` first: one
  `acquire()` per merged span (overlapping intervals merged by the live `merge_bursts`), for the selected
  probes only. Every cell then computes with `download=False`.
- **Proof** (`_wsl_scratch/acquire_wiring_proof.py`, pre-edit copy in `_wsl_scratch/pre_2026-09-28_snapshot/`):
  7 good-data requests had identical FFT frames, binned cells and manifest text, with zero downloads during
  compute. The requests covered overlapping and separate intervals, a stale anchor, probes 1+3, two pads,
  and E. With a corrupt file, only the intended difference appears: 5 downloads, then give-up and the
  manifest's `# corrupted` lines.
- **Other checks:** seam harness unchanged (163,907 rows, not recomputed); full suite 410 passed; nothing on
  E:; the per-probe rule's two mutations are both caught.
- **Cost to watch:** `main.py`'s three `plot()` calls now make three paced SDC calls, up to ~20 s of
  waiting before plots 2 and 3.

### Live run of `main.py` (2026-09-29, with Graham's go)

This was run in the Windows venv against the real SDC and E:. Log, plots and CDF snapshots are in
`_wsl_scratch/live_run_2026-09-29/`. Every acquisition behaviour held on real data:
- **SDC calls:** 3 in total, each for MMS1 only, and each OK. The 2nd and 3rd started 20.0 s after the
  previous call ended.
- **The spawned child:** ran cleanly under Windows spawn. `mms.scm()/edp()` took 0.61–0.70 s with
  `download_only`.
- **The validation database:** created on E: with 8 passes (5 scm, 3 edp) and no failures. That is exactly
  the predicted candidate sets.
- **Compute and outputs:** zero downloads or deletions during compute, and library validate-on-open was off
  (0) at the end. The CDFs were unchanged (774 files). There is no `corrupted.txt`, and the manifests have no
  `# corrupted` lines.

See the handoff, §2.7.

### Wired everywhere (2026-09-29, Graham: "After this, do the wiring")

- the event UI: `check_coverage` runs a validate-only acquire before counting (a deleted corrupt file lands in
  `Coverage.refetch` and is offered for download); `download_missing` → `acquire`, Cancel passed through
- `plot_biggest_spikes`: `_validate_present` in pass 1, `_download_request` → `acquire`
- `spike_biggest_burst_fft`: `acquire_inputs()` up front, then `download=False` everywhere (fce and event_images
  read only after an acquire)
- `download.py`: a paced `acquire` per day

Proof and status: handoff §2.10; `tests/test_acquisition_wiring.py`.

Then: remove the compute-path `cdf_check` calls, scope the manifest's `# corrupted` section to the request,
start pool workers with validation off, and re-point `open_audit.py` (§7.1).
