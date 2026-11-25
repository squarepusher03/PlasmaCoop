"""For SCM SCB (the only instrument with many local files), break the trend
analysis down by DATE. Question: on dates where MMS1 has SCB burst data, do
MMS2 and MMS3 also have it at the same timestamps?
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from data_paths import DATA_DIR  # per-platform E:/ vs /mnt/e

ROOT = Path(DATA_DIR)
SCS = ("mms1", "mms2", "mms3")
TS_RE = re.compile(r"_(20\d{12})_v")


def parse_ts(name: str):
    m = TS_RE.search(name)
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None


def collect_scb(sc):
    base = ROOT / sc / "scm/brst/l2/scb"
    if not base.exists():
        return []
    out = []
    for p in base.rglob("*.cdf"):
        ts = parse_ts(p.name)
        if ts is not None:
            out.append(ts)
    return sorted(out)


def has_within(target, sorted_list, tol_s=60):
    for ts in sorted_list:
        d = (ts - target).total_seconds()
        if abs(d) <= tol_s:
            return True
        if d > tol_s:
            return False
    return False


def main():
    files = {sc: collect_scb(sc) for sc in SCS}

    # date range
    print("=" * 96)
    print("Local SCM SCB date range per spacecraft")
    print("=" * 96)
    for sc in SCS:
        if files[sc]:
            print(f"  {sc}: {files[sc][0].date()}  ..  {files[sc][-1].date()}   ({len(files[sc])} files)")

    # group MMS1 files by date and ask: how many are also on mms2 and mms3?
    by_date = defaultdict(list)
    for ts in files["mms1"]:
        by_date[ts.date()].append(ts)

    print("\n" + "=" * 96)
    print("Per-date triplet check (using MMS1 SCM-SCB timestamps as the reference)")
    print("=" * 96)
    print(f"  {'date':<12} {'mms1':>5}  {'on mms2':>10}  {'on mms3':>10}  {'triplet':>10}  trend")
    rows = []
    for date in sorted(by_date.keys()):
        ts_list = by_date[date]
        n = len(ts_list)
        on2 = sum(1 for t in ts_list if has_within(t, files["mms2"]))
        on3 = sum(1 for t in ts_list if has_within(t, files["mms3"]))
        trip = sum(1 for t in ts_list if has_within(t, files["mms2"]) and has_within(t, files["mms3"]))
        rows.append((date, n, on2, on3, trip))
        tag = "FULL TRIPLET" if trip == n else (
            "mms3 missing" if on2 == n and on3 < n else
            "partial"
        )
        print(f"  {str(date):<12} {n:>5}  {on2:>5}/{n:<4}  {on3:>5}/{n:<4}  {trip:>5}/{n:<4}  {tag}")

    print(f"\n  Total dates: {len(rows)}")
    full = sum(1 for r in rows if r[4] == r[1])
    mms3_only_gap = sum(1 for r in rows if r[2] == r[1] and r[3] < r[1])
    print(f"  Dates where ALL MMS1 timestamps appear on MMS2 AND MMS3 (full triplet): {full}")
    print(f"  Dates where MMS1==MMS2 but MMS3 is short: {mms3_only_gap}")


if __name__ == "__main__":
    main()
