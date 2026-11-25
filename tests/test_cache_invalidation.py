"""
Staleness tests for ffttest._invalidate_if_stale.

Two independent signals decide whether a cached pickle is still usable:
source mtime (checked first) and generating-code version. Either one alone is
insufficient -- version misses a re-downloaded CDF, mtime misses a code change
-- so both are exercised here, including the cascade that makes one changed CDF
invalidate exactly the FFT pickles that depend on it.

Run (needs numpy/pandas -- the `plasma` conda env):

    python tests/test_cache_invalidation.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

import ffttest
from ffttest import _invalidate_if_stale, _write_version, _read_version, _version_path


def _touch(path, content='x', mtime=None):
    with open(path, 'w') as f:
        f.write(content)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _make_pickle(d, name, version, mtime):
    """A stand-in cached artifact plus its version sidecar."""
    p = _touch(os.path.join(d, name), 'cached', mtime=mtime)
    _write_version(p, version)
    os.utime(_version_path(p), (mtime, mtime))
    return p


def test_source_newer_invalidates_even_at_current_version():
    """The case existence+version could never catch: a re-downloaded CDF."""
    with tempfile.TemporaryDirectory() as d:
        pkl = _make_pickle(d, 'data.pkl', 2, mtime=1000)
        src = _touch(os.path.join(d, 'src.cdf'), mtime=2000)   # newer
        _invalidate_if_stale(pkl, 2, source_paths=(src,))
        assert not os.path.exists(pkl), 'stale-by-mtime pickle survived'
        assert not os.path.exists(_version_path(pkl)), 'sidecar left behind'
    print('  ok  source newer than pickle -> invalidated at matching version')


def test_source_older_and_version_matching_is_kept():
    with tempfile.TemporaryDirectory() as d:
        pkl = _make_pickle(d, 'data.pkl', 2, mtime=2000)
        src = _touch(os.path.join(d, 'src.cdf'), mtime=1000)   # older
        _invalidate_if_stale(pkl, 2, source_paths=(src,))
        assert os.path.exists(pkl), 'current pickle was wrongly deleted'
        assert _read_version(pkl) == 2
    print('  ok  source older + version match -> kept')


def test_equal_mtime_does_not_invalidate():
    """Strict >: a pickle written in the same tick as its source is fine, so
    coarse filesystem timestamp granularity cannot cause endless rebuilds."""
    with tempfile.TemporaryDirectory() as d:
        pkl = _make_pickle(d, 'data.pkl', 2, mtime=1500)
        src = _touch(os.path.join(d, 'src.cdf'), mtime=1500)
        _invalidate_if_stale(pkl, 2, source_paths=(src,))
        assert os.path.exists(pkl), 'equal mtime treated as stale (would thrash)'
    print('  ok  equal mtime -> kept (no thrash)')


def test_version_mismatch_still_invalidates_with_older_source():
    """Signal 2 must keep working on its own -- a code change with untouched data."""
    with tempfile.TemporaryDirectory() as d:
        pkl = _make_pickle(d, 'data.pkl', 2, mtime=2000)
        src = _touch(os.path.join(d, 'src.cdf'), mtime=1000)
        _invalidate_if_stale(pkl, 3, source_paths=(src,))   # expected version bumped
        assert not os.path.exists(pkl), 'version-stale pickle survived'
    print('  ok  version mismatch -> invalidated even with older source')


def test_no_sources_falls_back_to_version_only():
    """Backwards-compatible: the old call shape still behaves as before."""
    with tempfile.TemporaryDirectory() as d:
        keep = _make_pickle(d, 'keep.pkl', 2, mtime=1000)
        _invalidate_if_stale(keep, 2)
        assert os.path.exists(keep)
        drop = _make_pickle(d, 'drop.pkl', 1, mtime=1000)
        _invalidate_if_stale(drop, 2)
        assert not os.path.exists(drop)
    print('  ok  no source_paths -> version-only behaviour preserved')


def test_missing_source_keeps_pickle():
    """A vanished source is not evidence the cache is wrong -- deleting a
    usable entry over a missing input would be strictly worse."""
    with tempfile.TemporaryDirectory() as d:
        pkl = _make_pickle(d, 'data.pkl', 2, mtime=1000)
        _invalidate_if_stale(pkl, 2, source_paths=(os.path.join(d, 'gone.cdf'),))
        assert os.path.exists(pkl), 'pickle deleted because its source vanished'
    print('  ok  missing source -> pickle kept')


def test_any_newer_source_invalidates():
    """FFT pickles have TWO sources (own file + next file, for the seam):
    a newly arrived later file must invalidate the preceding file's FFT."""
    with tempfile.TemporaryDirectory() as d:
        fft = _make_pickle(d, 'fft.pkl', 2, mtime=1000)
        curr = _touch(os.path.join(d, 'curr.pkl'), mtime=900)    # unchanged
        nxt = _touch(os.path.join(d, 'next.pkl'), mtime=1100)    # just arrived
        _invalidate_if_stale(fft, 2, source_paths=(curr, nxt))
        assert not os.path.exists(fft), 'new next-file did not invalidate seam FFT'
    print('  ok  newer next-file source -> seam FFT invalidated')


def test_none_source_is_ignored():
    """next_path is None for the last file in a range -- must not crash."""
    with tempfile.TemporaryDirectory() as d:
        fft = _make_pickle(d, 'fft.pkl', 2, mtime=1000)
        curr = _touch(os.path.join(d, 'curr.pkl'), mtime=900)
        _invalidate_if_stale(fft, 2, source_paths=(curr, None))
        assert os.path.exists(fft)
    print('  ok  None source entry ignored')


def test_cascade_cdf_to_data_to_fft():
    """One changed CDF must invalidate its data pickle AND that pickle's FFTs.

    Simulated at the invalidation level: refreshing the data pickle bumps its
    mtime, which is exactly what the FFT layer checks against.
    """
    with tempfile.TemporaryDirectory() as d:
        cdf = _touch(os.path.join(d, 'src.cdf'), mtime=3000)     # re-downloaded
        data = _make_pickle(d, 'data.pkl', 2, mtime=1000)
        fft = _make_pickle(d, 'fft.pkl', 2, mtime=1500)

        # Layer 1: CDF newer than data pickle -> data pickle dropped.
        _invalidate_if_stale(data, 2, source_paths=(cdf,))
        assert not os.path.exists(data)

        # Rebuild it now (fresh mtime), as ensure_data_pickle would.
        data = _make_pickle(d, 'data.pkl', 2, mtime=time.time())

        # Layer 2: data pickle now newer than the FFT -> FFT dropped too.
        _invalidate_if_stale(fft, 2, source_paths=(data, None))
        assert not os.path.exists(fft), 'FFT survived a regenerated data pickle'
    print('  ok  cascade: changed CDF -> data pickle -> FFT pickle')


def test_call_sites_pass_sources():
    """Guard against a future edit dropping source_paths at a call site."""
    import inspect
    src = inspect.getsource(ffttest)
    calls = [ln.strip() for ln in src.splitlines() if '_invalidate_if_stale(' in ln
             and not ln.strip().startswith('def ')]
    assert calls, 'no call sites found -- test is stale'
    for c in calls:
        assert 'source_paths' in c or c.endswith(','), (
            f'call site without source_paths: {c}')
    print(f'  ok  all {len(calls)} call site(s) pass source_paths')


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\n{len(tests)} passed')


if __name__ == '__main__':
    _run()
