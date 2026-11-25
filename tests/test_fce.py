"""Synthetic-data unit tests for fce.py pure functions.

No disk / no E: access — exercises only the math + plotting helpers, which is the
part verifiable off the Windows data drive. The FGM round-trip (CDF->pickle->|B|)
is covered by code inspection, not here.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fce


def test_constant_matches_physics():
    # e / (2π m_e) in Hz/nT, computed independently from CODATA values.
    e, m_e = 1.602176634e-19, 9.1093837015e-31
    expected = e / (2 * np.pi * m_e) * 1e-9  # per nT
    assert abs(fce.FCE_HZ_PER_NT - expected) / expected < 1e-4


def test_fce_from_bmag_scalar():
    assert abs(fce.fce_from_bmag(100.0) - 2799.25) < 1.0  # ~28 Hz/nT * 100


def test_fce_from_bmag_array():
    out = fce.fce_from_bmag([0.0, 100.0, 200.0])
    assert out.shape == (3,)
    assert abs(out[2] - 2 * out[1]) < 1e-6  # linear in |B|


def test_frac_channel_monotonic():
    bins = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    freqs = np.linspace(2, 2000, 50)
    fc = fce._frac_channel(freqs, bins)
    assert np.all(np.diff(fc) >= 0)  # non-decreasing in frequency


def test_frac_channel_hits_bin_indices():
    bins = [1, 10, 100, 1000]
    # exact edges should land on their integer channel index
    assert abs(fce._frac_channel(10, bins) - 1.0) < 1e-9
    assert abs(fce._frac_channel(100, bins) - 2.0) < 1e-9


def test_overlay_draws_on_given_axes():
    bins = [1, 10, 100, 1000, 3000]
    st = np.datetime64('2021-02-23T11:07:33', 'ns')
    times = st + (np.arange(5) * np.timedelta64(100, 'ms')).astype('timedelta64[ns]')
    times = times.astype('datetime64[ns]')
    fce_hz = np.array([280., 281., 279., 282., 280.])
    fig, ax = plt.subplots()
    n_before = len(ax.lines)
    line = fce.overlay_fce(ax, st.astype('datetime64[us]').astype(object), bins, times, fce_hz)
    assert line is not None
    assert len(ax.lines) == n_before + 1
    assert len(ax.texts) >= 1            # inline label drawn (no legend)
    assert ax.get_legend() is None
    assert line.get_linewidth() >= 2.0   # "bigger" line
    xd = line.get_xdata()
    assert abs(xd[0]) < 1e-6 and abs(xd[1] - 0.1) < 1e-6  # seconds since st
    plt.close(fig)


def test_overlay_empty_is_noop():
    fig, ax = plt.subplots()
    line = fce.overlay_fce(ax, np.datetime64('2021-02-23T11:07:33'),
                           [1, 10, 100], np.array([], dtype='datetime64[ns]'),
                           np.array([]))
    assert line is None
    assert len(ax.lines) == 0
    plt.close(fig)


if __name__ == '__main__':
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as e:
            failed += 1
            print(f'FAIL {fn.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
