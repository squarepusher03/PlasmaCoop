import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np

THRESHOLD = 1e4
EPS = np.finfo(float).tiny

title = ""
name = ""

def symlog(y, linthresh=1e-3, base=10.0):
    """Simple symmetric log transform.
    Linear for |y| <= linthresh (scaled to [-1,1] region),
    and logarithmic outside with continuity at the boundary.
    """
    y = np.asarray(y, dtype=float)
    sign = np.sign(y)
    ay = np.abs(y)
    out = np.empty_like(ay)
    mask = ay <= linthresh
    # Linear region scaled so that y=±linthresh maps to ±1
    out[mask] = ay[mask] / linthresh
    # Log region offset by 1 to meet the linear region continuously
    out[~mask] = 1.0 + np.log(ay[~mask] / linthresh) / np.log(base)
    return sign * out

def yformat(val, pos):
    if val >= THRESHOLD:
        exp = int(np.round(np.log10(val)))
        foo = val / 10 ** exp
        return r"${:.1f} \times 10^{:d}$".format(foo, exp)
    elif val <= -THRESHOLD:
        exp = int(np.round(np.log10(np.abs(val))))
        foo = val / 10 ** exp
        return r"${:.1f} \times -10^{:d}$".format(foo, exp)
    else:
        return r'%d' % int(val)  # empty string => no label shown

# zoom is only on the x-axis
def bushi(df, logscl=False, zoom=None, bins=30):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title)
    ax.set_title(name)
    ax.set_ylabel("Counts of values")

    if logscl:
        ax.set_yscale('log')

    ax.hist(df, align='left', range=zoom, bins=bins)

    ax.yaxis.set_major_formatter(FuncFormatter(yformat))
    ax.grid(True, which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)

    fname = '1v'

    fname += 'So' if logscl else 'Si'
    if zoom:
        fname += 'z_'
        fname += str(zoom[0]) + ',' + str(zoom[1])
    else:
        fname += 'n'
    fname += '.png'

    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

def bushit(dfx, dfy, logscl=False, logcol=False, fill=False, zoom=None, bins=200):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title)

    ax.set_title(name)
    ax.set_xlabel(r'$B_z$ (nT)')
    ax.set_ylabel(r'$\Delta B_z$ $(^{\text{nT}}\!\!/_{6\text{ sec}})$')

    cm = 1

    # Apply simple symlog transform to y-values when logscl=True
    if logscl:
        y_plot = symlog(dfy)
    else:
        y_plot = dfy
    ax.set_yscale('linear')

    hrange = None
    if zoom:
        if logscl:
            y0, y1 = zoom[1]
            y0t, y1t = symlog(np.array([y0, y1]))
            hrange = (zoom[0], (float(min(y0t, y1t)), float(max(y0t, y1t))))
        else:
            hrange = zoom

    if logcol:
        h, xedges, yedges, im = ax.hist2d(dfx, y_plot, cmin=cm, range=hrange, norm=LogNorm(), bins=bins)
    else:
        h, xedges, yedges, im = ax.hist2d(dfx, y_plot, cmin=cm, range=hrange, bins=bins)

    cbar = fig.colorbar(im, ax=ax)
    colscale = 'log' if logcol else 'linear'
    cbar.set_label(f'Counts ({colscale} scale)')

    if fill:
        # Do not alter the image data or color limits; just paint the background
        # with the lowest color so empty bins show as the minimum without
        # affecting colors of non-empty bins.
        vmin0, vmax0 = im.get_clim()
        low_color = im.get_cmap()(im.norm(vmin0))
        ax.set_facecolor(low_color)

    # ax.set_yticks(range(-100, 101, 25)) #changed after running

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(yformat))

    # Draw grey gridlines on major ticks for better readability
    ax.grid(True, which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    fname = '2v'

    fname += 'So' if logscl else 'Si'
    fname += 'Co' if logcol else 'Ci'
    fname += 'f' if fill else 'n'
    if zoom:
        fname += 'z_'
        fname += str(zoom[0][0]) + ',' + str(zoom[0][1]) + '.' + str(zoom[1][0]) + ',' + str(zoom[1][1])
    else:
        fname += 'n'
    fname += '.png'

    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

def bushita(dfx, dfy, dfz, logscl=None, logcol=False, fill=False, zoom=None, clabel=None, bins=200):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title)

    ax.set_title(name)
    ax.set_xlabel(r'$B_z$ (nT)')
    ax.set_ylabel(r'$\Delta B_z$ $(^{\text{nT}}\!\!/_{6\text{ sec}})$')

    clabel = 'C' if not clabel else clabel

    if logscl:
        y_plot = symlog(dfy, linthresh=1e-4)
    else:
        y_plot = dfy
    ax.set_yscale('linear')

    # Always use original hexbin so existing bin colors remain identical.
    norm = LogNorm() if logcol else None
    _s_mean = lambda a: np.nan if len(a) == 0 else np.nanmean(a)
    if zoom:
        if logscl:
            y0, y1 = zoom[1]
            y0t, y1t = symlog(np.array([y0, y1]))
            extent = (zoom[0][0], zoom[0][1], float(min(y0t, y1t)), float(max(y0t, y1t)))
        else:
            extent = (zoom[0][0], zoom[0][1], zoom[1][0], zoom[1][1])
        hb = ax.hexbin(dfx, y_plot, C=dfz, reduce_C_function=_s_mean, extent=extent, bins=bins, norm=norm)
    else:
        hb = ax.hexbin(dfx, y_plot, C=dfz, reduce_C_function=_s_mean, bins=bins, norm=norm)
    cbar = fig.colorbar(hb, ax=ax)
    colscale = 'log' if logcol else 'linear'
    cbar.set_label(f'Mean {clabel} flux ({colscale} scale)')
    if fill:
        # Paint the background with the lowest color from the current normalization
        vmin0, vmax0 = hb.get_clim()
        low_color = hb.get_cmap()(hb.norm(vmin0))
        ax.set_facecolor(low_color)

    # ax.set_yticks(range(-100, 101, 25)) #changed after running

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(yformat))

    # Draw grey gridlines on major ticks for better readability
    ax.grid(True, which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    fname = clabel.split()[0] + '3v'

    fname += 'So' if logscl else 'Si'
    fname += 'Co' if logcol else 'Ci'
    fname += 'f' if fill else 'n'
    if zoom:
        fname += 'z_'
        fname += str(zoom[0][0]) + ',' + str(zoom[0][1]) + '.' + str(zoom[1][0]) + ',' + str(zoom[1][1])
    else:
        fname += 'n'
    fname += '.png'

    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

def draw_profile3v(dfx, dfy, bins, dfz=None, label='C', zoom=None):
    bushita(dfx=dfx, dfy=dfy, dfz=dfz, clabel=label, zoom=zoom, bins=bins)
    bushita(dfx=dfx, dfy=dfy, dfz=dfz, fill=True, clabel=label, zoom=zoom, bins=bins)
    bushita(dfx=dfx, dfy=dfy, dfz=dfz, logscl=True, clabel=label, zoom=zoom, bins=bins)
    bushita(dfx=dfx, dfy=dfy, dfz=dfz, logscl=True, fill=True, clabel=label, zoom=zoom, bins=bins)

    bushita(dfx=dfx, dfy=dfy, dfz=dfz, logcol=True, clabel=label, zoom=zoom, bins=bins)
    bushita(dfx=dfx, dfy=dfy, dfz=dfz, logcol=True, fill=True, clabel=label, zoom=zoom, bins=bins)
    bushita(dfx=dfx, dfy=dfy, dfz=dfz, logcol=True, logscl=True, clabel=label, zoom=zoom, bins=bins)
    bushita(dfx=dfx, dfy=dfy, dfz=dfz, logcol=True, logscl=True, fill=True, clabel=label, zoom=zoom, bins=bins)

def draw_profile2v(dfx, dfy, bins, zoom=None):
    bushit(dfx=dfx, dfy=dfy, zoom=zoom, bins=bins)
    bushit(dfx=dfx, dfy=dfy, logcol=True, zoom=zoom, bins=bins)
    bushit(dfx, dfy, fill=True, zoom=zoom, bins=bins)
    bushit(dfx=dfx, dfy=dfy, logcol=True, fill=True, zoom=zoom, bins=bins)

    bushit(dfx=dfx, dfy=dfy, logscl=True, zoom=zoom, bins=bins)
    bushit(dfx=dfx, dfy=dfy, logscl=True, logcol=True, zoom=zoom, bins=bins)
    bushit(dfx, dfy, logscl=True, fill=True, zoom=zoom, bins=bins)
    bushit(dfx=dfx, dfy=dfy, logscl=True, logcol=True, fill=True, zoom=zoom, bins=bins)

def draw_profile1v(dfx, bins, zoom=None):
    bushi(dfx, zoom=zoom, bins=bins)
    bushi(dfx, logscl=True, zoom=zoom, bins=bins)