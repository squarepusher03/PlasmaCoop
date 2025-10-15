import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np

THRESHOLD = 1e4


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


def bushit(dfx, dfy, logscl=False, logcol=False, fill=False, zoom=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('THA FGS GSM')

    ax.set_title('Differences in Bz readings where dBz = 6 seconds (nT) vs Bz (nT)')
    ax.set_xlabel('Bz (nT)')
    ax.set_ylabel(r'$\Delta B_z$ $(^{\text{nT}}\!\!/_{6\text{ sec}})$')

    cm = 1 if not (fill or logcol) else 0

    if logscl:
        if dfy.min() < 0:
            ax.set_yscale('symlog', linthreshy=1e-3)
        else:
            ax.set_yscale('log')
    else:
        ax.set_yscale('linear')

    if zoom:
        if logcol:
            h, xedges, yedges, im = ax.hist2d(dfx, dfy, cmin=cm, range=zoom, norm=LogNorm(),
                                              bins=200)
        else:
            h, xedges, yedges, im = ax.hist2d(dfx, dfy, cmin=cm, range=zoom, bins=200)
    else:
        if logcol:
            h, xedges, yedges, im = ax.hist2d(dfx, dfy, cmin=cm, norm=LogNorm(), bins=200)
        else:
            h, xedges, yedges, im = ax.hist2d(dfx, dfy, cmin=cm, bins=200)

    cbar = fig.colorbar(im, ax=ax)
    colscale = 'log' if logcol else 'linear'
    cbar.set_label(f'Counts ({colscale} scale)')

    if fill:
        h_filled = h.copy()
        h_filled[h_filled == 0] = 1
        im.set_array(h_filled.T.ravel())  # transpose to match orientation
        im.set_clim(1, h_filled.max())
        cbar.update_normal(im)

    # ax.set_yticks(range(-100, 101, 25)) #changed after running

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(yformat))

    # Draw grey gridlines on major ticks for better readability
    ax.grid(True, which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    fname = ''

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

def bushita(dfx, dfy, dfz, logscl=False, logcol=False, fill=False, zoom=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('THA FGS GSM')

    ax.set_title('Differences in Bz readings where dBz = 6 seconds (nT) vs Bz (nT)')
    ax.set_xlabel('Bz (nT)')
    ax.set_ylabel(r'$\Delta B_z$ $(^{\text{nT}}\!\!/_{6\text{ sec}})$')

    cm = 1 if not (fill or logcol) else 0

    if zoom:
        if logcol:
            h, xedges, yedges, im = ax.hexbin(dfx, dfy, C=dfz, reduce_C_function=np.mean,
                                              cmin=cm, range=zoom, norm=LogNorm(), bins=200)
        else:
            h, xedges, yedges, im = ax.hexbin(dfx, dfy, C=dfz, reduce_C_function=np.mean,
                                              cmin=cm, range=zoom, bins=200)
    else:
        if logcol:
            h, xedges, yedges, im = ax.hexbin(dfx, dfy, C=dfz, reduce_C_function=np.mean,
                                              cmin=cm, norm=LogNorm(), bins=200)
        else:
            h, xedges, yedges, im = ax.hexbin(dfx, dfy, C=dfz, reduce_C_function=np.mean,
                                              cmin=cm, bins=200)

    cbar = fig.colorbar(im, ax=ax)
    colscale = 'log' if logcol else 'linear'
    cbar.set_label(f'Counts ({colscale} scale)')

    if fill:
        h_filled = h.copy()
        h_filled[h_filled == 0] = 1
        im.set_array(h_filled.T.ravel())  # transpose to match orientation
        im.set_clim(1, h_filled.max())
        cbar.update_normal(im)

    if logscl:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')

    # ax.set_yticks(range(-100, 101, 25)) #changed after running

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(yformat))

    # Draw grey gridlines on major ticks for better readability
    ax.grid(True, which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    fname = ''

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
