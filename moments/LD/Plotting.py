# Methods to plot LD decay curves from moments.LD
# !!!!
# WARNING: These functions are meant to serve as examples or inspiration. They are not
# documented, nor are they nicely formatted or take friendly inputs. Similarly, they
# are not currently maintained, aside from being usable in some example code blocks in
# the documentation.
# !!!!

import matplotlib.pylab as plt, matplotlib
import numpy as np
import sys, os


FONT_SETTINGS = {
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'font.size': 8,
    'font.style': 'normal',
    'font.family': 'sans-serif',
}


def plot_ld_curves(
    ld_stats,
    stats_to_plot=[],
    rows=None,
    cols=None,
    statistics=None,
    fig_size=(6, 6),
    dpi=150,
    r_edges=None,
    numfig=1,
    cM=False,
    output=None,
    show=False,
):
    """
    Plot single set of LD curves
    LD curves are named as given in statistics

    ld_stats is the output of bin_stats
    stats_to_plot is a list of lists, where each inner list gives the stats
        to plot in a given pane, and the list of lists are each pane
    rows and cols tells us how to arrange the panes, if there are more than one
        sets of statistics to plot
    if statistics is None, we use the statistics as listed in ld_stats.names()

    For example, to plot four panes, two on each of two rows, with the D^2
        statistics, the between-population cov(D) statistics, the non-cross
        populations Dz statistics, and the non-cross pop pi2 statistics, we run
        plot_ld_curves(ld_stats, stats_to_plot=[['DD_0_0','DD_1_1','DD_2_2'],
            ['DD_0_1','DD_0_2','DD_1_2'],['Dz_0_0_0','Dz_1_1_1','Dz_2_2_2'],
            ['pi2_1_1_1_1','pi2_2_2_2_2']], rows=2, cols=2,
            statistics=statistics)

    If you want to save the figure, set output to the file path+name
    """
    num_axes = len(stats_to_plot)
    if num_axes == 0:
        return

    if rows == None and cols == None:
        cols = len(stats_to_plot)
        rows = 1

    if statistics == None:
        statistics = ld_stats.names()

    # make sure all stats are named properly
    r_edges = np.array(r_edges)
    r_centers = np.array((r_edges[:-1] + r_edges[1:]) / 2)
    x_label = "$r$"
    if cM == True:
        r_centers *= 100
        x_label = "cM"

    fig = plt.figure(numfig, figsize=fig_size, dpi=dpi)
    fig.clf()

    axes = {}
    # loop through stats_to_plot, update axis, and plot
    for i, stats in enumerate(stats_to_plot):
        # we don't want to plot log-scale if we predict negative values for stats
        neg_vals = False

        axes[i] = plt.subplot(rows, cols, i + 1)
        for stat in stats:
            k = statistics[0].index(stat)
            to_plot = [ld_stats[j][k] for j in range(len(r_centers))]
            axes[i].plot(r_centers, to_plot, label=stat)
            if np.any([e < 0 for e in to_plot]):
                neg_vals = True

        axes[i].set_xscale("log")
        # don't log scale y axis for pi stats
        for stat in stats:
            if not (stat.startswith("pi2") or neg_vals):
                axes[i].set_yscale("log")

        # only place x labels at bottom of columns
        if i >= len(stats_to_plot) - cols:
            axes[i].set_xlabel(x_label)
        axes[i].legend(frameon=False, fontsize=6)
        # only place y labels on left-most column
        if i % cols == 0:
            axes[i].set_ylabel("Statistic")

    with matplotlib.rc_context(FONT_SETTINGS):
        fig.tight_layout()

        if output != None:
            plt.savefig(output)

        if show == True:
            fig.show()
        else:
            return fig


def plot_ld_curves_comp(
    ld_stats,
    ms,
    vcs,
    stats_to_plot=[],
    rows=None,
    cols=None,
    statistics=None,
    fig_size=(6, 6),
    dpi=150,
    rs=None,
    numfig=1,
    cM=False,
    output=None,
    show=False,
    plot_means=True,
    plot_vcs=False,
    binned_data=True,
    ax=None,
    labels=None,
):
    """
    Plot comparison between expected stats (y) and data (ms, vcs)

    stats_to_plot is a list of lists, where each inner list gives the stats
        to plot in a given pane, and the list of lists are each pane
    rows and cols tells us how to arrange the panes, if there are more than one
        sets of statistics to plot
    if statistics is None, we use the statistics as listed in ld_stats.names()

    if binned_data is True, then rs defines the edges of bins, so that ld_stats,
        ms, and vcs have length of ld stats equal to rs-1, while if
        binned data is False, ld_stats and ms is plotted at values of rs,
        and have equal length to rs.

    For example, to plot four panes, two on each of two rows, with the D^2
        statistics, the between-population cov(D) statistics, the non-cross
        populations Dz statistics, and the non-cross pop pi2 statistics, we run
        plot_ld_curves(ld_stats, stats_to_plot=[['DD_0_0','DD_1_1','DD_2_2'],
            ['DD_0_1','DD_0_2','DD_1_2'],['Dz_0_0_0','Dz_1_1_1','Dz_2_2_2'],
            ['pi2_1_1_1_1','pi2_2_2_2_2']], rows=2, cols=2,
            statistics=statistics)

    Otherwise we can pass an ax object in a fig that already exists, in which
        case stats_to_plot must have length 1 (with as many statistics you want
        to plot within that axis).
    """

    # Check that all the data has the correct dimensions
    if binned_data and (len(ld_stats.LD()) != len(rs) - 1):
        raise ValueError("binned_data True, but incorrect length for given rs.")
    if (binned_data == False) and (len(ld_stats.LD()) != len(rs)):
        raise ValueError("binned_data False, incorrect length for given rs.")

    if labels is not None:
        assert len(labels) == len(stats_to_plot)
    if labels is None:
        labels = stats_to_plot

    # set up fig and axes
    if ax is None:
        num_axes = len(stats_to_plot)
        if num_axes == 0:
            return

        if rows == None and cols == None:
            cols = len(stats_to_plot)
            rows = 1
        elif cols == None:
            cols = int(np.ceil(len(stats_to_plot) / rows))
        elif rows == None:
            rows = int(np.ceil(len(stats_to_plot) / cols))

        fig = plt.figure(numfig, figsize=fig_size, dpi=dpi)
        fig.clf()
        axes = {}

        for i, stats in enumerate(stats_to_plot):
            axes[i] = plt.subplot(rows, cols, i + 1)

    else:
        axes = [ax]

    if statistics == None:
        statistics = ld_stats.names()

    # make sure all stats are named properly
    rs = np.array(rs)
    if binned_data:
        rs_to_plot = np.array((rs[:-1] + rs[1:]) / 2)
    else:
        rs_to_plot = rs

    x_label = "$r$"
    if cM == True:
        rs_to_plot *= 100
        x_label = "cM"

    # loop through stats_to_plot, update axis, and plot
    for i, (stats, label) in enumerate(zip(stats_to_plot, labels)):
        # we don't want to plot log-scale if we predict negative values for stats
        neg_vals = False

        axes[i].set_prop_cycle(None)
        if plot_vcs:
            for stat in stats:
                k = statistics[0].index(stat)
                data_to_plot = np.array([ms[j][k] for j in range(len(rs_to_plot))])
                data_error = np.array(
                    [vcs[j][k][k] ** 0.5 * 1.96 for j in range(len(rs_to_plot))]
                )
                axes[i].fill_between(
                    rs_to_plot,
                    data_to_plot - data_error,
                    data_to_plot + data_error,
                    alpha=0.25,
                    label=None,
                )

        # reset color cycle
        axes[i].set_prop_cycle(None)
        if plot_means:
            for stat in stats:
                k = statistics[0].index(stat)
                data_to_plot = np.array([ms[j][k] for j in range(len(rs_to_plot))])
                axes[i].plot(rs_to_plot, data_to_plot, "--", label=None)

        # reset color cycle
        axes[i].set_prop_cycle(None)
        for ind, stat in enumerate(stats):
            k = statistics[0].index(stat)
            exp_to_plot = [ld_stats[j][k] for j in range(len(rs_to_plot))]
            if np.any([e < 0 for e in exp_to_plot]):
                neg_vals = True
            axes[i].plot(rs_to_plot, exp_to_plot, label=label[ind])

        axes[i].set_xscale("log")
        # don't log scale y axis for pi stats
        for stat in stats:
            if not (stat.startswith("pi2") or neg_vals):
                axes[i].set_yscale("log")

        # only place x labels at bottom of columns
        if i >= len(stats_to_plot) - cols:
            axes[i].set_xlabel(x_label)
        axes[i].legend(frameon=False, fontsize=6)
        # only place y labels on left-most column
        if i % cols == 0:
            axes[i].set_ylabel("Statistic")

    if ax is None:
        with matplotlib.rc_context(FONT_SETTINGS):
            fig.tight_layout()
            if output != None:
                plt.savefig(output)
            if show == True:
                fig.show()
            else:
                return fig
