# Methods to plot LD decay curves from moments.LD
# !!!!
# WARNING: These functions are meant to serve as examples or inspiration. They are not
# nicely formatted and do not take friendly inputs. Similarly, they are not currently
# maintained, aside from being usable in some example code blocks in the documentation.
# !!!!

import matplotlib.pylab as plt, matplotlib
import numpy as np
import sys, os

FONT_SETTINGS = {
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "font.size": 8,
    "font.style": "normal",
    "font.family": "sans-serif",
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
    Plot LD curves (statistics against recombination distance) from a model.

    For example, to plot four panes, two on each of two rows, with the
    :math:`D^2` statistics, the between-population :math:`D_iD_j` statistics,
    the single-population :math:`Dz` statistics, and the single-population
    :math:`\\pi_2` statistics of a three-population model:

    .. code-block:: python

        moments.LD.Plotting.plot_ld_curves(
            ld_stats,
            stats_to_plot=[
                ["DD_0_0", "DD_1_1", "DD_2_2"],
                ["DD_0_1", "DD_0_2", "DD_1_2"],
                ["Dz_0_0_0", "Dz_1_1_1", "Dz_2_2_2"],
                ["pi2_0_0_0_0", "pi2_1_1_1_1", "pi2_2_2_2_2"],
            ],
            rows=2,
            cols=2,
            r_edges=r_bins,
        )

    :param ld_stats: The binned LD statistics, as returned by
        ``moments.LD.Inference.bin_stats``.
    :param stats_to_plot: A list of lists of statistic names. Each inner list
        gives the statistics to plot in one pane.
    :param rows: The number of rows of panes. If neither ``rows`` nor ``cols`` is
        given, the panes are placed in a single row.
    :param cols: The number of columns of panes.
    :param statistics: The names of the statistics in ``ld_stats``, as
        ``[ld_stat_names, h_stat_names]``. If None, ``ld_stats.names()`` is used.
    :param fig_size: The figure size, in inches.
    :param dpi: The figure resolution.
    :param r_edges: The edges of the recombination bins used to compute
        ``ld_stats``. The statistics are plotted at the bin midpoints.
    :param numfig: The matplotlib figure number to draw in.
    :param cM: If True, label the x-axis in cM (scaling ``r`` by 100).
    :param output: If given, save the figure to this file.
    :param show: If True, show the figure. Otherwise, return it.
    :returns: The matplotlib figure, if ``show`` is False.
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
    Plot LD curves from a model along with the data they were fit to, with
    the model drawn as solid lines and the data means as dashed lines.

    For example, to plot four panes, two on each of two rows, with the
    :math:`D^2` statistics, the between-population :math:`D_iD_j` statistics,
    the single-population :math:`Dz` statistics, and the single-population
    :math:`\\pi_2` statistics of a three-population model:

    .. code-block:: python

        moments.LD.Plotting.plot_ld_curves_comp(
            ld_stats,
            means[:-1],
            varcovs[:-1],
            stats_to_plot=[
                ["DD_0_0", "DD_1_1", "DD_2_2"],
                ["DD_0_1", "DD_0_2", "DD_1_2"],
                ["Dz_0_0_0", "Dz_1_1_1", "Dz_2_2_2"],
                ["pi2_0_0_0_0", "pi2_1_1_1_1", "pi2_2_2_2_2"],
            ],
            rows=2,
            cols=2,
            rs=r_bins,
            plot_vcs=True,
        )

    :param ld_stats: The expected LD statistics from the model, binned as the data
        are and normalized in the same way, for example using
        ``moments.LD.Inference.sigmaD2(moments.LD.Inference.bin_stats(...))``.
    :param ms: The data means, one array of LD statistics per recombination bin.
    :param vcs: The data variance-covariance matrices, one per recombination bin.
    :param stats_to_plot: A list of lists of statistic names. Each inner list
        gives the statistics to plot in one pane.
    :param rows: The number of rows of panes. If neither ``rows`` nor ``cols`` is
        given, the panes are placed in a single row.
    :param cols: The number of columns of panes.
    :param statistics: The names of the statistics in the data, as
        ``[ld_stat_names, h_stat_names]``. If None, ``ld_stats.names()`` is used,
        and the data must include all of those statistics, including the
        normalizing statistics (as returned by the ``moments.LD.Parsing``
        functions).
    :param fig_size: The figure size, in inches.
    :param dpi: The figure resolution.
    :param rs: If ``binned_data`` is True, the edges of the recombination bins,
        so that the model and data have one fewer entry than ``rs``, and are
        plotted at the bin midpoints. Otherwise, the recombination distances at
        which the model and data are plotted, with the same length as ``rs``.
    :param numfig: The matplotlib figure number to draw in.
    :param cM: If True, label the x-axis in cM (scaling ``r`` by 100).
    :param output: If given, save the figure to this file.
    :param show: If True, show the figure. Otherwise, return it.
    :param plot_means: If True, plot the data means as dashed lines.
    :param plot_vcs: If True, shade the 95% confidence intervals of the data,
        computed from the variances in ``vcs``.
    :param binned_data: Whether ``rs`` gives bin edges (True) or the points at
        which statistics were computed (False).
    :param ax: A matplotlib axis to draw in, in an existing figure. In this case
        ``stats_to_plot`` must have length one, and nothing is returned.
    :param labels: Legend labels, as a list of lists with the same shape as
        ``stats_to_plot``. If None, the statistic names are used.
    :returns: The matplotlib figure, if ``ax`` is not given and ``show`` is False.
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

    if statistics == None:
        statistics = ld_stats.names()
    # Data statistics are found by their index in statistics, so the data must have
    # the same number of statistics, or the wrong statistics would be plotted
    num_bins = len(rs) - 1 if binned_data else len(rs)
    for x in list(ms[:num_bins]) + list(vcs[:num_bins]):
        if len(x) != len(statistics[0]):
            raise ValueError(
                f"The data have {len(x)} statistics in each recombination bin, but "
                f"{len(statistics[0])} are listed in statistics (or in the model, if "
                "statistics is not given). If statistics is not given, the data must "
                "include the normalizing statistics, as returned by the "
                "moments.LD.Parsing functions."
            )
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
