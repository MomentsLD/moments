import numpy as np
import matplotlib
import pylab
import matplotlib.pyplot as plt
import copy


#: Custom ticks that label only the lowest and highest bins in an FS plot.
class _sfsTickLocator(matplotlib.ticker.Locator):
    def __call__(self):
        "Return the locations of the ticks"

        try:
            vmin, vmax = self.axis.get_view_interval()
            dmin, dmax = self.axis.get_data_interval()
        except AttributeError:
            self.verify_intervals()
            vmin, vmax = self.viewInterval.get_bounds()
            dmin, dmax = self.dataInterval.get_bounds()

        tmin = max(vmin, dmin)
        tmax = min(vmax, dmax)

        return np.array([round(tmin) + 0.5, round(tmax) - 0.5])


#: Custom tick formatter
_ctf = matplotlib.ticker.FuncFormatter(lambda x, pos: "%i" % (x - 0.4))


from moments import Numerics, Inference

##
## 1-population functions
##


def plot_1d_fs(fs, fig_num=None, show=True, ax=None, out=None, ms=3, lw=1):
    """
    Plot a 1-dimensional frequency spectrum.

    Note that all the plotting is done with pylab. To see additional pylab
    methods: "import pylab; help(pylab)". Pylab's many functions are documented
    at http://matplotlib.sourceforge.net/contents.html

    :param fs: A single-population Spectrum
    :param fig_num: If used, clear and use figure fig_num for display.
        If None, a new figure window is created.
    :param show: If True, execute pylab.show command to make sure plot displays.
    :param ax: If None, uses new or specified figure. Otherwise plots in axes object
        that is given after clearing.
    :param out: If file name is given, saves before showing.
    """
    if ax is None:
        if fig_num is None:
            fig = pylab.gcf()
        else:
            fig = pylab.figure(fig_num, figsize=(8, 4))
        plt.clf()
        axes = fig.add_subplot(1, 1, 1)
    else:
        axes = ax
        plt.cla()

    axes.semilogy(fs, "-o", ms=ms, lw=lw)

    if fs.folded:
        axes.set_xlim(0, fs.sample_sizes[0] // 2 + 1)
        axes.set_xlabel("Minor allele frequency")
    else:
        axes.set_xlim(0, fs.sample_sizes[0])
        axes.set_xlabel("Allele frequency")

    axes.set_ylabel("Count")

    if ax is None:
        plt.tight_layout()
        if out is not None:
            plt.savefig(out)
        if show:
            plt.show()


def plot_1d_comp_multinom(
    model,
    data,
    fig_num=None,
    residual="Anscombe",
    plot_masked=False,
    out=None,
    show=True,
    labels=["Model", "Data"],
):
    """
    Multinomial comparison between 1d model and data.

    :param model: 1-dimensional model SFS
    :param data: 1-dimensional data SFS
    :param fig_num: Clear and use figure fig_num for display. If None, an new figure
        window is created.
    :param residual: 'Anscombe' for Anscombe residuals, which are more normally
        distributed for Poisson sampling. 'linear' for the linear
        residuals, which can be less biased.
    :param plot_masked: Additionally plots (in open circles) results for points in the
        model or data that were masked.
    :param out: Output filename to save figure, if given.
    :param show: If True, displays figure. Set to False to supress.
    """
    model = Inference.optimally_scaled_sfs(model, data)

    plot_1d_comp_Poisson(model, data, fig_num, residual, plot_masked, out, show, labels)


def plot_1d_comp_Poisson(
    model,
    data,
    fig_num=None,
    residual="Anscombe",
    plot_masked=False,
    out=None,
    show=True,
    labels=["Model", "Data"],
):
    """
    Poisson comparison between 1d model and data.

    :param model: 1-dimensional model SFS
    :param data: 1-dimensional data SFS
    :param fig_num: Clear and use figure fig_num for display. If None, an new figure
        window is created.
    :param residual: 'Anscombe' for Anscombe residuals, which are more normally
        distributed for Poisson sampling. 'linear' for the linear
        residuals, which can be less biased.
    :param plot_masked: Additionally plots (in open circles) results for points in the
        model or data that were masked.
    :param out: Output filename to save figure, if given.
    :param show: If True, execute pylab.show command to make sure plot displays.
    :param labels: A list of strings of length two, labels for the first and second
        input frequency spectra. Defaults to "Model" and "Data".
    """
    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(8, 8))
    pylab.clf()

    if data.folded and not model.folded:
        model = model.fold()

    masked_model, masked_data = Numerics.intersect_masks(model, data)

    ax = pylab.subplot(2, 1, 1)
    ax.semilogy(masked_data, "-o", ms=6, lw=1, mfc="w", label=labels[1])
    ax.semilogy(masked_model, "-o", ms=3, lw=1, label=labels[0])

    if plot_masked:
        ax.semilogy(
            masked_data.data, "--o", ms=6, lw=1, mfc="w", zorder=-100, label=None
        )
        ax.semilogy(
            masked_model.data, "--o", ms=4, lw=1, mfc="w", zorder=-100, label=None
        )

    ax2 = pylab.subplot(2, 1, 2, sharex=ax)
    if residual == "Anscombe":
        resid = Inference.Anscombe_Poisson_residual(masked_model, masked_data)
    elif residual == "linear":
        resid = Inference.linear_Poisson_residual(masked_model, masked_data)
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)

    ax2.plot([], [])
    ax2.plot([], [])
    ax2.plot(resid, "-o", ms=4, lw=1)
    if plot_masked:
        ax2.plot(resid.data, "--o", ms=4, lw=1, mfc="w", zorder=-100)

    if data.folded:
        ax.set_xlim(0, data.sample_sizes[0] // 2 + 1)
        ax2.set_xlabel("Minor allele frequency")
    else:
        ax.set_xlim(0, data.sample_sizes[0])
        ax2.set_xlabel("Allele frequency")

    ax.set_ylabel("Count")
    ax2.set_ylabel("Residual")
    ax.legend()

    if out is not None:
        f.tight_layout()
        pylab.savefig(out)
    if show:
        pylab.show()


##
## 2-population functions
##


def plot_single_2d_sfs(
    sfs,
    vmin=None,
    vmax=None,
    ax=None,
    pop_ids=None,
    extend="neither",
    colorbar=True,
    cmap=pylab.cm.hsv,
    out=None,
    show=True,
):
    """
    Heatmap of single 2d SFS.

    If vmax is greater than a factor of 10, plot on log scale.

    Returns colorbar that is created.

    :param sfs: SFS to plot
    :param vmin: Values in sfs below vmin are masked in plot.
    :param vmax: Values in sfs above vmax saturate the color spectrum.
    :param ax: Axes object to plot into. If None, the result of pylab.gca() is used.
    :param pop_ids: If not None, override pop_ids stored in Spectrum.
    :param extend: Whether the colorbar should have 'extension' arrows. See
        help(pylab.colorbar) for more details.
    :param colorbar: Should we plot a colorbar?
    :param cmap: Pylab colormap to use for plotting.
    :param out: Output filename to save figure, if given.
    :param show: If True, execute pylab.show command to make sure plot displays.
    """
    if ax is None:
        fig = pylab.gcf()
        pylab.clf()
        axes = pylab.gca()
    else:
        axes = ax

    # this fails with entries of zero, so we mask those:
    sfs_plot = copy.copy(sfs)
    sfs_plot.mask[sfs_plot == 0] = True

    if vmin is None:
        vmin = sfs_plot.min()
    if vmax is None:
        vmax = sfs_plot.max()

    pylab.cm.hsv.set_under("w")
    if vmax / vmin > 10:
        # Under matplotlib 1.0.1, default LogFormatter omits some tick lines.
        # This works more consistently.
        norm = matplotlib.colors.LogNorm(vmin=vmin * (1 - 1e-3), vmax=vmax * (1 + 1e-3))
        format = matplotlib.ticker.LogFormatterMathtext()
    else:
        norm = matplotlib.colors.Normalize(
            vmin=vmin * (1 - 1e-3), vmax=vmax * (1 + 1e-3)
        )
        format = None
    mappable = axes.pcolor(
        np.ma.masked_where(sfs_plot < vmin, sfs), cmap=cmap, edgecolors="none", norm=norm
    )
    cb = axes.figure.colorbar(mappable, extend=extend, format=format)
    if not colorbar:
        axes.figure.delaxes(axes.figure.axes[-1])
    else:
        # A hack so we can manually work around weird ticks in some colorbars
        try:
            axes.figure.moments_colorbars.append(cb)
        except AttributeError:
            axes.figure.moments_colorbars = [cb]

    axes.plot([0, sfs_plot.shape[1]], [0, sfs_plot.shape[0]], "-k", lw=0.2)

    if pop_ids is None:
        if sfs_plot.pop_ids is not None:
            pop_ids = sfs_plot.pop_ids
        else:
            pop_ids = ["pop0", "pop1"]
    axes.set_ylabel(pop_ids[0], verticalalignment="top")
    axes.set_xlabel(pop_ids[1], verticalalignment="bottom")

    axes.xaxis.set_major_formatter(_ctf)
    axes.xaxis.set_major_locator(_sfsTickLocator())
    axes.yaxis.set_major_formatter(_ctf)
    axes.yaxis.set_major_locator(_sfsTickLocator())
    for tick in axes.xaxis.get_ticklines() + axes.yaxis.get_ticklines():
        tick.set_visible(False)

    axes.set_xlim(0, sfs_plot.shape[1])
    axes.set_ylim(0, sfs_plot.shape[0])

    if ax is None:
        plt.tight_layout()
        if out is not None:
            plt.savefig(out)
        if show:
            plt.show()


def plot_2d_resid(
    resid,
    resid_range=None,
    ax=None,
    pop_ids=None,
    extend="neither",
    colorbar=True,
    out=None,
    show=True,
):
    """
    Linear heatmap of 2d residual array.

    :param sfs: Residual array to plot.
    :param resid_range: Values > resid range or < resid_range saturate the color
        spectrum.
    :param ax: Axes object to plot into. If None, the result of pylab.gca() is used.
    :param pop_ids: If not None, override pop_ids stored in Spectrum.
    :param extend: Whether the colorbar should have 'extension' arrows. See
        help(pylab.colorbar) for more details.
    :param colorbar: Should we plot a colorbar?
    :param out: Output filename to save figure, if given.
    :param show: If True, execute pylab.show command to make sure plot displays.
    """
    if ax is None:
        fig = pylab.gcf()
        pylab.clf()
        axes = pylab.gca()
    else:
        axes = ax

    if resid_range is None:
        resid_range = abs(resid).max()

    mappable = axes.pcolor(
        resid,
        cmap=pylab.cm.RdBu_r,
        vmin=-resid_range,
        vmax=resid_range,
        edgecolors="none",
    )

    cbticks = [-resid_range, 0, resid_range]
    format = matplotlib.ticker.FormatStrFormatter("%.2g")
    cb = axes.figure.colorbar(mappable, ticks=cbticks, format=format, extend=extend)
    if not colorbar:
        axes.figure.delaxes(axes.figure.axes[-1])
    else:
        try:
            axes.figure.moments_colorbars.append(cb)
        except AttributeError:
            axes.figure.moments_colorbars = [cb]

    axes.plot([0, resid.shape[1]], [0, resid.shape[0]], "-k", lw=0.2)

    if pop_ids is None:
        if resid.pop_ids is not None:
            pop_ids = resid.pop_ids
        else:
            pop_ids = ["pop0", "pop1"]
    axes.set_ylabel(pop_ids[0], verticalalignment="top")
    axes.set_xlabel(pop_ids[1], verticalalignment="bottom")

    axes.xaxis.set_major_formatter(_ctf)
    axes.xaxis.set_major_locator(_sfsTickLocator())
    axes.yaxis.set_major_formatter(_ctf)
    axes.yaxis.set_major_locator(_sfsTickLocator())
    for tick in axes.xaxis.get_ticklines() + axes.yaxis.get_ticklines():
        tick.set_visible(False)

    axes.set_xlim(0, resid.shape[1])
    axes.set_ylim(0, resid.shape[0])

    if ax is None:
        plt.tight_layout()
        if out is not None:
            plt.savefig(out)
        if show:
            plt.show()


# Used to determine whether colorbars should have 'extended' arrows
_extend_mapping = {
    (True, True): "neither",
    (False, True): "min",
    (True, False): "max",
    (False, False): "both",
}


def plot_2d_comp_multinom(
    model,
    data,
    vmin=None,
    vmax=None,
    resid_range=None,
    fig_num=None,
    pop_ids=None,
    residual="Anscombe",
    adjust=True,
    out=None,
    show=True,
):
    """
    Multinomial comparison between 2d model and data.

    :param model: 2-dimensional model SFS
    :param data: 2-dimensional data SFS
    :param vmin: Minimum value plotted.
    :param vmax: Maximum value plotted.
    :param resid_range: Residual plot saturates at +- resid_range.
    :param fig_num: Clear and use figure fig_num for display. If None, an new figure
        window is created.
    :param pop_ids: If not None, override pop_ids stored in Spectrum.
    :param residual: 'Anscombe' for Anscombe residuals, which are more normally
        distributed for Poisson sampling. 'linear' for the linear
        residuals, which can be less biased.
    :param adjust: Should method use automatic 'subplots_adjust'? For advanced
        manipulation of plots, it may be useful to make this False.
    :param out: Output filename to save figure, if given.
    :param show: If True, execute pylab.show command to make sure plot displays.
    """
    model = Inference.optimally_scaled_sfs(model, data)

    plot_2d_comp_Poisson(
        model,
        data,
        vmin=vmin,
        vmax=vmax,
        resid_range=resid_range,
        fig_num=fig_num,
        pop_ids=pop_ids,
        residual=residual,
        adjust=adjust,
        out=out,
        show=show,
    )


def plot_2d_comp_Poisson(
    model,
    data,
    vmin=None,
    vmax=None,
    resid_range=None,
    fig_num=None,
    pop_ids=None,
    residual="Anscombe",
    adjust=True,
    out=None,
    show=True,
):
    """
    Poisson comparison between 2d model and data.

    :param model: 2-dimensional model SFS
    :param data: 2-dimensional data SFS
    :param vmin: Minimum value plotted.
    :param vmax: Maximum value plotted.
    :param resid_range: Residual plot saturates at +- resid_range.
    :param fig_num: Clear and use figure fig_num for display. If None, an new figure
        window is created.
    :param pop_ids: If not None, override pop_ids stored in Spectrum.
    :param residual: 'Anscombe' for Anscombe residuals, which are more normally
        distributed for Poisson sampling. 'linear' for the linear
        residuals, which can be less biased.
    :param adjust: Should method use automatic 'subplots_adjust'? For advanced
        manipulation of plots, it may be useful to make this False.
    :param out: Output filename to save figure, if given.
    :param show: If True, execute pylab.show command to make sure plot displays.
    """
    data_plot = copy.copy(data)
    model_plot = copy.copy(model)

    if data_plot.folded and not model_plot.folded:
        model_plot = model_plot.fold()

    # errors if there are zero entries in the data or model, mask them:
    model_plot.mask[model_plot == 0] = True
    data_plot.mask[data_plot == 0] = True

    masked_model, masked_data = Numerics.intersect_masks(model_plot, data_plot)

    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(7, 7))

    pylab.clf()
    if adjust:
        pylab.subplots_adjust(
            bottom=0.07, left=0.07, top=0.94, right=0.95, hspace=0.26, wspace=0.26
        )

    max_toplot = max(masked_model.max(), masked_data.max())
    min_toplot = min(masked_model.min(), masked_data.min())
    if vmax is None:
        vmax = max_toplot
    if vmin is None:
        vmin = min_toplot
    extend = _extend_mapping[vmin <= min_toplot, vmax >= max_toplot]

    if pop_ids is not None:
        data_pop_ids = model_pop_ids = resid_pop_ids = pop_ids
        if len(pop_ids) != 2:
            raise ValueError("pop_ids must be of length 2.")
    else:
        data_pop_ids = masked_data.pop_ids
        model_pop_ids = masked_model.pop_ids
        if masked_model.pop_ids is None:
            model_pop_ids = data_pop_ids

        if model_pop_ids == data_pop_ids:
            resid_pop_ids = model_pop_ids
        else:
            resid_pop_ids = None

    ax = pylab.subplot(2, 2, 1)
    plot_single_2d_sfs(
        masked_data,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        pop_ids=data_pop_ids,
        colorbar=False,
        show=False,
    )
    ax.set_title("data")

    ax2 = pylab.subplot(2, 2, 2, sharex=ax, sharey=ax)
    plot_single_2d_sfs(
        masked_model,
        vmin=vmin,
        vmax=vmax,
        ax=ax2,
        pop_ids=model_pop_ids,
        extend=extend,
        show=False,
    )
    ax2.set_title("model")

    if residual == "Anscombe":
        resid = Inference.Anscombe_Poisson_residual(
            masked_model, masked_data, mask=vmin
        )
    elif residual == "linear":
        resid = Inference.linear_Poisson_residual(masked_model, masked_data, mask=vmin)
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)

    if resid_range is None:
        resid_range = max((abs(resid.max()), abs(resid.min())))
    resid_extend = _extend_mapping[
        -resid_range <= resid.min(), resid_range >= resid.max()
    ]

    ax3 = pylab.subplot(2, 2, 3, sharex=ax, sharey=ax)
    plot_2d_resid(
        resid,
        resid_range,
        pop_ids=resid_pop_ids,
        extend=resid_extend,
        ax=ax3,
        show=False,
    )
    ax3.set_title("residuals")

    ax4 = pylab.subplot(2, 2, 4)
    flatresid = np.compress(np.logical_not(resid.mask.ravel()), resid.ravel())
    ax4.hist(flatresid, bins=20, density=True)
    ax4.set_title("residuals")
    ax4.set_yticks([])

    plt.tight_layout()
    if out is not None:
        plt.savefig(out)
    if show:
        plt.show()


##
## 3-population functions
##


def plot_3d_comp_multinom(
    model,
    data,
    vmin=None,
    vmax=None,
    resid_range=None,
    fig_num=None,
    pop_ids=None,
    residual="Anscombe",
    adjust=True,
    out=None,
    show=True,
):
    """
    Multinomial comparison between 3d model and data.

    :param model: 3-dimensional model SFS
    :param data: 3-dimensional data SFS
    :param vmin: Minimum value plotted.
    :param vmax: Maximum value plotted.
    :param resid_range: Residual plot saturates at +- resid_range.
    :param fig_num: Clear and use figure fig_num for display. If None, an new figure
        window is created.
    :param pop_ids: If not None, override pop_ids stored in Spectrum.
    :param residual: 'Anscombe' for Anscombe residuals, which are more normally
        distributed for Poisson sampling. 'linear' for the linear
        residuals, which can be less biased.
    :param adjust: Should method use automatic 'subplots_adjust'? For advanced
        manipulation of plots, it may be useful to make this False.
    :param out: Output filename to save figure, if given.
    :param show: If True, execute pylab.show command to make sure plot displays.
    """
    model = Inference.optimally_scaled_sfs(model, data)

    plot_3d_comp_Poisson(
        model,
        data,
        vmin=vmin,
        vmax=vmax,
        resid_range=resid_range,
        fig_num=fig_num,
        pop_ids=pop_ids,
        residual=residual,
        adjust=adjust,
        out=out,
        show=show,
    )


def plot_3d_comp_Poisson(
    model,
    data,
    vmin=None,
    vmax=None,
    resid_range=None,
    fig_num=None,
    pop_ids=None,
    residual="Anscombe",
    adjust=True,
    out=None,
    show=True,
):
    """
    Poisson comparison between 3d model and data.

    :param model: 3-dimensional model SFS
    :param data: 3-dimensional data SFS
    :param vmin: Minimum value plotted.
    :param vmax: Maximum value plotted.
    :param resid_range: Residual plot saturates at +- resid_range.
    :param fig_num: Clear and use figure fig_num for display. If None, an new figure
        window is created.
    :param pop_ids: If not None, override pop_ids stored in Spectrum.
    :param residual: 'Anscombe' for Anscombe residuals, which are more normally
        distributed for Poisson sampling. 'linear' for the linear
        residuals, which can be less biased.
    :param adjust: Should method use automatic 'subplots_adjust'? For advanced
        manipulation of plots, it may be useful to make this False.
    :param out: Output filename to save figure, if given.
    :param show: If True, execute pylab.show command to make sure plot displays.
    """
    model_plot = copy.copy(model)
    data_plot = copy.copy(data)
    if data_plot.folded and not model_plot.folded:
        model_plot = model_plot.fold()

    # errors if there are zero entries in the data or model, mask them:
    model_plot.mask[model_plot == 0] = True
    data_plot.mask[data_plot == 0] = True

    masked_model, masked_data = Numerics.intersect_masks(model_plot, data_plot)

    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(8, 10))

    pylab.clf()
    if adjust:
        pylab.subplots_adjust(bottom=0.07, left=0.07, top=0.95, right=0.95)

    modelmax = max(masked_model.sum(axis=sax).max() for sax in range(3))
    datamax = max(masked_data.sum(axis=sax).max() for sax in range(3))
    modelmin = min(masked_model.sum(axis=sax).min() for sax in range(3))
    datamin = min(masked_data.sum(axis=sax).min() for sax in range(3))
    max_toplot = max(modelmax, datamax)
    min_toplot = min(modelmin, datamin)

    if vmax is None:
        vmax = max_toplot
    if vmin is None:
        vmin = min_toplot
    extend = _extend_mapping[vmin <= min_toplot, vmax >= max_toplot]

    # Calculate the residuals
    if residual == "Anscombe":
        resids = [
            Inference.Anscombe_Poisson_residual(
                masked_model.sum(axis=2 - sax), masked_data.sum(axis=2 - sax), mask=vmin
            )
            for sax in range(3)
        ]
    elif residual == "linear":
        resids = [
            Inference.linear_Poisson_residual(
                masked_model.sum(axis=2 - sax), masked_data.sum(axis=2 - sax), mask=vmin
            )
            for sax in range(3)
        ]
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)

    min_resid = min([r.min() for r in resids])
    max_resid = max([r.max() for r in resids])
    if resid_range is None:
        resid_range = max((abs(max_resid), abs(min_resid)))
    resid_extend = _extend_mapping[-resid_range <= min_resid, resid_range >= max_resid]

    if pop_ids is not None:
        if len(pop_ids) != 3:
            raise ValueError("pop_ids must be of length 3.")
        data_ids = model_ids = resid_ids = pop_ids
    else:
        data_ids = masked_data.pop_ids
        model_ids = masked_model.pop_ids

        if model_ids is None:
            model_ids = data_ids

        if model_ids == data_ids:
            resid_ids = model_ids
        else:
            resid_ids = None

    for sax in range(3):
        marg_data = masked_data.sum(axis=2 - sax)
        marg_model = masked_model.sum(axis=2 - sax)

        curr_ids = []
        for ids in [data_ids, model_ids, resid_ids]:
            if ids is None:
                ids = ["pop0", "pop1", "pop2"]

            if ids is not None:
                ids = list(ids)
                del ids[2 - sax]

            curr_ids.append(ids)

        ax = pylab.subplot(4, 3, sax + 1)
        plot_colorbar = sax == 2
        plot_single_2d_sfs(
            marg_data,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            pop_ids=curr_ids[0],
            extend=extend,
            colorbar=plot_colorbar,
            show=False,
        )

        ax2 = pylab.subplot(4, 3, sax + 4, sharex=ax, sharey=ax)
        plot_single_2d_sfs(
            marg_model,
            vmin=vmin,
            vmax=vmax,
            ax=ax2,
            pop_ids=curr_ids[1],
            extend=extend,
            colorbar=False,
            show=False,
        )

        resid = resids[sax]
        ax3 = pylab.subplot(4, 3, sax + 7, sharex=ax, sharey=ax)
        plot_2d_resid(
            resid,
            resid_range,
            ax=ax3,
            pop_ids=curr_ids[2],
            extend=resid_extend,
            colorbar=plot_colorbar,
            show=False,
        )

        ax4 = pylab.subplot(4, 3, sax + 10)
        flatresid = np.compress(np.logical_not(resid.mask.ravel()), resid.ravel())
        ax4.hist(flatresid, bins=20, density=True)
        ax4.set_yticks([])

    f.tight_layout()
    if out is not None:
        f.savefig(out)
    if show:
        pylab.show()


def plot_3d_spectrum(
    fs, fignum=None, vmin=None, vmax=None, pop_ids=None, out=None, show=True
):
    """
    Logarithmic heatmap of single 3d FS.

    Note that this method is slow, because it relies on matplotlib's software
    rendering. For faster and better looking plots, use plot_3d_spectrum_mayavi.

    :param fs: FS to plot
    :param vmin: Values in fs below vmin are masked in plot.
    :param vmax: Values in fs above vmax saturate the color spectrum.
    :param fignum: Figure number to plot into. If None, a new figure will be created.
    :param pop_ids: If not None, override pop_ids stored in Spectrum.
    :param out: Output filename to save figure, if given.
    :param show: If True, execute pylab.show command to make sure plot displays.
    """
    import mpl_toolkits.mplot3d as mplot3d

    fig = pylab.figure(fignum)
    ax = mplot3d.Axes3D(fig)

    if vmin is None:
        vmin = fs.min()
    if vmax is None:
        vmax = fs.max()

    # Which entries should I plot?
    toplot = np.logical_not(fs.mask)
    toplot = np.logical_and(toplot, fs.data >= vmin)

    # Figure out the color mapping.
    normalized = (np.log(fs) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
    normalized = np.minimum(normalized, 1)
    colors = pylab.cm.hsv(normalized)

    # We draw by calculating which faces are visible and including each as a
    # polygon.
    polys, polycolors = [], []
    for ii in range(fs.shape[0]):
        for jj in range(fs.shape[1]):
            for kk in range(fs.shape[2]):
                if not toplot[ii, jj, kk]:
                    continue
                if kk < fs.shape[2] - 1 and toplot[ii, jj, kk + 1]:
                    pass
                else:
                    polys.append(
                        [
                            [ii - 0.5, jj + 0.5, kk + 0.5],
                            [ii + 0.5, jj + 0.5, kk + 0.5],
                            [ii + 0.5, jj - 0.5, kk + 0.5],
                            [ii - 0.5, jj - 0.5, kk + 0.5],
                        ]
                    )
                    polycolors.append(colors[ii, jj, kk])
                if kk > 0 and toplot[ii, jj, kk - 1]:
                    pass
                else:
                    polys.append(
                        [
                            [ii - 0.5, jj + 0.5, kk - 0.5],
                            [ii + 0.5, jj + 0.5, kk - 0.5],
                            [ii + 0.5, jj - 0.5, kk - 0.5],
                            [ii - 0.5, jj - 0.5, kk - 0.5],
                        ]
                    )
                    polycolors.append(colors[ii, jj, kk])
                if jj < fs.shape[1] - 1 and toplot[ii, jj + 1, kk]:
                    pass
                else:
                    polys.append(
                        [
                            [ii - 0.5, jj + 0.5, kk + 0.5],
                            [ii + 0.5, jj + 0.5, kk + 0.5],
                            [ii + 0.5, jj + 0.5, kk - 0.5],
                            [ii - 0.5, jj + 0.5, kk - 0.5],
                        ]
                    )
                    polycolors.append(colors[ii, jj, kk])
                if jj > 0 and toplot[ii, jj - 1, kk]:
                    pass
                else:
                    polys.append(
                        [
                            [ii - 0.5, jj - 0.5, kk + 0.5],
                            [ii + 0.5, jj - 0.5, kk + 0.5],
                            [ii + 0.5, jj - 0.5, kk - 0.5],
                            [ii - 0.5, jj - 0.5, kk - 0.5],
                        ]
                    )
                    polycolors.append(colors[ii, jj, kk])
                if ii < fs.shape[0] - 1 and toplot[ii + 1, jj, kk]:
                    pass
                else:
                    polys.append(
                        [
                            [ii + 0.5, jj - 0.5, kk + 0.5],
                            [ii + 0.5, jj + 0.5, kk + 0.5],
                            [ii + 0.5, jj + 0.5, kk - 0.5],
                            [ii + 0.5, jj - 0.5, kk - 0.5],
                        ]
                    )
                    polycolors.append(colors[ii, jj, kk])
                if ii > 0 and toplot[ii - 1, jj, kk]:
                    pass
                else:
                    polys.append(
                        [
                            [ii - 0.5, jj - 0.5, kk + 0.5],
                            [ii - 0.5, jj + 0.5, kk + 0.5],
                            [ii - 0.5, jj + 0.5, kk - 0.5],
                            [ii - 0.5, jj - 0.5, kk - 0.5],
                        ]
                    )
                    polycolors.append(colors[ii, jj, kk])

    polycoll = mplot3d.art3d.Poly3DCollection(
        polys, facecolor=polycolors, edgecolor="k", linewidths=0.5
    )
    ax.add_collection(polycoll)

    # Set the limits
    ax.set_xlim3d(-0.5, fs.shape[0] - 0.5)
    ax.set_ylim3d(-0.5, fs.shape[1] - 0.5)
    ax.set_zlim3d(-0.5, fs.shape[2] - 0.5)

    if pop_ids is None:
        if fs.pop_ids is not None:
            pop_ids = fs.pop_ids
        else:
            pop_ids = ["pop0", "pop1", "pop2"]
    ax.set_xlabel(pop_ids[0], horizontalalignment="left")
    ax.set_ylabel(pop_ids[1], verticalalignment="bottom")
    ax.set_zlabel(pop_ids[2], verticalalignment="bottom")

    # XXX: I can't set the axis ticks to be just the endpoints.

    plt.tight_layout()
    if out is not None:
        plt.savefig(out)
    if show:
        pylab.show()


def plot_4d_comp_multinom(
    model,
    data,
    vmin=None,
    vmax=None,
    resid_range=None,
    fig_num=None,
    pop_ids=None,
    residual="Anscombe",
    adjust=True,
    out=None,
    show=True,
):
    """
    Multinomial comparison between 4d model and data.

    :param model: 4-dimensional model SFS
    :param data: 4-dimensional data SFS
    :param vmin: Minimum value plotted.
    :param vmax: Maximum value plotted.
    :param resid_range: Residual plot saturates at +- resid_range.
    :param fig_num: Clear and use figure fig_num for display. If None, an new figure
        window is created.
    :param pop_ids: If not None, override pop_ids stored in Spectrum.
    :param residual: 'Anscombe' for Anscombe residuals, which are more normally
        distributed for Poisson sampling. 'linear' for the linear
        residuals, which can be less biased.
    :param adjust: Should method use automatic 'subplots_adjust'? For advanced
        manipulation of plots, it may be useful to make this False.
    :param out: Output filename to save figure, if given.
    :param show: If True, execute pylab.show command to make sure plot displays.
    """
    model = Inference.optimally_scaled_sfs(model, data)

    plot_4d_comp_Poisson(
        model,
        data,
        vmin=vmin,
        vmax=vmax,
        resid_range=resid_range,
        fig_num=fig_num,
        pop_ids=pop_ids,
        residual=residual,
        adjust=adjust,
        out=out,
        show=show,
    )


def plot_4d_comp_Poisson(
    model,
    data,
    vmin=None,
    vmax=None,
    resid_range=None,
    fig_num=None,
    pop_ids=None,
    residual="Anscombe",
    adjust=True,
    out=None,
    show=True,
):
    """
    Poisson comparison between 4d model and data.

    :param model: 4-dimensional model SFS
    :param data: 4-dimensional data SFS
    :param vmin: Minimum value plotted.
    :param vmax: Maximum value plotted.
    :param resid_range: Residual plot saturates at +- resid_range.
    :param fig_num: Clear and use figure fig_num for display. If None, an new figure
        window is created.
    :param pop_ids: If not None, override pop_ids stored in Spectrum.
    :param residual: 'Anscombe' for Anscombe residuals, which are more normally
        distributed for Poisson sampling. 'linear' for the linear
        residuals, which can be less biased.
    :param adjust: Should method use automatic 'subplots_adjust'? For advanced
        manipulation of plots, it may be useful to make this False.
    :param out: Output filename to save figure, if given.
    :param show: If True, execute pylab.show command to make sure plot displays.
    """
    model_plot = copy.copy(model)
    data_plot = copy.copy(data)
    if data_plot.folded and not model_plot.folded:
        model_plot = model_plot.fold()

    # errors if there are zero entries in the data or model, mask them:
    model_plot.mask[model_plot == 0] = True
    data_plot.mask[data_plot == 0] = True

    masked_model, masked_data = Numerics.intersect_masks(model_plot, data_plot)

    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(8, 10))

    pylab.clf()
    if adjust:
        pylab.subplots_adjust(bottom=0.07, left=0.07, top=0.95, right=0.95)

    modelmax = max(masked_model.sum(axis=sax).max() for sax in range(4))
    datamax = max(masked_data.sum(axis=sax).max() for sax in range(4))
    modelmin = min(masked_model.sum(axis=sax).min() for sax in range(4))
    datamin = min(masked_data.sum(axis=sax).min() for sax in range(4))
    max_toplot = max(modelmax, datamax)
    min_toplot = min(modelmin, datamin)

    if vmax is None:
        vmax = max_toplot
    if vmin is None:
        vmin = min_toplot
    extend = _extend_mapping[vmin <= min_toplot, vmax >= max_toplot]

    # Calculate the residuals
    list_ind = [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]
    if residual == "Anscombe":
        resids = [
            Inference.Anscombe_Poisson_residual(
                masked_model.sum(axis=int(list_ind[i][1])).sum(
                    axis=int(list_ind[i][0])
                ),
                masked_data.sum(axis=int(list_ind[i][1])).sum(axis=int(list_ind[i][0])),
                mask=vmin,
            )
            for i in range(6)
        ]
    elif residual == "linear":
        resids = [
            Inference.linear_Poisson_residual(
                masked_model.sum(axis=int(list_ind[i][1])).sum(
                    axis=int(list_ind[i][0])
                ),
                masked_data.sum(axis=int(list_ind[i][1])).sum(axis=int(list_ind[i][0])),
                mask=vmin,
            )
            for i in range(6)
        ]
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)

    min_resid = min([r.min() for r in resids])
    max_resid = max([r.max() for r in resids])
    if resid_range is None:
        resid_range = max((abs(max_resid), abs(min_resid)))
    resid_extend = _extend_mapping[-resid_range <= min_resid, resid_range >= max_resid]

    if pop_ids is not None:
        if len(pop_ids) != 4:
            raise ValueError("pop_ids must be of length 4.")
        data_ids = model_ids = resid_ids = pop_ids
    else:
        data_ids = masked_data.pop_ids
        model_ids = masked_model.pop_ids

        if model_ids is None:
            model_ids = data_ids

        if model_ids == data_ids:
            resid_ids = model_ids
        else:
            resid_ids = None
    cptr = 0
    for i in range(4):
        for j in range(i + 1, 4):
            ind = list(range(4))
            ind.remove(j)
            ind.remove(i)
            marg_data = masked_data.sum(axis=int(ind[1])).sum(axis=int(ind[0]))
            marg_model = masked_model.sum(axis=int(ind[1])).sum(axis=int(ind[0]))

            curr_ids = []
            for ids in [data_ids, model_ids, resid_ids]:
                if ids is None:
                    ids = ["pop0", "pop1", "pop2", "pop3"]

                if ids is not None:
                    ids = [ids[j], ids[i]]

                curr_ids.append(ids)

            ax = pylab.subplot(4, 6, cptr + 1)
            plot_colorbar = cptr == 5

            plot_single_2d_sfs(
                marg_data,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                pop_ids=curr_ids[0],
                extend=extend,
                colorbar=plot_colorbar,
                show=False,
            )

            ax2 = pylab.subplot(4, 6, cptr + 7, sharex=ax, sharey=ax)
            plot_single_2d_sfs(
                marg_model,
                vmin=vmin,
                vmax=vmax,
                ax=ax2,
                pop_ids=curr_ids[1],
                extend=extend,
                colorbar=False,
                show=False,
            )

            resid = resids[cptr]
            ax3 = pylab.subplot(4, 6, cptr + 13, sharex=ax, sharey=ax)
            plot_2d_resid(
                resid,
                resid_range,
                ax=ax3,
                pop_ids=curr_ids[2],
                extend=resid_extend,
                colorbar=plot_colorbar,
                show=False,
            )

            ax4 = pylab.subplot(4, 6, cptr + 19)
            flatresid = np.compress(np.logical_not(resid.mask.ravel()), resid.ravel())
            ax4.hist(flatresid, bins=20, density=True)
            ax4.set_yticks([])
            cptr += 1

    f.tight_layout()
    if out is not None:
        f.savefig(out)
    if show:
        pylab.show()
