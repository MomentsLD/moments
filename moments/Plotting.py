"""
Routines for Plotting comparisons between model and data.

These can serve as inspiration for custom routines for one's own purposes.
Note that all the plotting is done with pylab. To see additional pylab methods:
"import pylab; help(pylab)". Pylab's many functions are documented at 
http://matplotlib.sourceforge.net/contents.html
"""

import matplotlib
import pylab
import numpy

#: Custom ticks that label only the lowest and highest bins in an FS plot.
class _sfsTickLocator(matplotlib.ticker.Locator):
    def __call__(self):
        'Return the locations of the ticks'

        try:
            vmin, vmax = self.axis.get_view_interval()
            dmin, dmax = self.axis.get_data_interval()
        except AttributeError:
            self.verify_intervals()
            vmin, vmax = self.viewInterval.get_bounds()
            dmin, dmax = self.dataInterval.get_bounds()

        tmin = max(vmin, dmin)
        tmax = min(vmax, dmax)

        return numpy.array([round(tmin) + 0.5, round(tmax) - 0.5])
#: Custom tick formatter
_ctf = matplotlib.ticker.FuncFormatter(lambda x,pos: '%i' % (x-0.4))


from moments import Numerics, Inference

def plot_1d_fs(fs, fig_num=None, show=True):
    """
    Plot a 1-dimensional frequency spectrum.

    fs: 1-dimensional Spectrum
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    show: If True, execute pylab.show command to make sure plot displays.

    Note that all the plotting is done with pylab. To see additional pylab
    methods: "import pylab; help(pylab)". Pylab's many functions are documented
    at http://matplotlib.sourceforge.net/contents.html
    """

    if fig_num is None:
        fig = pylab.gcf()
    else:
        fig = pylab.figure(fig_num, figsize=(7, 7))
    fig.clear()

    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(fs, '-ob')

    ax.set_xlim(0, fs.sample_sizes[0])
    if show:
        fig.show()

def plot_1d_comp_multinom(model, data, fig_num=None, residual='Anscombe',
                          plot_masked=False):
    """
    Mulitnomial comparison between 1d model and data.


    model: 1-dimensional model SFS
    data: 1-dimensional data SFS
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    plot_masked: Additionally plots (in open circles) results for points in the 
                 model or data that were masked.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    model = Inference.optimally_scaled_sfs(model, data)

    plot_1d_comp_Poisson(model, data, fig_num, residual,
                         plot_masked)

def plot_1d_comp_Poisson(model, data, fig_num=None, residual='Anscombe',
                         plot_masked=False, show=True):
    """
    Poisson comparison between 1d model and data.


    model: 1-dimensional model SFS
    data: 1-dimensional data SFS
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    plot_masked: Additionally plots (in open circles) results for points in the 
                 model or data that were masked.
    show: If True, execute pylab.show command to make sure plot displays.
    """
    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(7, 7))
    pylab.clf()

    if data.folded and not model.folded:
        model = model.fold()

    masked_model, masked_data = Numerics.intersect_masks(model, data)

    ax = pylab.subplot(2, 1, 1)
    pylab.semilogy(masked_data, '-ob')
    pylab.semilogy(masked_model, '-or')

    if plot_masked:
        pylab.semilogy(masked_data.data, '--ob', mfc='w', zorder=-100)
        pylab.semilogy(masked_model.data, '--or', mfc='w', zorder=-100)

    pylab.subplot(2, 1, 2, sharex = ax)
    if residual == 'Anscombe':
        resid = Inference.Anscombe_Poisson_residual(masked_model, masked_data)
    elif residual == 'linear':
        resid = Inference.linear_Poisson_residual(masked_model, masked_data)
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)
    pylab.plot(resid, '-og')
    if plot_masked:
        pylab.plot(resid.data, '--og', mfc='w', zorder=-100)

    ax.set_xlim(0, data.shape[0] - 1)
    if show:
        pylab.show()

def plot_single_2d_sfs(sfs, vmin=None, vmax=None, ax=None, 
                       pop_ids=None, extend='neither', colorbar=True,
                       cmap=pylab.cm.hsv):
    """
    Heatmap of single 2d SFS. 
    
    If vmax is greater than a factor of 10, plot on log scale.

    Returns colorbar that is created.

    sfs: SFS to plot
    vmin: Values in sfs below vmin are masked in plot.
    vmax: Values in sfs above vmax saturate the color spectrum.
    ax: Axes object to plot into. If None, the result of pylab.gca() is used.
    pop_ids: If not None, override pop_ids stored in Spectrum.
    extend: Whether the colorbar should have 'extension' arrows. See
            help(pylab.colorbar) for more details.
    colorbar: Should we plot a colorbar?
    cmap: Pylab colormap to use for plotting.
    """
    if ax is None:
        ax = pylab.gca()

    if vmin is None:
        vmin = sfs.min()
    if vmax is None:
        vmax = sfs.max()

    pylab.cm.hsv.set_under('w')
    if vmax / vmin > 10:
        # Under matplotlib 1.0.1, default LogFormatter omits some tick lines.
        # This works more consistently.
        norm = matplotlib.colors.LogNorm(vmin=vmin * (1-1e-3), vmax=vmax * (1+1e-3))
        format = matplotlib.ticker.LogFormatterMathtext()
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin * (1-1e-3), 
                                           vmax=vmax * (1+1e-3))
        format = None
    mappable=ax.pcolor(numpy.ma.masked_where(sfs < vmin, sfs), 
                       cmap=cmap, edgecolors='none',
                       norm=norm)
    cb = ax.figure.colorbar(mappable, extend=extend, format=format)
    if not colorbar:
        ax.figure.delaxes(ax.figure.axes[-1])
    else:
        # A hack so we can manually work around weird ticks in some colorbars
        try:
            ax.figure.moments_colorbars.append(cb)
        except AttributeError:
            ax.figure.moments_colorbars = [cb]

    ax.plot([0,sfs.shape[1]],[0, sfs.shape[0]], '-k', lw=0.2)

    if pop_ids is None:
        if sfs.pop_ids is not None:
            pop_ids = sfs.pop_ids
        else:
            pop_ids = ['pop0','pop1']
    ax.set_ylabel(pop_ids[0], verticalalignment='top')
    ax.set_xlabel(pop_ids[1], verticalalignment='bottom')

    ax.xaxis.set_major_formatter(_ctf)
    ax.xaxis.set_major_locator(_sfsTickLocator())
    ax.yaxis.set_major_formatter(_ctf)
    ax.yaxis.set_major_locator(_sfsTickLocator())
    for tick in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        tick.set_visible(False)

    ax.set_xlim(0, sfs.shape[1])
    ax.set_ylim(0, sfs.shape[0])

    return cb


def plot_2d_resid(resid, resid_range=None, ax=None, pop_ids=None,
                  extend='neither', colorbar=True):
    """
    Linear heatmap of 2d residual array.

    sfs: Residual array to plot.
    resid_range: Values > resid range or < resid_range saturate the color
                 spectrum.
    ax: Axes object to plot into. If None, the result of pylab.gca() is used.
    pop_ids: If not None, override pop_ids stored in Spectrum.
    extend: Whether the colorbar should have 'extension' arrows. See
            help(pylab.colorbar) for more details.
    colorbar: Should we plot a colorbar?
    """
    if ax is None:
        ax = pylab.gca()

    if resid_range is None:
        resid_range = abs(resid).max()

    mappable=ax.pcolor(resid, cmap=pylab.cm.RdBu_r, vmin=-resid_range, 
                       vmax=resid_range, edgecolors='none')

    cbticks = [-resid_range, 0, resid_range]
    format = matplotlib.ticker.FormatStrFormatter('%.2g')
    cb = ax.figure.colorbar(mappable, ticks=cbticks, format=format,
                            extend=extend)
    if not colorbar:
        ax.figure.delaxes(ax.figure.axes[-1])
    else:
        try:
            ax.figure.moments_colorbars.append(cb)
        except AttributeError:
            ax.figure.moments_colorbars = [cb]

    ax.plot([0, resid.shape[1]],[0, resid.shape[0]], '-k', lw=0.2)

    if pop_ids is None:
        if resid.pop_ids is not None:
            pop_ids = resid.pop_ids
        else:
            pop_ids = ['pop0','pop1']
    ax.set_ylabel(pop_ids[0], verticalalignment='top')
    ax.set_xlabel(pop_ids[1], verticalalignment='bottom')

    ax.xaxis.set_major_formatter(_ctf)
    ax.xaxis.set_major_locator(_sfsTickLocator())
    ax.yaxis.set_major_formatter(_ctf)
    ax.yaxis.set_major_locator(_sfsTickLocator())
    for tick in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        tick.set_visible(False)

    ax.set_xlim(0, resid.shape[1])
    ax.set_ylim(0, resid.shape[0])

# Used to determine whether colorbars should have 'extended' arrows
_extend_mapping = {(True, True): 'neither',
                   (False, True): 'min',
                   (True, False): 'max',
                   (False, False): 'both'}

def plot_2d_comp_multinom(model, data, vmin=None, vmax=None,
                          resid_range=None, fig_num=None,
                          pop_ids=None, residual='Anscombe',
                          adjust=True):
    """
    Mulitnomial comparison between 2d model and data.


    model: 2-dimensional model SFS
    data: 2-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop_ids: If not None, override pop_ids stored in Spectrum.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    adjust: Should method use automatic 'subplots_adjust'? For advanced
            manipulation of plots, it may be useful to make this False.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    model = Inference.optimally_scaled_sfs(model, data)

    plot_2d_comp_Poisson(model, data, vmin=vmin, vmax=vmax,
                         resid_range=resid_range, fig_num=fig_num,
                         pop_ids=pop_ids, residual=residual,
                         adjust=adjust)
    
def plot_2d_comp_Poisson(model, data, vmin=None, vmax=None,
                         resid_range=None, fig_num=None,
                         pop_ids=None, residual='Anscombe',
                         adjust=True, show=True):
    """
    Poisson comparison between 2d model and data.


    model: 2-dimensional model SFS
    data: 2-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop_ids: If not None, override pop_ids stored in Spectrum.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    adjust: Should method use automatic 'subplots_adjust'? For advanced
            manipulation of plots, it may be useful to make this False.
    """
    if data.folded and not model.folded:
        model = model.fold()

    masked_model, masked_data = Numerics.intersect_masks(model, data)

    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(7, 7))

    pylab.clf()
    if adjust:
        pylab.subplots_adjust(bottom=0.07, left=0.07, top=0.94, right=0.95, 
                              hspace=0.26, wspace=0.26)

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
            raise ValueError('pop_ids must be of length 2.')
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
    plot_single_2d_sfs(masked_data, vmin=vmin, vmax=vmax,
                       pop_ids=data_pop_ids, colorbar=False)
    ax.set_title('data')

    ax2 = pylab.subplot(2, 2, 2, sharex=ax, sharey=ax)
    plot_single_2d_sfs(masked_model, vmin=vmin, vmax=vmax,
                       pop_ids=model_pop_ids, extend=extend)
    ax2.set_title('model')

    if residual == 'Anscombe':
        resid = Inference.Anscombe_Poisson_residual(masked_model, masked_data,
                                                    mask=vmin)
    elif residual == 'linear':
        resid = Inference.linear_Poisson_residual(masked_model, masked_data,
                                                  mask=vmin)
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)

    if resid_range is None:
        resid_range = max((abs(resid.max()), abs(resid.min())))
    resid_extend = _extend_mapping[-resid_range <= resid.min(), 
                                   resid_range >= resid.max()]

    ax3 = pylab.subplot(2, 2, 3, sharex=ax, sharey=ax)
    plot_2d_resid(resid, resid_range, pop_ids=resid_pop_ids,
                  extend=resid_extend)
    ax3.set_title('residuals')

    ax = pylab.subplot(2,2,4)
    flatresid = numpy.compress(numpy.logical_not(resid.mask.ravel()), 
                               resid.ravel())
    ax.hist(flatresid, bins=20, normed=True)
    ax.set_title('residuals')
    ax.set_yticks([])
    if show:
        pylab.show()

def plot_3d_comp_multinom(model, data, vmin=None, vmax=None,
                          resid_range=None, fig_num=None,
                          pop_ids=None, residual='Anscombe', adjust=True):
    """
    Multinomial comparison between 3d model and data.


    model: 3-dimensional model SFS
    data: 3-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop_ids: If not None, override pop_ids stored in Spectrum.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    adjust: Should method use automatic 'subplots_adjust'? For advanced
            manipulation of plots, it may be useful to make this False.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    model = Inference.optimally_scaled_sfs(model, data)

    plot_3d_comp_Poisson(model, data, vmin=vmin, vmax=vmax,
                         resid_range=resid_range, fig_num=fig_num,
                         pop_ids=pop_ids, residual=residual,
                         adjust=adjust)

def plot_3d_comp_Poisson(model, data, vmin=None, vmax=None,
                         resid_range=None, fig_num=None, pop_ids=None, 
                         residual='Anscombe', adjust=True, show=True):
    """
    Poisson comparison between 3d model and data.


    model: 3-dimensional model SFS
    data: 3-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop_ids: If not None, override pop_ids stored in Spectrum.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    adjust: Should method use automatic 'subplots_adjust'? For advanced
            manipulation of plots, it may be useful to make this False.
    show: If True, execute pylab.show command to make sure plot displays.
    """
    if data.folded and not model.folded:
        model = model.fold()

    masked_model, masked_data = Numerics.intersect_masks(model, data)

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
    if residual == 'Anscombe':
        resids = [Inference.\
                  Anscombe_Poisson_residual(masked_model.sum(axis=2 - sax), 
                                            masked_data.sum(axis=2 - sax), 
                                            mask=vmin) for sax in range(3)]
    elif residual == 'linear':
        resids =[Inference.\
                 linear_Poisson_residual(masked_model.sum(axis=2 - sax), 
                                         masked_data.sum(axis=2 - sax), 
                                         mask=vmin) for sax in range(3)]
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)


    min_resid = min([r.min() for r in resids])
    max_resid = max([r.max() for r in resids])
    if resid_range is None:
        resid_range = max((abs(max_resid), abs(min_resid)))
    resid_extend = _extend_mapping[-resid_range <= min_resid, 
                                   resid_range >= max_resid]

    if pop_ids is not None:
        if len(pop_ids) != 3:
            raise ValueError('pop_ids must be of length 3.')
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
                ids = ['pop0', 'pop1', 'pop2']

            if ids is not None:
                ids = list(ids)
                del ids[2 - sax]

            curr_ids.append(ids)

        ax = pylab.subplot(4, 3, sax + 1)
        plot_colorbar = (sax == 2)
        plot_single_2d_sfs(marg_data, vmin=vmin, vmax=vmax, pop_ids=curr_ids[0],
                           extend=extend, colorbar=plot_colorbar)

        pylab.subplot(4, 3, sax + 4, sharex=ax, sharey=ax)
        plot_single_2d_sfs(marg_model, vmin=vmin, vmax=vmax, 
                           pop_ids=curr_ids[1], extend=extend, colorbar=False)

        resid = resids[sax]
        pylab.subplot(4, 3, sax + 7, sharex=ax, sharey=ax)
        plot_2d_resid(resid, resid_range, pop_ids=curr_ids[2],
                      extend=resid_extend, colorbar=plot_colorbar)

        ax = pylab.subplot(4, 3, sax + 10)
        flatresid = numpy.compress(numpy.logical_not(resid.mask.ravel()), 
                                   resid.ravel())
        ax.hist(flatresid, bins=20, normed=True)
        ax.set_yticks([])
    if show:
        pylab.show()

def plot_3d_spectrum(fs, fignum=None, vmin=None, vmax=None, pop_ids=None,
                     show=True):
    """
    Logarithmic heatmap of single 3d FS.

    Note that this method is slow, because it relies on matplotlib's software
    rendering. For faster and better looking plots, use plot_3d_spectrum_mayavi.

    fs: FS to plot
    vmin: Values in fs below vmin are masked in plot.
    vmax: Values in fs above vmax saturate the color spectrum.
    fignum: Figure number to plot into. If None, a new figure will be created.
    pop_ids: If not None, override pop_ids stored in Spectrum.
    show: If True, execute pylab.show command to make sure plot displays.
    """
    import mpl_toolkits.mplot3d as mplot3d
    
    fig = pylab.figure(fignum)
    ax = mplot3d.Axes3D(fig)

    if vmin is None:
        vmin = fs.min()
    if vmax is None:
        vmax = fs.max()

    # Which entries should I plot?
    toplot = numpy.logical_not(fs.mask)
    toplot = numpy.logical_and(toplot, fs.data >= vmin)
    
    # Figure out the color mapping.
    normalized = (numpy.log(fs)-numpy.log(vmin))\
            /(numpy.log(vmax)-numpy.log(vmin))
    normalized = numpy.minimum(normalized, 1)
    colors = pylab.cm.hsv(normalized)
    
    # We draw by calculating which faces are visible and including each as a
    # polygon.
    polys, polycolors = [],[]
    for ii in range(fs.shape[0]):
        for jj in range(fs.shape[1]):
            for kk in range(fs.shape[2]):
                if not toplot[ii, jj, kk]:
                    continue
                if kk < fs.shape[2] - 1 and toplot[ii, jj, kk + 1]:
                    pass
                else:
                    polys.append([[ii - 0.5, jj + 0.5, kk + 0.5], [ii + 0.5, jj + 0.5, kk + 0.5],
                                  [ii + 0.5, jj - 0.5, kk + 0.5], [ii - 0.5, jj - 0.5, kk + 0.5]]
                                 )
                    polycolors.append(colors[ii, jj, kk])
                if kk > 0 and toplot[ii, jj, kk - 1]:
                    pass
                else:
                    polys.append([[ii - 0.5, jj + 0.5, kk - 0.5], [ii + 0.5, jj + 0.5, kk - 0.5],
                                  [ii + 0.5, jj - 0.5, kk - 0.5], [ii - 0.5, jj - 0.5, kk - 0.5]]
                                 )
                    polycolors.append(colors[ii, jj, kk])
                if jj < fs.shape[1] - 1 and toplot[ii, jj + 1, kk]:
                    pass
                else:
                    polys.append([[ii - 0.5, jj + 0.5, kk + 0.5], [ii + 0.5, jj + 0.5, kk + 0.5],
                                  [ii + 0.5, jj + 0.5, kk - 0.5], [ii - 0.5, jj + 0.5, kk - 0.5]]
                                 )
                    polycolors.append(colors[ii, jj, kk])
                if jj > 0 and toplot[ii, jj - 1, kk]:
                    pass
                else:
                    polys.append([[ii - 0.5, jj - 0.5, kk + 0.5], [ii + 0.5, jj - 0.5, kk + 0.5],
                                  [ii + 0.5, jj - 0.5, kk - 0.5], [ii - 0.5, jj - 0.5, kk - 0.5]]
                                 )
                    polycolors.append(colors[ii, jj, kk])
                if ii < fs.shape[0] - 1 and toplot[ii + 1, jj, kk]:
                    pass
                else:
                    polys.append([[ii + 0.5, jj - 0.5, kk + 0.5], [ii + 0.5, jj + 0.5, kk + 0.5],
                                  [ii + 0.5, jj + 0.5, kk - 0.5], [ii + 0.5, jj - 0.5, kk - 0.5]]
                                 )
                    polycolors.append(colors[ii, jj, kk])
                if ii > 0 and toplot[ii - 1, jj, kk]:
                    pass
                else:
                    polys.append([[ii - 0.5, jj - 0.5, kk + 0.5], [ii - 0.5, jj + 0.5, kk + 0.5],
                                  [ii - 0.5, jj + 0.5, kk - 0.5], [ii - 0.5, jj - 0.5, kk - 0.5]]
                                 )
                    polycolors.append(colors[ii, jj, kk])
                    

    polycoll = mplot3d.art3d.Poly3DCollection(polys, facecolor=polycolors, 
                                              edgecolor='k', linewidths=0.5)
    ax.add_collection(polycoll)

    # Set the limits
    ax.set_xlim3d(-0.5, fs.shape[0] - 0.5)
    ax.set_ylim3d(-0.5, fs.shape[1] - 0.5)
    ax.set_zlim3d(-0.5, fs.shape[2] - 0.5)

    if pop_ids is None:
        if fs.pop_ids is not None:
            pop_ids = fs.pop_ids
        else:
            pop_ids = ['pop0', 'pop1', 'pop2']
    ax.set_xlabel(pop_ids[0], horizontalalignment='left')
    ax.set_ylabel(pop_ids[1], verticalalignment='bottom')
    ax.set_zlabel(pop_ids[2], verticalalignment='bottom')

    # XXX: I can't set the axis ticks to be just the endpoints.

    if show:
        pylab.show()

def plot_3d_spectrum_mayavi(fs, fignum=None, vmin=None, vmax=None, 
                            pop_ids=None, show=True):
    """
    Logarithmic heatmap of single 3d FS.

    This method relies on MayaVi2's mlab interface. See http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/mlab.html . To edit plot
    properties, click leftmost icon in the toolbar.

    If you get an ImportError upon calling this function, it is likely that you
    don't have mayavi installed.

    fs: FS to plot
    vmin: Values in fs below vmin are masked in plot.
    vmax: Values in fs above vmax saturate the color spectrum.
    fignum: Figure number to plot into. If None, a new figure will be created.
            Note that these are MayaVi figures, which are separate from
            matplotlib figures.
    pop_ids: If not None, override pop_ids stored in Spectrum.
    show: If True, execute mlab.show command to make sure plot displays.
    """
    from enthought.mayavi import mlab

    fig = mlab.figure(fignum, bgcolor=(1, 1, 1))
    mlab.clf(fig)

    if vmin is None:
        vmin = fs.min()
    if vmax is None:
        vmax = fs.max()

    # Which entries should I plot?
    toplot = numpy.logical_not(fs.mask)
    toplot = numpy.logical_and(toplot, fs.data >= vmin)

    # For the color mapping
    normalized = (numpy.log(fs)-numpy.log(vmin))\
            /(numpy.log(vmax)-numpy.log(vmin))
    normalized = numpy.minimum(normalized, 1)

    xs, ys, zs = numpy.indices(fs.shape)
    flat_xs = xs.flatten()
    flat_ys = ys.flatten()
    flat_zs = zs.flatten()
    flat_toplot = toplot.flatten()
    
    mlab.barchart(flat_xs[flat_toplot], flat_ys[flat_toplot], 
                  flat_zs[flat_toplot], normalized.flatten()[flat_toplot], 
                  colormap='hsv', scale_mode='none', lateral_scale=1, 
                  figure=fig)

    if pop_ids is None:
        if fs.pop_ids is not None:
            pop_ids = fs.pop_ids
        else:
            pop_ids = ['pop0', 'pop1', 'pop2']

    a = mlab.axes(xlabel=pop_ids[0], ylabel=pop_ids[1], zlabel=pop_ids[2], 
                  figure=fig, color=(0, 0, 0))
    a.axes.label_format = ""
    a.title_text_property.color = (0, 0, 0)
    mlab.text3d(fs.sample_sizes[0], fs.sample_sizes[1], fs.sample_sizes[2] + 1, 
                '(%i, %i, %i)'%tuple(fs.sample_sizes), scale=0.75, figure=fig,
                color=(0, 0, 0))
    mlab.view(azimuth=-40, elevation=65, distance='auto', focalpoint='auto')

    if show:
        mlab.show()


def plot_4d_comp_multinom(model, data, vmin=None, vmax=None,
                          resid_range=None, fig_num=None,
                          pop_ids=None, residual='Anscombe', adjust=True):
    """
    Multinomial comparison between 3d model and data.
    
    model: 4-dimensional model SFS
    data: 4-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
        vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
        window is created.
    pop_ids: If not None, override pop_ids stored in Spectrum.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
        distributed for Poisson sampling. 'linear' for the linear
        residuals, which can be less biased.
    adjust: Should method use automatic 'subplots_adjust'? For advanced
        manipulation of plots, it may be useful to make this False.
        
    This comparison is multinomial in that it rescales the model to optimally
        fit the data.
    """
    model = Inference.optimally_scaled_sfs(model, data)
    
    plot_4d_comp_Poisson(model, data, vmin=vmin, vmax=vmax,
                         resid_range=resid_range, fig_num=fig_num,
                         pop_ids=pop_ids, residual=residual,
                         adjust=adjust)

def plot_4d_comp_Poisson(model, data, vmin=None, vmax=None,
                         resid_range=None, fig_num=None, pop_ids=None,
                         residual='Anscombe', adjust=True, show=True):
    """
    Poisson comparison between 4d model and data.
    
    model: 4-dimensional model SFS
    data: 4-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
        vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
        window is created.
    pop_ids: If not None, override pop_ids stored in Spectrum.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
        distributed for Poisson sampling. 'linear' for the linear
        residuals, which can be less biased.
    adjust: Should method use automatic 'subplots_adjust'? For advanced
        manipulation of plots, it may be useful to make this False.
    show: If True, execute pylab.show command to make sure plot displays.
    """
    if data.folded and not model.folded:
        model = model.fold()

    masked_model, masked_data = Numerics.intersect_masks(model, data)

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
    if residual == 'Anscombe':
        resids = [Inference.\
                  Anscombe_Poisson_residual(masked_model.sum(axis=int(list_ind[i][1])).sum(axis=int(list_ind[i][0])),
                                            masked_data.sum(axis=int(list_ind[i][1])).sum(axis=int(list_ind[i][0])),
                                            mask=vmin) for i in range(6)]
    elif residual == 'linear':
        resids =[Inference.\
                 linear_Poisson_residual(masked_model.sum(axis=int(list_ind[i][1])).sum(axis=int(list_ind[i][0])),
                                         masked_data.sum(axis=int(list_ind[i][1])).sum(axis=int(list_ind[i][0])),
                                         mask=vmin) for i in range(6)]
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)


    min_resid = min([r.min() for r in resids])
    max_resid = max([r.max() for r in resids])
    if resid_range is None:
        resid_range = max((abs(max_resid), abs(min_resid)))
    resid_extend = _extend_mapping[-resid_range <= min_resid,
                               resid_range >= max_resid]
    
    if pop_ids is not None:
        if len(pop_ids) != 4:
            raise ValueError('pop_ids must be of length 4.')
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
                    ids = ['pop0', 'pop1', 'pop2', 'pop3']
            
                if ids is not None:
                    ids = [ids[j], ids[i]]
            
                curr_ids.append(ids)
    
            ax = pylab.subplot(4, 6, cptr + 1)
            plot_colorbar = (cptr == 5)
            
            plot_single_2d_sfs(marg_data, vmin=vmin, vmax=vmax, pop_ids=curr_ids[0],
                               extend=extend, colorbar=plot_colorbar)
            
            pylab.subplot(4, 6, cptr + 7, sharex=ax, sharey=ax)
            plot_single_2d_sfs(marg_model, vmin=vmin, vmax=vmax,
                               pop_ids=curr_ids[1], extend=extend, colorbar=False)
                           
            resid = resids[cptr]
            pylab.subplot(4, 6, cptr + 13, sharex=ax, sharey=ax)
            plot_2d_resid(resid, resid_range, pop_ids=curr_ids[2],
                          extend=resid_extend, colorbar=plot_colorbar)
                                       
            ax = pylab.subplot(4, 6, cptr + 19)
            flatresid = numpy.compress(numpy.logical_not(resid.mask.ravel()),
                                       resid.ravel())
            ax.hist(flatresid, bins=20, normed=True)
            ax.set_yticks([])
            cptr += 1

    if show:
        pylab.show()
