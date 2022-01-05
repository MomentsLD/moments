"""
This module provides functionality to gather and store information about a
demographic model (as defined by a demographic model function), and to use this
information to generate a visual representation of the model.

Just two functions are used for generating and plotting models. For example, the
following is all that is necessary to generate a plot of the model defined by
demographic_model_func(params, ns):

model = ModelPlot.generate_model(demographic_model_func, params, ns)
ModelPlot.plot_model(model)

Additional options for customizing the model are described in the documentation
of these two functions. The module is currently compatible with the following 
methods in moments:

LinearSystem_1D.steady_state_1D
LinearSystem.steady_state
Manips.split_{1D_to_2D,2D_to_3D_2,2D_to_3D_1,3D_to_4D_3,4D_to_5D_4}
Spectrum_mod.Spectrum.integrate
"""
try:
    import matplotlib.colors as mcolors
    import matplotlib.offsetbox as mbox
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _imported_mpl = True
except ImportError:
    _imported_mpl = False


def _check_mpl_imported():
    if not _imported_mpl:
        raise ValueError(
            "matplotlib not found - moments can be used but "
            "plotting features will not work"
        )


import numpy as np

## USER FUNCTIONS ##
def generate_model(model_func, params, ns, precision=100):
    """
    Generates information about a demographic model, and stores the information
    in a format that can be used by the plot_model function

    model_func : Function of the form model_func(params, ns). Describes the
                 demographic model to collect information on.

    params : List of values of the demographic variables to be passed as the
             params argument to model_func.

    ns : List of sample sizes to be passed as the ns argument to model_func.
         Actual values do not matter, as long as the dimensionality is correct.

    precision : Number of times to evaluate population sizes per period. This
                value can be increased if any of the plotted populations do
                not appear smooth.

    Returns a _ModelInfo object storing the information.
    """
    _check_mpl_imported()
    # Initialize model and collect necessary information from model function
    model = _ModelInfo(precision)
    model_func(params, ns)
    # Closing model prevents continued collection of information
    _close_model()
    # Determine size of population trees for space allocation in plot
    model.determine_framesizes()
    # Determine location and orientation of each plotted population
    vstart = 0
    tp1 = model.tp_list[0]
    for pop_index in range(len(tp1.popsizes)):
        model.determine_drawinfo(0, pop_index, (0, vstart), 1)
        vstart += tp1.framesizes[pop_index]
    return model


def plot_model(
    model,
    save_file=None,
    ax=None,
    show=False,
    fig_title="Demographic Model",
    pop_labels=None,
    nref=None,
    draw_ancestors=True,
    draw_migrations=True,
    draw_scale=True,
    scale_bar=False,
    arrow_size=0.01,
    transition_size=0.05,
    gen_time=0,
    gen_time_units="Years",
    reverse_timeline=False,
    fig_bg_color="#ffffff",
    plot_bg_color="#ffffff",
    text_color="#002b36",
    gridline_color="#586e75",
    pop_color="#268bd2",
    arrow_color="#073642",
    label_size=16,
    tick_size=12,
    grid=True,
):
    """
    Plots a demographic model based on information contained within a _ModelInfo
    object. See the matplotlib docs for valid entries for the color parameters.

    model : A _ModelInfo object created using generate_model().

    save_file : If not None, the figure will be saved to this location. Otherwise
                the figure will be displayed to the screen.

    fig_title : Title of the figure.

    pop_labels : If not None, should be a list of strings of the same length as
                 the total number of final populations in the model. The string
                 at index i should be the name of the population along axis i in
                 the model's SFS.

    nref : If specified, this will update the time and population size labels to
           use units based on an ancestral population size of nref. See the
           documentation for details.

    draw_ancestors : Specify whether the ancestral populations should be drawn
                     in beginning of plot. Will fade off with a gradient.

    draw_migrations : Specify whether migration arrows are drawn.

    draw_scale : Specify whether scale bar should be shown in top-left corner.

    scale_bar : If True, draw the scale bar. If False (default), only draw
                arrows for scale.

    arrow_size : Float to control the size of the migration arrows.

    transition_size : Float specifying size of the "transitional periods"
                      between populations.

    gen_time : If greater than 0, and nref given, timeline will be adjusted to
               show absolute time values, using this value as the time elapsed
               per generation.

    gen_time_units : Units used for gen_time (e.g. Years, Thousand Years, etc.).

    reverse_timeline : If True, the labels on the timeline will be reversed, so
                       that "0 time" is the present time period, rather than the
                       time of the original population.

    fig_bg_color : Background color of figure (i.e. border surrounding the
                   drawn model).

    plot_bg_color : Background color of the actual plot area.

    text_color : Color of text in the figure.

    gridline_color : Color of the plot gridlines.

    pop_color : Color of the populations.

    arrow_color : Color of the arrows showing migrations between populations.
    """
    _check_mpl_imported()
    # Set up the plot with a title and axis labels
    fig_kwargs = {
        "figsize": (9.6, 5.4),
        "dpi": 200,
        "facecolor": fig_bg_color,
        "edgecolor": fig_bg_color,
    }
    if ax == None:
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111)
    ax.set_facecolor(plot_bg_color)
    ax.set_title(fig_title, color=text_color, fontsize=24)
    xlabel = "Time Ago" if reverse_timeline else "Time"
    if nref:
        if gen_time > 0:
            xlabel += " ({})".format(gen_time_units)
        else:
            xlabel += " (Generations)"
        ylabel = "Population Sizes"
    else:
        xlabel += " (Genetic Units)"
        ylabel = "Relative Population Sizes"
    ax.set_xlabel(xlabel, color=text_color, fontsize=label_size)
    ax.set_ylabel(ylabel, color=text_color, fontsize=label_size)

    # Determine various maximum values for proper scaling within the plot
    xmax = model.tp_list[-1].time[-1]
    ymax = sum(model.tp_list[0].framesizes)
    ax.set_xlim([-1 * xmax * 0.1, xmax])
    ax.set_ylim([0, ymax])
    mig_max = 0
    for tp in model.tp_list:
        if tp.migrations is None:
            continue
        mig = np.amax(tp.migrations)
        mig_max = mig_max if mig_max > mig else mig

    # Configure axis border colors
    ax.spines["top"].set_color(text_color)
    ax.spines["right"].set_color(text_color)
    ax.spines["bottom"].set_color(text_color)
    ax.spines["left"].set_color(text_color)

    # Major ticks along x-axis (time) placed at each population split
    xticks = [tp.time[0] for tp in model.tp_list]
    xticks.append(xmax)
    ax.xaxis.set_major_locator(mticker.FixedLocator(xticks))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(
        which="both", axis="x", labelcolor=text_color, labelsize=tick_size, top=False
    )
    # Choose correct time labels based on nref, gen_time, and reverse_timeline
    if reverse_timeline:
        xticks = [xmax - x for x in xticks]
    if nref:
        if gen_time > 0:
            xticks = [2 * nref * gen_time * x for x in xticks]
        else:
            xticks = [2 * nref * x for x in xticks]
        ax.set_xticklabels(["{:.0f}".format(x) for x in xticks])
    else:
        ax.set_xticklabels(["{:.2f}".format(x) for x in xticks])

    # Gridlines along y-axis (population size) spaced by nref size
    if grid:
        ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(ymax)))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(b=True, which="major", axis="y", color=gridline_color)
        ax.tick_params(which="both", axis="y", colors="none", labelsize=tick_size)
    else:
        ax.set_yticks([])

    # Add scale in top-left corner displaying ancestral population size (Nref)
    if draw_scale:
        # Bidirectional arrow of height Nref
        arrow = mbox.AuxTransformBox(ax.transData)
        awidth = xmax * arrow_size * 0.2
        alength = ymax * arrow_size
        arrow_kwargs = {
            "width": awidth,
            "head_width": awidth * 3,
            "head_length": alength,
            "color": text_color,
            "length_includes_head": True,
        }
        arrow.add_artist(plt.arrow(0, 0.25, 0, 0.75, zorder=100, **arrow_kwargs))
        arrow.add_artist(plt.arrow(0, 0.75, 0, -0.75, zorder=100, **arrow_kwargs))
        # Population bar of height Nref
        bar = mbox.AuxTransformBox(ax.transData)
        bar.add_artist(mpatches.Rectangle((0, 0), xmax / ymax, 1, color=pop_color))
        # Appropriate label depending on scale
        label = mbox.TextArea(str(nref) if nref else "Nref")
        label.get_children()[0].set_color(text_color)
        if scale_bar:
            children = [label, arrow, bar]
        else:
            children = [label, arrow]
        bars = mbox.HPacker(children=children, pad=0, sep=2, align="center")
        scalebar = mbox.AnchoredOffsetbox(
            2, pad=0.25, borderpad=0.25, child=bars, frameon=False
        )
        ax.add_artist(scalebar)

    # Add ancestral populations using a gradient fill.
    if draw_ancestors:
        time = -1 * xmax * 0.1
        for i, ori in enumerate(model.tp_list[0].origins):
            # Draw ancestor for each initial pop
            xlist = np.linspace(time, 0.0, model.precision)
            dx = xlist[1] - xlist[0]
            low, mid, top = (
                ori[1],
                ori[1] + 1.0,
                ori[1] + model.tp_list[0].popsizes[i][0],
            )
            tsize = int(transition_size * model.precision)
            y1list = np.array([low] * model.precision)
            y2list = np.array([mid] * (model.precision - tsize))
            y2list = np.append(y2list, np.linspace(mid, top, tsize))
            # Custom color map runs from bg color to pop color
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "custom_map", [plot_bg_color, pop_color]
            )
            colors = np.array(cmap(np.linspace(0.0, 1.0, model.precision - tsize)))
            # Gradient created by drawing multiple small rectangles
            for x, y1, y2, color in zip(
                xlist[: -1 * tsize], y1list[: -1 * tsize], y2list[: -1 * tsize], colors
            ):
                rect = mpatches.Rectangle((x, y1), dx, y2 - y1, color=color)
                ax.add_patch(rect)
            ax.fill_between(
                xlist[-1 * tsize :],
                y1list[-1 * tsize :],
                y2list[-1 * tsize :],
                color=pop_color,
                edgecolor=pop_color,
            )

    # Iterate through time periods and populations to draw everything
    for tp_index, tp in enumerate(model.tp_list):
        # Keep track of migrations to evenly space arrows across time period
        total_migrations = np.count_nonzero(tp.migrations)
        num_migrations = 0

        for pop_index in range(len(tp.popsizes)):
            # Draw current population
            origin = tp.origins[pop_index]
            popsize = tp.popsizes[pop_index]
            direc = tp.direcs[pop_index]
            y1 = origin[1]
            y2 = origin[1] + (direc * popsize)
            ax.fill_between(tp.time, y1, y2, color=pop_color, edgecolor=pop_color)

            # Draw connections to next populations if necessary
            if tp.descendants is not None and tp.descendants[pop_index] != -1:
                desc = tp.descendants[pop_index]
                tp_next = model.tp_list[tp_index + 1]
                # Split population case
                if isinstance(desc, tuple):
                    # Get origins
                    connect_below = tp_next.origins[desc[0]][1]
                    connect_above = tp_next.origins[desc[1]][1]
                    # Get popsizes
                    subpop_below = tp_next.popsizes[desc[0]][0]
                    subpop_above = tp_next.popsizes[desc[1]][0]
                    # Determine correct connection location
                    connect_below -= direc * subpop_below
                    connect_above += direc * subpop_above
                # Single population case
                else:
                    connect_below = tp_next.origins[desc][1]
                    subpop = tp_next.popsizes[desc][0]
                    connect_above = connect_below + direc * subpop
                # Draw the connections
                tsize = int(transition_size * model.precision)
                cx = tp.time[-1 * tsize :]
                cy_below_1 = [origin[1]] * tsize
                cy_above_1 = origin[1] + direc * popsize[-1 * tsize :]
                cy_below_2 = np.linspace(cy_below_1[0], connect_below, tsize)
                cy_above_2 = np.linspace(cy_above_1[0], connect_above, tsize)
                ax.fill_between(
                    cx, cy_below_1, cy_below_2, color=pop_color, edgecolor=pop_color
                )
                ax.fill_between(
                    cx, cy_above_1, cy_above_2, color=pop_color, edgecolor=pop_color
                )

            # Draw migrations if necessary
            if draw_migrations and tp.migrations is not None:
                # Iterate through migrations for current population
                for mig_index, mig_val in enumerate(tp.migrations[pop_index]):
                    # If no migration, continue
                    if mig_val == 0:
                        continue
                    # Calculate proper offset for arrow within this period
                    num_migrations += 1
                    offset = int(
                        tp.precision * num_migrations / (total_migrations + 1.0)
                    )
                    x = tp.time[offset]
                    dx = 0
                    # Determine which sides of populations are closest
                    y1 = origin[1]
                    y2 = y1 + direc * popsize[offset]
                    mig_y1 = tp.origins[mig_index][1]
                    mig_y2 = mig_y1 + (
                        tp.direcs[mig_index] * tp.popsizes[mig_index][offset]
                    )
                    y = y1 if abs(mig_y1 - y1) < abs(mig_y1 - y2) else y2
                    dy = mig_y1 - y if abs(mig_y1 - y) < abs(mig_y2 - y) else mig_y2 - y
                    # Scale arrow to proper size
                    mig_scale = max(0.1, mig_val / mig_max)
                    awidth = xmax * arrow_size * mig_scale
                    alength = ymax * arrow_size
                    ax.arrow(
                        x,
                        y,
                        dx,
                        dy,
                        width=awidth,
                        head_width=awidth * 3,
                        head_length=alength,
                        color=arrow_color,
                        length_includes_head=True,
                    )

    # Label populations if proper labels are given
    tp_last = model.tp_list[-1]
    if pop_labels and len(pop_labels) == len(tp_last.popsizes):
        ax2 = ax.twinx()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())
        # Determine placement of ticks
        yticks = [
            tp_last.origins[i][1] + 0.5 * tp_last.direcs[i] * tp_last.popsizes[i][-1]
            for i in range(len(tp_last.popsizes))
        ]
        ax2.yaxis.set_major_locator(mticker.FixedLocator(yticks))
        ax2.set_yticklabels(pop_labels)
        ax2.tick_params(
            which="both",
            color="none",
            labelcolor=text_color,
            labelsize=label_size,
            left=False,
            top=False,
            right=False,
        )
        ax2.spines["top"].set_color(text_color)
        ax2.spines["left"].set_color(text_color)
        ax2.spines["right"].set_color(text_color)
        ax2.spines["bottom"].set_color(text_color)

    # Display figure
    if save_file:
        plt.savefig(save_file, **fig_kwargs)
    else:
        if show == True:
            plt.show()
    if ax == None:
        plt.close(fig)


## IMPLEMENTATION FUNCTIONS ##
_current_model = None


def _get_model():
    """
    Used by the various methods in moments to determine the correct _ModelInfo
    object to send information to (if any).
    """
    global _current_model
    return _current_model


def _close_model():
    """
    Sets _current_model to None to ensure that ModelPlot does not waste
    resources by collecting information when it is not being used. Also
    ends model by setting descendants of final populations to None.
    """
    global _current_model
    _current_model.tp_list[-1].descendants = None
    _current_model = None


class _ModelInfo:
    """
    Uses information sent by method calls in moments to generate a demographic
    model. Model information is stored as a list of TimePeriod objects, each of
    which stores demographic information (population sizes, splits, migration
    rates) for a given time step in the model.

    current_time : Float, to keep track of current point in time of model.

    tp_list : List of TimePeriod objects of this model.

    precision : Number of times population sizes are evaluated in each period.
    """

    def __init__(self, precision):
        """
        Sets itself as the global current model, to be able to collect data from
        other methods in moments, and initializes instance variables.

        precision : Sets the precision variable.
        """
        global _current_model
        _current_model = self
        self.current_time = 0.0
        self.tp_list = []
        self.precision = precision

    def initialize(self, npops):
        """
        Creates initial steady-state population(s) at time = 0.0.

        npops : Number of original populations.
        """
        self.current_time = 0.0
        tp = self.TimePeriod(self.current_time, npops, self.precision)
        self.tp_list = [tp]

    def split(self, initial_pop, split_pops):
        """
        Sets the appropriate descendants for the current time period and
        correctly splits one of them.

        initial_pop : index of population to split.

        split_pops : tuple of the two indices of the resulting populations.
        """
        tp = self.tp_list[-1]
        count = 0
        for i in range(len(tp.descendants)):
            if tp.descendants[i] == -1:
                continue
            if count == initial_pop:
                tp.descendants[i] = split_pops
                break
            count += 1

    def merge(self, source_pops, new_pop):
        """
        Merges two populations to one - this is always a 2 to 1 population
        function

        source_pops : tuple of populations that merge (always 0,1) - these go
                      extinct, and we have a new population left over

        new_pop: index of new population (always 0)

        To implements, we'll first use admix_new with a time period of zero,
        and then have the first two populations go extinct, leaving behind the
        merged population
        """
        pass

    def admix_new(self, source_pops, new_pop, f):
        """
        Creates a new population through admixture, with fraction f from first
        source pop

        source_pops : tuple of parental populations

        new_pop : index of new admixed population (if not in last position,
            shifts the rest)

        f : admixture fraction from first source population

        Assume that source population indices are adjacent, place new pop
        between them
        """
        tp = self.tp_list[-1]
        tp.admixture_new = []
        for i in range(len(tp.descendants)):
            if i in source_pops:
                if i == source_pops[0]:
                    tp.admixture_new.append(f)
                else:
                    tp.admixture_new.append(1 - f)
            else:
                tp.admixture_new.append(0)

    def admix_inplace(self, source_pop, target_pop, f):
        """
        New admixed population replaces second source population, with fraction
        f from first source pop

        source_pop : non-replaced source population

        new_pop : replaced source population

        f : admixture fraction from non-replaced source population
        """
        pass

    def evolve(self, time, popsizes, migrations):
        """
        Begins a new time period if necessary. Evolves current populations
        forward in time by calculating their sizes throughout the interval. Also
        moves model time forward and sets migration rates.

        time : Length of time to evolve.

        popsizes : Either a list of sizes for each current population, or a
                   function that returns a list of sizes for any time value
                   given between 0 and time.

        migrations : 2D array describing migration rates between populations.
        """
        # Create new time period
        tp = self.tp_list[-1]
        npops = 0
        for desc in tp.descendants:
            if isinstance(desc, tuple):
                npops += 2
            elif desc != -1:
                npops += 1
        if tp.admixture_new is not None:
            npops += 1
        new_tp = self.TimePeriod(self.current_time, npops, self.precision)
        self.tp_list.append(new_tp)
        tp = new_tp
        # Update current time period
        self.current_time += time
        tp.time = np.linspace(tp.time[0], self.current_time, num=self.precision)
        if callable(popsizes):
            popfunc = popsizes
        else:
            popfunc = lambda t: popsizes
        time_vals = np.linspace(0, time, num=self.precision)
        tp.popsizes = np.transpose([popfunc(t) for t in time_vals])
        # Use average migration if it is function
        if callable(migrations):
            sum_mig = 0
            step = 1 / (self.precision - 1)
            for t in np.arange(0, 1 + step, step):
                sum_mig += migrations(t)
            migrations = sum_mig / self.precision
        # Transpose because plotting assumes m[i,j] corresponds to migration from i to j
        if migrations is not None:
            tp.migrations = migrations.transpose()
        else:
            tp.migrations = migrations

    def extinction(self, extinct_pops):
        """
        Cause extinction of populations in extinct_pops and begin a new time
        period without them.

        extinct_pops : Sequence listing the indices of the populations to go
                       extinct.
        """
        tp = self.tp_list[-1]
        count = 0
        for i in range(len(tp.descendants)):
            if i in extinct_pops:
                tp.descendants[i] = -1
                count += 1
            else:
                if isinstance(tp.descendants[i], tuple):
                    tp.descendants[i][0] -= count
                    tp.descendants[i][1] -= count
                else:
                    tp.descendants[i] -= count

    def determine_framesizes(self):
        """
        Determines the overall size of the tree rooted at each population by
        working backwards through the list of time periods. This is necessary
        for allocating the proper amount of space to a given population when
        drawing it (i.e. a plotted population must be given enough vertical
        room for both itself and all of its descendants.
        """
        # Work backwards through population histories to update all tree sizes
        for tpindex in range(len(self.tp_list) - 1, -1, -1):
            tp = self.tp_list[tpindex]
            # Begin by assigning size to be largest size from this time period
            tp.framesizes = [max(size) for size in tp.popsizes]

            # Done if there are no descendants to consider
            if tp.descendants is None:
                continue

            tpnext = self.tp_list[tpindex + 1]
            # if a population contributes to an admixture, add that amount
            # to frame size
            if tp.admixture_new is not None:
                first = 1
                for i, f in enumerate(tp.admixture_new):
                    if f > 0:
                        if first == 1:
                            tp.framesizes[i] += f * tpnext.framesizes[i + 1]
                            first = 0
                        else:
                            tp.framesizes[i] += f * tpnext.framesizes[i]
            # Add information from descendant populations
            for i, desc in enumerate(tp.descendants):
                # If a population splits, add information from both descendants
                if isinstance(desc, tuple):
                    for dindex in desc:
                        tp.framesizes[i] += tpnext.framesizes[dindex]
                # Otherwise, update size to max of this pop and its descendant
                elif desc != -1:
                    mysize = tp.framesizes[i]
                    nextsize = tpnext.framesizes[desc]
                    tp.framesizes[i] = max(mysize, nextsize)

    def determine_drawinfo(self, tp_index, pop_index, origin, direc):
        """
        Determines the origin and draw direction of each population in the model
        so that they may be properly plotted. Works recursively through each
        population tree.

        tp_index : Index of the current TimePeriod in self.tp_list.

        pop_index : Index of the current population within the TimePeriod.

        origin : Initial origin value for the current population. May be
                 adjusted based on subpopulations.

        direc : Plot direction for the current population.
        """
        tp = self.tp_list[tp_index]
        # Set initial direc and origin info
        if not tp.direcs:
            tp.direcs = [1 for pop in range(len(tp.popsizes))]
        if not tp.origins:
            tp.origins = [(0, 0) for pop in range(len(tp.popsizes))]
        tp.direcs[pop_index] = direc
        tp.origins[pop_index] = origin
        vshift = 0

        # Recursively determine sub-population information if present
        if tp.descendants is not None:
            desc = tp.descendants[pop_index]
            # Split population case (harder)
            if isinstance(desc, tuple):
                # Shift origin to account for sub-population taking up space
                vshift = direc * (self.tp_list[tp_index + 1].framesizes[desc[0]])
                # Determine info for first sub-population
                vshift += self.determine_drawinfo(
                    tp_index + 1, desc[0], (tp.time[-1], origin[1] + vshift), -1 * direc
                )
                # Determine info for second subpopulation
                # Shift its origin to account for current population's size
                popsize = tp.popsizes[pop_index]
                self.determine_drawinfo(
                    tp_index + 1,
                    desc[1],
                    (tp.time[-1], origin[1] + direc * max(popsize) + vshift),
                    direc,
                )
            # Single descendant case (easier)
            elif desc != -1:
                vshift = self.determine_drawinfo(
                    tp_index + 1, desc, (tp.time[-1], origin[1]), direc
                )
            tp.origins[pop_index] = (origin[0], origin[1] + vshift)
        return vshift

    class TimePeriod:
        """
        Keeps track of population information and relationships during a
        specific time period of the demographic model. Also contains information
        about how populations should be drawn.

        precision : Number of times to evaluate population size within period.

        time : List, of length 'precision'. Equally spaced time intervals
               running from the start of this time period to the end of it.

        popsizes : List containing the size of each population in the current
                   time period. Size of each population is stored as a list of
                   length 'precision', effectively providing a function from
                   time to size for each population.

        descendants : List that maps the descendant populations for the next
                      TimePeriod. Each population has an entry in descendants,
                      which is either a single int specifying the index of the
                      descendant, or a tuple of two ints specifying the indices
                      of the populations it splits into.

        migrations : 2D array of floats, specifying the migration rates between
                     populations within the current time period.

        framesizes : List specifying the overall space the tree rooted at each
                     population takes up. This is useful for plotting later, and
                     is dependent on all descendant population sizes.

        direcs : List of directions that each population in time period is
                 drawn. For each population the value is equal to 1 if the
                 population should be drawn facing up, and -1 if the population
                 should be drawn facing down.

        origins : List of where each population in the time period should begin
                  to be drawn. For each population, if direc is 1, then the
                  origin is the lower-left corner of the space. If direc is -1,
                  then it is the upper-left corner. Represented as a tuple (x,y)
        """

        def __init__(self, time, npops, precision):
            """
            Sets basic information for the time period.

            time : Starting time of period.

            npops : Number of populations in the time period.

            precision : Value of precision variable.
            """
            self.precision = precision
            self.time = [time] * precision
            self.popsizes = [np.array([1.0] * precision) for pop in range(npops)]
            self.descendants = [pop for pop in range(npops)]
            self.migrations = None
            self.framesizes = None
            self.direcs = None
            self.origins = None

            self.admixture_new = None
