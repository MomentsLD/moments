"""
This module provides functionality to gather and store information about a
demographic model (as defined by a demographic model function), and to generate
a visual representation of the model by using this information.

Generating information about a model and plotting it is simple. The following
two functions are all that is necessary to generate a plot of the model defined
by demographic_model_func(params, ns):

model = ModelPlot.generate_model(demographic_model_func, params, ns)
ModelPlot.plot_model(model)

This module is currently compatible with the following methods in moments used 
to build demographic models:

LinearSystem_1D.steady_state_1D
LinearSystem.steady_state
Manips.split_{1D_to_2D,2D_to_3D_2,2D_to_3D_1,3D_to_4D_3,4D_to_5D_4}
Spectrum_mod.Spectrum.integrate
"""
import matplotlib.offsetbox as mbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

## USER FUNCTIONS ##
def generate_model(model_func, params, ns, precision=100):
    """
    Generates information about a demographic model, and returns the information
    in a format that can be used by the plot_model function (i.e. the info is 
    stored in a ModelPlot._ModelInfo object).
    
    model_func : Function of the form model_func(params, ns). Describes the
                 demographic model to collect information on.

    params : List of values of the demographic variables to be passed as the 
             params argument to model_func.

    ns : List of sample sizes to be passed as the ns argument to model_func.

    precision : Number of times to evaluate population sizes per period.
    """
    # Initialize model and collect necessary information
    model = _ModelInfo(precision)
    model_func(params, ns)
    _close_model()
    # Determine size of population trees
    model.determine_framesizes()
    # Determine information for plotting recursively for each population tree
    vstart = 0
    tp1 = model.tp_list[0]
    for pop_index in range(len(tp1.popsizes)):
        model.determine_drawinfo(0, pop_index, (0,vstart), 1)
        vstart += tp1.framesizes[pop_index]
    return model

def plot_model(model, save_file=None, pop_labels=None, nref=None, 
               fig_title="Demographic Model", fig_bg_color='#ffffff', 
               plot_bg_color='#93a1a1', gridline_color='#586e75', 
               text_color='#002b36', pop_color='#268bd2', 
               draw_migrations=True, arrow_color='#073642', arrow_scale=0.01):         
    """
    Plots a demographic model based on information contained within a _ModelInfo
    object. Returns the matplotlib Figure object that was created. See the
    matplotlib documentation for valid entries for the color parameters.

    model : A _ModelInfo object created using generate_model().

    save_file : If not None, the figure will be saved to this location.

    nref : If specified, this will update the time and population size labels to
           use units based on an ancestral population size of nref. See the
           documentation for details.

    pop_labels : If not None, should be a list of strings of the same length as
                 the total number of final populations in the model. The string
                 at index i should be the name of the population along axis i in
                 the model's SFS.
    
    fig_title : Title of the figure.

    fig_bg_color : Background color of figure (i.e. border surrounding the
                   drawn model).

    plot_bg_color : Background color of the actual plot area.

    gridline_color : Color of the plot gridlines.

    text_color : Color of text in the figure.

    pop_color : Color of the populations.

    draw_migrations : If False, migration arrows will not be drawn.

    arrow_color : Color of the arrows showing migrations between populations.

    arrow_scale : Float to control the size of the migration arrows.
    """
    # Set up basics of plot
    fig = plt.figure(facecolor=fig_bg_color)
    ax = fig.add_subplot(111)
    ax.set_axis_bgcolor(plot_bg_color)
    ax.set_title(fig_title, color=text_color,fontsize=24)
    if nref:
        ax.set_xlabel("Time (Generations)", color=text_color, fontsize=16)
        ax.set_ylabel("Population Sizes", color=text_color, fontsize=16)
    else:
        ax.set_xlabel("Time (Genetic Units)", color=text_color, fontsize=16)
        ax.set_ylabel("Relative Population Sizes", color=text_color, fontsize=16)
    # Determine various max values for proper scaling within the model
    xmax = model.tp_list[-1].time[-1]
    ymax = sum(model.tp_list[0].framesizes)
    mig_max = 0
    for tp in model.tp_list:
        if tp.migrations is None:
            continue
        mig = np.amax(tp.migrations)
        mig_max = mig_max if mig_max > mig else mig

    # Configure axis border colors
    ax.spines['top'].set_color(text_color)
    ax.spines['right'].set_color(text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    
    # Configure ticks along x-axis (time)
    xticks = [tp.time[0] for tp in model.tp_list]
    xticks.append(xmax)
    ax.xaxis.set_major_locator(mticker.FixedLocator(xticks))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(which='both', axis='x', labelcolor=text_color,
                   labelsize=12, top=False)
    # If nref is given use the appropriate time units (2*nref generations)
    if nref:
        ax.set_xticklabels([str(2*nref*x) for x in xticks])

    # Configure gridlines along y-axis (population size)
    ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(ymax)))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(b=True, which='major', axis='y', color=gridline_color)
    ax.tick_params(which='both', axis='y', colors='none')

    # Add population size scale
    bar = mbox.AuxTransformBox(ax.transData)
    bar.add_artist(mpatches.Rectangle((0,0), 0, 1, edgecolor=text_color,
                                      facecolor='none'))
    label = mbox.TextArea(str(nref) if nref else "Nref")
    label.get_children()[0].set_color(text_color)
    bar = mbox.HPacker(children=[label, bar], pad=0, sep=2,
                       align="center")
    scalebar = mbox.AnchoredOffsetbox(2, pad=0.25, borderpad=0.25, child=bar,
                                      frameon=False)
    ax.add_artist(scalebar)

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
            y2 = origin[1] + (direc*popsize)
            ax.fill_between(tp.time, y1, y2, color=pop_color, 
                            edgecolor=pop_color)

            # Draw connections to next populations if necessary
            if tp.descendants is not None:
                desc = tp.descendants[pop_index]
                tp_next = model.tp_list[tp_index+1]
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
                cx = tp.time[-10:]
                cy_below_1 = [origin[1]]*10
                cy_above_1 = origin[1] + direc*popsize[-10:]
                cy_below_2 = np.linspace(cy_below_1[0], connect_below, 10)
                cy_above_2 = np.linspace(cy_above_1[0], connect_above, 10)
                ax.fill_between(cx, cy_below_1, cy_below_2, color=pop_color,
                                edgecolor=pop_color)
                ax.fill_between(cx, cy_above_1, cy_above_2, color=pop_color,
                                edgecolor=pop_color)

            # Draw migrations if necessary
            if draw_migrations and tp.migrations is not None:
                # Iterate through migrations for current population 
                for mig_index, mig_val in enumerate(tp.migrations[pop_index]):
                    # If no migration, continue
                    if mig_val == 0:
                        continue
                    num_migrations += 1
                    offset = int(tp.precision * num_migrations / 
                             (total_migrations + 1.0))
                    x = tp.time[offset]
                    dx = 0
                    # Determine which sides of populations are closest
                    y1 = origin[1]
                    y2 = y1 + direc*popsize[offset]
                    mig_y1 = tp.origins[mig_index][1]
                    mig_y2 = mig_y1 + (tp.direcs[mig_index] *
                                       tp.popsizes[mig_index][offset])
                    y = y1 if abs(mig_y1 - y1) < abs(mig_y1 - y2) else y2
                    dy = mig_y1-y if abs(mig_y1 - y) < abs(mig_y2 - y) \
                                  else mig_y2-y
                    # Scale arrow to proper size                  
                    mig_scale = max(0.1, mig_val/mig_max)
                    awidth = xmax * arrow_scale * mig_scale
                    alength = ymax * arrow_scale
                    ax.arrow(x,y,dx,dy, width=awidth, head_width=awidth*3,
                             head_length = alength, color=arrow_color,
                             length_includes_head = True)
    
    # Label populations, if correct labels are given
    tp_last = model.tp_list[-1]
    if pop_labels and len(pop_labels) == len(tp_last.popsizes):
        ax2 = ax.twinx()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())
        # Determine placement of ticks
        yticks = [tp_last.origins[i][1] + 0.5 * tp_last.direcs[i] *
                  tp_last.popsizes[i][-1] for i in range(len(tp_last.popsizes))]
        ax2.yaxis.set_major_locator(mticker.FixedLocator(yticks))
        ax2.set_yticklabels(pop_labels)
        ax2.tick_params(which='both', color='none', labelcolor=text_color,
                        labelsize=16, left=False, top=False, right=False)
        ax2.spines['top'].set_color(text_color)
        ax2.spines['left'].set_color(text_color)
        ax2.spines['right'].set_color(text_color)
        ax2.spines['bottom'].set_color(text_color)

    # Display figure
    if save_file:
        fig.set_size_inches(9.6, 5.4)
        plt.savefig(save_file, dpi=200, facecolor=fig_bg_color)
    else:
        plt.show(fig)

    return fig
       

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
    resources by collecting information when it is not being used.
    """
    global _current_model
    _current_model = None


class _ModelInfo():
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
        Sets itself as the current model, to be able to collect data from other
        methods in moments, and initializes the class variables.

        precision : Sets the precision variable.
        """
        global _current_model
        _current_model = self
        self.current_time = 0.0
        self.tp_list = []
        self.precision = precision

    class TimePeriod():
        """
        Keeps track of population information and relationships during a 
        specific time period of the demographic model. Also contains information
        about how populations should be drawn.
        
        precision : Number of times to evaluate population size within period.

        time : List, with length specified by ModelInfo's precision variable.
               Equally spaced time intervals running from the start of this time
               period to the end of it.
        
        popsizes : List containing the size of each population in the current
                   time period. Size of each population is stored as a list the
                   same length as time, effectively providing a function from
                   time to size for each population.

        descendants : List that maps the descendant populations for the next
                      TimePeriod. Each population has an entry in descendants,
                      which is either a single int specifying the index of the
                      descendant, or a tuple of two ints specifying the indices
                      of the populations it splits into.
        
        migrations : 2D array of floats, specifying the migration rates between
                     populations within the current time period.
                     migrations[i][j]=x means that the migration rate from i to 
                     j was x during this time period.
        
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
            self.time = [time]*precision
            self.popsizes = [[1]*precision for pop in range(npops)]
            self.descendants = None
            self.migrations = None
            self.framesizes = None
            self.direcs = None
            self.origins = None


    def initialize(self, npops):
        """
        Creates initial ancestral population(s) in steady-state.

        npops : Number of ancestral populations.
        """
        self.current_time = 0.0
        tp = self.TimePeriod(self.current_time, npops, self.precision)
        self.tp_list = [tp]

    def split(self, initial_pop, split_pops):
        """
        Splits one of the current populations into two.

        initial_pop : index of population to split.

        split_pops : tuple of the two indices of the resulting populations.
        """
        tp = self.tp_list[-1]
        tp.descendants = [pop for pop in range(len(tp.popsizes))]
        tp.descendants[initial_pop] = split_pops
        new_tp = self.TimePeriod(self.current_time, len(tp.popsizes) + 1,
                                 self.precision)
        self.tp_list.append(new_tp)

    def evolve(self, time, popsizes, migrations):
        """
        Evolves current populations forward in time.
        
        time : Length of time to evolve.

        popsizes : Either a list of sizes for each current population, or a
                   function that returns a list of sizes for any time value 
                   given between 0 and time.

        migrations : 2D array describing migration rates between populations.
        """
        self.current_time += time
        tp = self.tp_list[-1]
        tp.time = np.linspace(tp.time[0], self.current_time, num=self.precision)
        if callable(popsizes):
            popfunc = popsizes
        else:
            popfunc = lambda t : popsizes
        time_vals = np.linspace(0, time, num=self.precision)
        tp.popsizes = np.transpose([popfunc(t) for t in time_vals])
        tp.migrations = migrations

    def determine_framesizes(self):
        """
        Determines the overall size of the tree rooted at each population by
        working backwards through the list of time periods. This is necessary
        for allocating the proper amount of space to a given population when 
        drawing it.
        """
        # Leaf node populations only dependent on their own max size reached
        last_tp = self.tp_list[-1]
        last_tp.framesizes = [max(size) for size in last_tp.popsizes]
        
        # If only one time period, method is complete
        if len(self.tp_list) == 1:
            return

        # Work backwards through population histories to update all tree sizes
        for tpindex in range(len(self.tp_list)-2,-1,-1):
            tp = self.tp_list[tpindex]
            tpnext = self.tp_list[tpindex+1]
            # Begin by assigning size to be largest size from this time period
            tp.framesizes = [max(size) for size in tp.popsizes]
            # Add information from descendant populations
            for i,desc in enumerate(tp.descendants):
                # If a population splits, add information from both descendants
                if isinstance(desc, tuple):
                    for dindex in desc:
                        tp.framesizes[i] += tpnext.framesizes[dindex]
                # Otherwise, update size to max of this pop and its descendant
                else:
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

        direc : Direc value for the current population.
        """
        tp = self.tp_list[tp_index]
        # Set initial direc and origin info
        if not tp.direcs:
            tp.direcs = [1 for pop in range(len(tp.popsizes))]
        if not tp.origins:
            tp.origins = [(0,0) for pop in range(len(tp.popsizes))]
        tp.direcs[pop_index] = direc
        tp.origins[pop_index] = origin

        # Recursively determine sub-population information if present
        if tp.descendants is not None:
            desc = tp.descendants[pop_index]
            # Split population case (harder)
            if isinstance(desc, tuple):
                # Shift origin to account for sub-population taking up space
                origin = (origin[0], origin[1] + (direc *
                          (self.tp_list[tp_index+1].framesizes[desc[0]])))
                tp.origins[pop_index] = origin
                # Determine info for first sub-population
                self. determine_drawinfo(tp_index+1, desc[0], 
                                         (tp.time[-1], origin[1]), -1*direc)
                # Determine info for second subpopulation 
                # Shift origin to account for this population's size
                popsize = tp.popsizes[pop_index]
                self.determine_drawinfo(tp_index+1, desc[1], (tp.time[-1],
                                        origin[1] + direc*max(popsize)), direc)
            # Single descendant case (easier)
            else:
                self.determine_drawinfo(tp_index+1, desc, 
                                        (tp.time[-1], origin[1]), direc)


# The following code was adapted from scalebars.py, by dmeliza, located here on
# GitHub: https://gist.github.com/dmeliza/3251476. scalebars.py is licensed with
# the Python Software Foundation license (http://docs.python.org/license.html).
class _AnchoredScaleBar(mbox.AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, 
                 loc=4, pad=0.1, borderpad=0.1, sep=2, prop=None, **kwargs):
        """
        Draw a horizontal and/or vertical bar with the size in data coordinates
        of the give axes. A label will be drawn underneath (center-aligned).
        
        transform : Coordinate frame to use (typically axes.transData)
        
        sizex,sizey: Width of the x and y bars, in data units. 0 to omit.

        labelx,labely : Labels for the x and y  bars. None to omit.

        loc : Position in containing axes.

        pad,borderpad : Padding, in fraction of the legend font size (or prop).

        sep : Separation between labels and bars in points.
        
        **kwargs : Additional arguments passed to base class constructor.
        """
        bars = mbox.AuxTransformBox(transform)
        if sizex:
            bars.add_artist(mpatches.Rectangle((0,0), sizex, 0, fc="none"))
        if sizey:
            bars.add_artist(mpatches.Rectangle((0,0), 0, sizey, fc="none"))

        if sizex and labelx:
            bars = mbox.VPacker(children=[bars, mbox.TextArea(labelx, 
                                minimumdescent=False)], pad=0, sep=sep,
                                align="center")
        if sizey and labely:
            bars = mbox.HPacker(children=[mbox.TextArea(labely), bars], pad=0,
                                sep=sep, align="center")

        mbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                        child=bars, prop=prop, frameon=False, 
                                        **kwargs)
