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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

## USER FUNCTIONS ##
def generate_model(model_func, params, ns):
    """
    Generates information about a demographic model, and returns the information
    in a format that can be used by the plot_model function (i.e. the info is 
    stored in a ModelPlot._ModelInfo object).
    
    model_func : Function of the form model_func(params, ns). Describes the
                 demographic model to collect information on.

    params : List of values of the demographic variables to be passed as the 
             params argument to model_func.

    ns : List of sample sizes to be passed as the ns argument to model_func.
    """
    model = _ModelInfo()
    model_func(params, ns)
    _close_model()
    model.determine_poptree_sizes()
    return model

def plot_model(model):
    """
    Plots a demographic model based on information contained within a _ModelInfo
    object.

    model : A _ModelInfo object created using generate_model().
    """
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time (Genetic Units)")
    ax.set_ylabel("Relative Population Sizes")

    origin = (0,0)
    tp1 = model.tp_list[0]
    for pop_index in range(len(tp1.popsizes)):
        _draw_poptree(ax, model.tp_list, 0, pop_index, origin, 1)
        origin = (0, origin[1] + tp1.poptree_sizes[pop_index])
    plt.show(fig)
        

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

def _draw_poptree(ax, tp_list, tp_index, pop_index, origin, direc):
    if tp_index >= len(tp_list):
        return
    tp = tp_list[tp_index]
    popsize = tp.popsizes[pop_index]
    max_popsize = max(popsize) if isinstance(popsize, np.ndarray) else popsize
    desc = tp.descendants[pop_index]
    if isinstance(desc, tuple):
        subtree_size = tp_list[tp_index+1].poptree_sizes[desc[0]]
        _draw_poptree(ax, tp_list, tp_index+1, desc[0], 
                    (tp.end_time, origin[1] + direc*(subtree_size)), 
                    -1*direc)
        _draw_poptree(ax, tp_list, tp_index+1, desc[1], 
                    (tp.end_time, origin[1] + direc*(subtree_size+max_popsize)),
                    direc)
        origin = (origin[0], origin[1] + direc*subtree_size)
    else:
        _draw_poptree(ax, tp_list, tp_index+1, desc, 
                (tp.end_time, origin[1]), direc)
    x = np.arange(tp.start_time, tp.end_time + 0.01, 0.02)
    y1 = origin[1]
    y2 = origin[1] + (direc*popsize)
    ax.fill_between(x, y1, y2)

class _ModelInfo():
    """
    Uses information sent by method calls in moments to generate a demographic
    model. Model information is stored as a list of TimePeriod objects, each of
    which stores demographic information (population sizes, splits, migration 
    rates) for a given time step in the model.
    """
    
    class TimePeriod():
        """
        Keeps track of population information and relationships during a 
        specific time period of the demographic model.
        """
        def __init__(self, time, npops):
            self.start_time = time
            self.end_time = time
            self.popsizes = [1 for x in range(npops)]
            self.descendants = [x for x in range(npops)]
            self.migrations = None
            self.poptree_sizes = None

    def __init__(self):
        global _current_model
        _current_model = self
        self.current_time = -1
        self.tp_list = None

    def initialize(self, npops):
        """
        Creates initial ancestral population(s) in steady-state.

        npops : Number of ancestral populations.
        """
        self.current_time = 0.0
        tp = self.TimePeriod(0.0, npops)
        self.tp_list = [tp]

    def split(self, initial_pop, split_pops):
        """
        Splits one of the current populations into two.

        initial_pop : index of population to split.
        split_pops : indices of the resulting populations. (tuple)
        """
        current_tp = self.tp_list[-1]
        current_tp.descendants[initial_pop] = split_pops
        new_tp = self.TimePeriod(current_tp.end_time, 
                                 len(current_tp.descendants) + 1)
        self.tp_list.append(new_tp)

    def evolve(self, time, popsizes, migrations):
        """
        Evolves current populations forward in time.
        
        time : Length of time to evolve.
        popsizes : List of sizes for each current population.
        mig : 2D array describing migration rates between populations.
        """
        self.current_time += time
        current_tp = self.tp_list[-1]
        current_tp.end_time = self.current_time
        if callable(popsizes):
            times = np.arange(0, time+0.01, 0.02)
            current_tp.popsizes = np.transpose([popsizes(t) for t in times])
        else:
            current_tp.popsizes = popsizes
        current_tp.migrations = migrations

    def determine_poptree_sizes(self):
        """
        Determines the overall size of the tree rooted at each population by
        working backwards through the list of time periods. This is necessary
        for allocating the proper amount of space to a given population when 
        drawing it.
        """
        # Can't be called before time period info has been generated
        if self.tp_list is None:
            raise Exception("Must generate model information before determining"
                            "sizes of the population trees")
        
        # Size of tree rooted at leaf node populations only dependent on the
        # max size reached by that population
        last_tp = self.tp_list[-1]
        last_tp.poptree_sizes = [max(size) if isinstance(size, np.ndarray) 
                                 else size for size in last_tp.popsizes]
        # If only one time period, method is complete
        if len(self.tp_list) == 1:
            return

        # Work backwards through population histories to update all tree sizes
        for tpindex in range(len(self.tp_list)-2,-1,-1):
            tp = self.tp_list[tpindex]
            tpnext = self.tp_list[tpindex+1]
            # Begin by assigning size to be largest size from this time period
            tp.poptree_sizes = [max(size) if isinstance(size, np.ndarray) 
                                else size for size in tp.popsizes]
            # Add information from descendant populations
            for i,desc in enumerate(tp.descendants):
                # If a population splits, add information from both descendants
                if isinstance(desc, tuple):
                    for dindex in desc:
                        tp.poptree_sizes[i] += tpnext.poptree_sizes[dindex]
                # Otherwise, update size to max of this pop and its descendant
                else:
                    mysize = tp.poptree_sizes[i]
                    nextsize = tpnext.poptree_sizes[desc]
                    tp.poptree_sizes[i] = max(mysize, nextsize)
