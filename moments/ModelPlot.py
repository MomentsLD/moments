"""
This module provides access to the ModelPlotter class, which runs in the
background and gathers information on a demographic model as it is created by
the various methods provided in moments. ModelPlotter can then use this
information to generate a visual representation of the model.

ModelPlot is designed to be simple to use. The following code is all that is 
necessary to generate a plot of the model specified by 
demographic_model_func(params, ns):

mp = ModelPlot.generate_model(demographic_model_func, params)
ModelPlot.plot(mp)
"""
import numpy as np
import matplotlib.pyplot as plt

## USER FUNCTIONS ##
def generate_model(model_func, params, ns):
    """
    Generates the information necessary to plot the demographic model specified
    by model_func.
    
    model_func : Function of the form model_func(params, ns). Describes the
                 demographic model to be plotted.

    params : List of values of the demographic variables to be passed as the 
             params argument to model_plot.

    ns : List of sample sizes to be passed as the ns argument toy model_func.
    """
    mp = _ModelPlotter()
    model_func(params, ns)
    mp.determine_poptree_sizes()
    _close()
    return mp

def plot(model_plotter):
    """
    Plots a demographic model based on information from a ModelPlotter object.

    model_plotter : A ModelPlotter object created using generate_model().
    """
    for i, tp in enumerate(model_plotter.time_list):
        print("Time Period #{}".format(i))
        print("Start: {}, End: {}".format(tp.start_time, tp.end_time))
        print("Population Sizes: {}".format(','.join([str(x) for x in tp.popsizes])))
        print("Descendants: {}".format(','.join([str(x) for x in tp.descendants])))
        print("Migration Matrix:")
        if tp.migrations is None:
            print("None")
        else:
            for row in tp.migrations:
                print(' '.join([str(x) for x in row]))
        print("Pop Tree Sizes: {}".format(','.join([str(x) for x in tp.poptree_sizes])))
        print('')


## MOMENTS IMPLEMENTATION FUNCTIONS
_current_plotter = None
def _get_plotter():
    """
    Used by the various methods in moments to determine the correct 
    ModelPlotter object to send information to (if any).
    """
    global _current_plotter
    return _current_plotter

def _close():
    """
    Sets _current_plotter to None to ensure that ModelPlot does not waste
    resources by collecting information when it is not being used.
    """
    global _current_plotter
    _current_plotter = None


class _ModelPlotter():
    """
    Uses information sent by methods in moments to generate a demographic
    model. Model information is stored as a list of _TimePeriod objects, each of
    which stores demographic information (population sizes, splits, migration 
    rates) for a given time step in the model.
    """
    def __init__(self):
        global _current_plotter
        _current_plotter = self
        self.current_time = -1
        self.time_list = None

    def initialize(self, npops):
        """
        Creates initial ancestral population(s) in steady-state.

        npops : Number of ancestral populations.

        Only currently supports input of one ancestral population
        (i.e. from LinearSystem_1D.steady_state)
        """
        self.current_time = 0.0
        tp = _TimePeriod(0.0, npops)
        self.time_list = [tp]

    def split(self, initial_pop, split_pops):
        """
        Splits one of the current populations into two.

        initial_pop : index of population to split.
        split_pops : indices of the resulting populations.
        """
        current_tp = self.time_list[-1]
        current_tp.descendants[initial_pop] = split_pops
        new_tp = _TimePeriod(current_tp.end_time, 
                             len(current_tp.descendants) + 1)
        self.time_list.append(new_tp)

    def evolve(self, time, popsizes, migrations):
        """
        Evolves current populations forward in time.
        
        time : Length of time to evolve.
        popsizes : List of sizes for each current population.
        mig : 2D array describing migration rates between populations.
        """
        self.current_time += time
        current_tp = self.time_list[-1]
        current_tp.end_time = self.current_time
        if callable(popsizes):
            start = popsizes(0.0)
            end = popsizes(time)
            current_tp.popsizes = zip(start,end)
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
        if self.time_list is None:
            raise Exception("Must generate model information before determining"
                            "sizes of the population trees")
        
        # Size of tree rooted at leaf node populations only dependent on the
        # max size reached by that population
        last_tp = self.time_list[-1]
        last_tp.poptree_sizes = [max(size) if isinstance(size, tuple) else size
                                 for size in last_tp.popsizes]
        # If only one time period, method is complete
        if len(self.time_list) == 1:
            return

        # Work backwards through population histories to update all tree sizes
        for tpindex in range(len(self.time_list)-2,-1,-1):
            tp = self.time_list[tpindex]
            tpnext = self.time_list[tpindex+1]
            # Begin by assigning size to be largest size from this time period
            tp.poptree_sizes = [max(size) if isinstance(size, tuple) else size
                                for size in tp.popsizes]
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

class _TimePeriod():
    """
    Used by ModelPlotter to keep track of population information and 
    relationships during a specified time period in the model.
    """
    def __init__(self, time, npops):
        self.start_time = time
        self.end_time = time
        self.popsizes = [1 for x in range(npops)]
        self.descendants = [x for x in range(npops)]
        self.migrations = None
        self.poptree_sizes = None

