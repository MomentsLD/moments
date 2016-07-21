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
    _close()
    return mp

def plot(model_plotter):
    """
    Plots a demographic model based on information from a ModelPlotter object.

    model_plotter : A ModelPlotter object created using generate_model().
    """


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
    Uses information sent by methods with moments to generate a demographic
    model. Model information is stored as a list of _TimePeriod objects, each of
    which stores demographic information (population sizes, splits, migration 
    rates) for a given time step in the model.
    """
    def __init__(self):
        global _current_plotter
        _current_plotter = self
        self.current_time = 0
        self.time_list = list()

    def initial_pop(npops):
        # To be read from steady_state functions

    def split(split_pops):
        # To be read from split functions

    def evolve(info):
        # To be read from integrate functions


class _TimePeriod():
    """
    Used by ModelPlotter to keep track of population information and 
    relationships during a specified time period in the model.
    """
    def __init__(self):
        self.popsizes = None
        self.migrations = None
        self.descendants = None
