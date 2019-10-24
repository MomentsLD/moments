import matplotlib.pylab as plt, matplotlib
import numpy as np
import sys,os


# Set fontsize to 10
matplotlib.rc('font',**{'family':'sans-serif',
                        'sans-serif':['Helvetica'],
                        'style':'normal',
                        'size':10 })
# Set label tick sizes to 8
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)


def plot_ld_curves(ld_stats, stats_to_plot=[], rows=None, cols=None,
                   statistics=None, fig_size=(6,6), dpi=150, r_edges=None,
                   numfig=1, cM=False, output=None, show=False):
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
        plot_ld_curves(ld_stats, stats_to_plot=[['DD_1_1','DD_2_2','DD_3_3'],
            ['DD_1_2','DD_1_3','DD_2_3'],['Dz_1_1_1','Dz_2_2_2','Dz_3_3_3'],
            ['pi2_2_2_2_2','pi2_3_3_3_3']], rows=2, cols=2, 
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
    
    r_centers = np.array((r_edges[:-1]+r_edges[1:])/2)
    x_label = '$r$'
    if cM == True:
        r_centers *= 100
        x_label = 'cM'
    
    fig = plt.figure(numfig, figsize=fig_size, dpi=dpi)
    fig.clf()
    
    axes={}
    # loop through stats_to_plot, update axis, and plot
    for i,stats in enumerate(stats_to_plot):
        axes[i] = plt.subplot(rows,cols,i+1)
        for stat in stats:
            k = statistics[0].index(stat)
            to_plot = [ld_stats[j][k] for j in range(len(r_centers))]
            axes[i].plot(r_centers, to_plot, label=stat)
        
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlabel(x_label)
        axes[i].legend(frameon=False)
    
    fig.tight_layout()
    
    if output != None:
        plt.savefig(output)
    
    if show == True:
        fig.show()
    else:
        return fig

def plot_ld_curves_comp(ld_stats, ms, vcs, stats_to_plot=[], rows=None, cols=None,
                   statistics=None, fig_size=(6,6), dpi=150, r_edges=None,
                   numfig=1, cM=False, output=None, show=False,
                   plot_means=True, plot_vcs=True):
    """
    Plot comparison between expected stats (y) and data (ms, vcs)
    
    stats_to_plot is a list of lists, where each inner list gives the stats
        to plot in a given pane, and the list of lists are each pane
    rows and cols tells us how to arrange the panes, if there are more than one 
        sets of statistics to plot
    if statistics is None, we use the statistics as listed in ld_stats.names()
    
    For example, to plot four panes, two on each of two rows, with the D^2
        statistics, the between-population cov(D) statistics, the non-cross
        populations Dz statistics, and the non-cross pop pi2 statistics, we run
        plot_ld_curves(ld_stats, stats_to_plot=[['DD_1_1','DD_2_2','DD_3_3'],
            ['DD_1_2','DD_1_3','DD_2_3'],['Dz_1_1_1','Dz_2_2_2','Dz_3_3_3'],
            ['pi2_2_2_2_2','pi2_3_3_3_3']], rows=2, cols=2, 
            statistics=statistics)
    """
    
    # Check that all the data has the correct dimensions
    
    num_axes = len(stats_to_plot)
    if num_axes == 0:
        return
    
    if rows == None and cols == None:
        cols = len(stats_to_plot)
        rows = 1
    elif cols == None:
        cols = int(np.ceil(len(stats_to_plot)/rows))
    elif rows == None:
        rows = int(np.ceil(len(stats_to_plot)/cols))
    
    if statistics == None:
        statistics = ld_stats.names()
    
    # make sure all stats are named properly
    
    r_centers = np.array((r_edges[:-1]+r_edges[1:])/2)
    x_label = '$r$'
    if cM == True:
        r_centers *= 100
        x_label = 'cM'
    
    fig = plt.figure(numfig, figsize=fig_size, dpi=dpi)
    fig.clf()
    axes = {}
    # loop through stats_to_plot, update axis, and plot
    for i,stats in enumerate(stats_to_plot):
        axes[i] = plt.subplot(rows,cols,i+1)
        if plot_vcs:
            for stat in stats:
                k = statistics[0].index(stat)
                data_to_plot = np.array([ms[j][k] for j in range(len(r_centers))])
                data_error = np.array([vcs[j][k][k]**.5 * 1.96 for j in range(len(r_centers))])
                axes[i].fill_between(r_centers, data_to_plot-data_error, data_to_plot+data_error,
                                alpha=.25, label=None)
        
        # reset color cycle
        plt.gca().set_prop_cycle(None)
        if plot_means:
            for stat in stats:
                k = statistics[0].index(stat)
                data_to_plot = np.array([ms[j][k] for j in range(len(r_centers))])
                axes[i].plot(r_centers, data_to_plot, '--', label=None)
        
        # reset color cycle
        plt.gca().set_prop_cycle(None)
        for stat in stats:
            k = statistics[0].index(stat)
            exp_to_plot = [ld_stats[j][k] for j in range(len(r_centers))]
            axes[i].plot(r_centers, exp_to_plot, label=stat)
        
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlabel(x_label)
        axes[i].legend(frameon=False)
    
    fig.tight_layout()
    
    if output != None:
        plt.savefig(output)
    
    if show == True:
        fig.show()
    else:
        return fig

