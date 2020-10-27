# Numpy is the numerical library moments is built upon
from numpy import array

# import dadi
import moments

# In demographic_models.py, we've defined a custom model for this problem
import demographic_models

# Load the data
data = moments.Spectrum.from_file("YRI_CEU.fs")
ns = data.sample_sizes

# The Demographics1D and Demographics2D modules contain a few simple models,
# mostly as examples. We could use one of those.
func = moments.Demographics2D.split_mig
# Instead, we'll work with our custom model
func = demographic_models.prior_onegrow_mig

# Now let's optimize parameters for this model.

# The upper_bound and lower_bound lists are for use in optimization.
# Occasionally the optimizer will try wacky parameter values. We in particular
# want to exclude values with very long times, very small population sizes, or
# very high migration rates, as they will take a long time to evaluate.
# Parameters are: (nu1F, nu2B, nu2F, m, Tp, T)
upper_bound = [100, 100, 100, 10, 3, 3]
lower_bound = [1e-2, 1e-2, 1e-2, 0, 0, 0]

# This is our initial guess for the parameters, which is somewhat arbitrary.
p0 = [2, 0.1, 2, 1, 0.2, 0.2]

# Perturb our parameters before optimization. This does so by taking each
# parameter a up to a factor of two up or down.
p0 = moments.Misc.perturb_params(
    p0, fold=1, upper_bound=upper_bound, lower_bound=lower_bound
)
# Do the optimization. By default we assume that theta is a free parameter,
# since it's trivial to find given the other parameters. If you want to fix
# theta, add a multinom=False to the call.
# The maxiter argument restricts how long the optimizer will run. For real
# runs, you will want to set this value higher (at least 10), to encourage
# better convergence. You will also want to run optimization several times
# using multiple sets of intial parameters, to be confident you've actually
# found the true maximum likelihood parameters.
print("Beginning optimization ************************************************")
popt = moments.Inference.optimize_log(
    p0,
    data,
    func,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    verbose=len(p0),
    maxiter=100,
)
# The verbose argument controls how often progress of the optimizer should be
# printed. It's useful to keep track of optimization process.
print("Finished optimization **************************************************")
print(popt)

# These are the actual best-fit model parameters, which we found through
# longer optimizations and confirmed by running multiple optimizations.
# We'll work with them through the rest of this script.
popt = [1.881, 0.0710, 1.845, 0.911, 0.355, 0.111]
print("Best-fit parameters: {0}".format(popt))

# Calculate the best-fit model AFS.
model = func(popt, ns)
# Likelihood of the data given the model AFS.
ll_model = moments.Inference.ll_multinom(model, data)
print("Maximum log composite likelihood: {0}".format(ll_model))
# The optimal value of theta given the model.
theta = moments.Inference.optimal_sfs_scaling(model, data)
print("Optimal value of theta: {0}".format(theta))

# Plot a comparison of the resulting fs with the data.
import pylab

pylab.figure(1)
moments.Plotting.plot_2d_comp_multinom(
    model, data, vmin=1, resid_range=3, pop_ids=("YRI", "CEU")
)
# This ensures that the figure pops up. It may be unecessary if you are using
# ipython.
pylab.show()
# Save the figure
pylab.savefig("YRI_CEU.png", dpi=50)

# Now that we've found the optimal parameters, we can use ModelPlot to
# automatically generate a graph of our determined model.

# First we generate the model by passing in the demographic function we used,
# and the optimal parameters determined for it.
model = moments.ModelPlot.generate_model(func, popt, ns)

# Next, we plot the model. See ModelPlot.py for more information on the various
# parameters that can be passed to the plotting function. In this case, we scale
# the model to have an original starting population of size 11293 and a
# generation time of 29 years. Results are saved to YRI_CEU_model.png.
moments.ModelPlot.plot_model(
    model,
    save_file="YRI_CEU_model.png",
    fig_title="YRI CEU Example Model",
    pop_labels=["YRI", "CEU"],
    nref=11293,
    gen_time=29.0,
    gen_time_units="Years",
    reverse_timeline=True,
)

# Let's generate some data using ms, if you have it installed.
"""
mscore = demographic_models.prior_onegrow_mig_mscore(popt)
# I find that it's most efficient to simulate with theta=1, average over many
# iterations, and then scale up.
mscommand = moments.Misc.ms_command(1., ns, mscore, int(1e5))
# If you have ms installed, uncomment these lines to see the results.

# We use Python's os module to call this command from within the script.
import os
return_code = os.system('{0} > test.msout'.format(mscommand))
# We check the return code, so the script doesn't crash if you don't have ms
# installed
if return_code == 0:
    msdata = moments.Spectrum.from_ms_file('test.msout')
    pylab.figure(2)
    moments.Plotting.plot_2d_comp_multinom(model, theta*msdata, vmin=1,
                                        pop_ids=('YRI','CEU'))
    pylab.show()

# Estimate parameter uncertainties using the Godambe Information Matrix, to
# account for linkage in the data. To use the GIM approach, we need to have
# spectra from bootstrapping our data.  Let's load the ones we've provided for
# the example.  
# (We're using Python list comprehension syntax to do this in one line.)
all_boot = [moments.Spectrum.from_file('bootstraps/{0:02d}.fs'.format(ii))
            for ii in range(100)]
uncerts = moments.Godambe.GIM_uncert(func, all_boot, popt, data,
                                  multinom=True)
# uncert contains the estimated standard deviations of each parameter, with
# theta as the final entry in the list.
print('Estimated parameter standard deviations from GIM: {0}'.format(uncerts))

# For comparison, we can estimate uncertainties with the Fisher Information
# Matrix, which doesn't account for linkage in the data and thus underestimates
# uncertainty. (Although it's a fine approach if you think your data is truly
# unlinked.)
uncerts_fim = moments.Godambe.FIM_uncert(func, popt, data, multinom=True)
print('Estimated parameter standard deviations from FIM: {0}'.format(uncerts_fim))

print('Factors by which FIM underestimates parameter uncertainties: {0}'.format(uncerts/uncerts_fim))

# What if we fold the data?
# These are the optimal parameters when the spectrum is folded. They can be
# found simply by passing data.fold() to the above call to optimize_log. 
popt_fold =  array([1.907,  0.073,  1.830,  0.899,  0.425,  0.113])
uncerts_folded = moments.Godambe.GIM_uncert(func, all_boot, popt_fold,
                                         data.fold(), multinom=True)
print('Folding increases parameter uncertainties by factors of: {0}'.format(uncerts_folded/uncerts))

# Let's do a likelihood-ratio test comparing models with and without migration.
# The no migration model is implemented as 
# demographic_models.prior_onegrow_nomig
func_nomig = demographic_models.prior_onegrow_nomig
# These are the best-fit parameters, which we found by multiple optimizations
popt_nomig = array([ 1.897,  0.0388,  9.677,  0.395,  0.070])
model_nomig = func_nomig(popt_nomig, ns)
ll_nomig = moments.Inference.ll_multinom(model_nomig, data)

# Since LRT evaluates the complex model using the best-fit parameters from the
# simple model, we need to create list of parameters for the complex model
# using the simple (no-mig) best-fit params.  Since evalution is done with more
# complex model, need to insert zero migration value at corresponding migration
# parameter index in complex model. And we need to tell the LRT adjust function
# that the 3rd parameter (counting from 0) is the nested one.
p_lrt = [1.897,  0.0388,  9.677, 0, 0.395,  0.070]

adj = moments.Godambe.LRT_adjust(func, all_boot, p_lrt, data,
                              nested_indices=[3], multinom=True)
D_adj = adj*2*(ll_model - ll_nomig)
print('Adjusted D statistic: {0:.4f}'.format(D_adj))

# Because this is test of a parameter on the boundary of parameter space 
# (m cannot be less than zero), our null distribution is an even proportion 
# of chi^2 distributions with 0 and 1 d.o.f. To evaluate the p-value, we use the
# point percent function for a weighted sum of chi^2 dists.
pval = moments.Godambe.sum_chi2_ppf(D_adj, weights=(0.5,0.5))
print('p-value for rejecting no-migration model: {0:.4f}'.format(pval))

w_adj = moments.Godambe.Wald_stat(func, all_boot, p_lrt, data,
                               nested_indices=[3], full_params=popt, 
                               multinom=True)
print('Adjusted Wald statistic: {0:.4f}'.format(w_adj))
pval = moments.Godambe.sum_chi2_ppf(w_adj, weights=(0.5,0.5))
print('p-value for rejecting no-migration model: {0:.4f}'.format(pval))

score_adj = moments.Godambe.score_stat(func, all_boot, p_lrt, data,
                                    nested_indices=[3], multinom=True)
print('Adjusted score statistic: {0:.4f}'.format(score_adj))
pval = moments.Godambe.sum_chi2_ppf(score_adj, weights=(0.5,0.5))
print('p-value for rejecting no-migration model: {0:.4f}'.format(pval))"""
