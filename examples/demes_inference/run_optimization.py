import moments

# A demes graph specified in YAML format
deme_graph = "gutenkunst_ooa.yml"
# A file with options for running inference
options_file = "inference_options.yml"
# The saved SFS, simulated with u*L = 0.36
data_file = "data.uL_0.36.fs"

ret = moments.Demes.Inference.optimize(
    deme_graph, options_file, data_file, perturb=1, maxiter=1000, method="fmin"
)

param_names, opt_params, LL = ret
LL = -LL

print("log-likelihood:", f"{LL:.1f}")
for n, p in zip(param_names, opt_params):
    print(f"{n}\t{p:.3}")

