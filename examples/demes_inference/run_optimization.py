import moments

deme_graph = "gutenkunst_ooa.yml"
options_file = "inference_options.yml"
data_file = "data.uL_0.36.fs"

ret = moments.Demes.Inference.optimize(
    deme_graph, options_file, data_file, perturb=1, maxiter=10000, method="fmin"
)

param_names, opt_params, LL = ret
LL = -LL

print("log-likelihood:", f"{LL:.1f}")
for n, p in zip(param_names, opt_params):
    print(f"{n}\t{p:.3}")

