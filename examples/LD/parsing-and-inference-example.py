"""
This script uses msprime to simulate under an isolation with migration model,
writing the outputs to VCF. We'll simulate a small dataset: 100 x 1Mb regions,
each with recombination and mutation rates of 1.5e-8. We'll then use moments
to compute LD statistics from each of the 100 replicates to compute statistic
means and variances/covariances. These are then used to refit the simulated
model using moments.LD, and then we use bootstrapped datasets to estimate
confidence intervals.

The demographic model is a population of size 10,000 that splits into a
population of size 2,000 and a population of size 20,000. The split occurs
1,500 generations ago followed by symmetric migration at rate 1e-4.
"""

import os
import time
import gzip
import numpy as np
import pickle
import msprime
import moments
import demes

assert msprime.__version__ >= "1"

if not os.path.isdir("./data/"):
    os.makedirs("./data/")
os.system("rm ./data/*.vcf.gz")
os.system("rm ./data/*.h5")


def demographic_model():
    b = demes.Builder()
    b.add_deme("anc", epochs=[dict(start_size=10000, end_time=1500)])
    b.add_deme("deme0", ancestors=["anc"], epochs=[dict(start_size=2000)])
    b.add_deme("deme1", ancestors=["anc"], epochs=[dict(start_size=20000)])
    b.add_migration(demes=["deme0", "deme1"], rate=1e-4)
    g = b.resolve()
    return g


def run_msprime_replicates(num_reps=100, L=5000000, u=1.5e-8, r=1.5e-8, n=10):
    g = demographic_model()
    demog = msprime.Demography.from_demes(g)
    tree_sequences = msprime.sim_ancestry(
        {"deme0": n, "deme1": n},
        demography=demog,
        sequence_length=L,
        recombination_rate=r,
        num_replicates=num_reps,
        random_seed=42,
    )
    for ii, ts in enumerate(tree_sequences):
        ts = msprime.sim_mutations(ts, rate=u, random_seed=ii + 1)
        vcf_name = "./data/split_mig.{0}.vcf".format(ii)
        with open(vcf_name, "w+") as fout:
            ts.write_vcf(fout, allow_position_zero=True)
        os.system(f"gzip {vcf_name}")


def write_samples_and_rec_map(L=5000000, r=1.5e-8, n=10):
    # samples file
    with open("./data/samples.txt", "w+") as fout:
        fout.write("sample\tpop\n")
        for jj in range(2):
            for ii in range(n):
                fout.write(f"tsk_{jj * n + ii}\tdeme{jj}\n")
    # recombination map
    with open("./data/flat_map.txt", "w+") as fout:
        fout.write("pos\tMap(cM)\n")
        fout.write("0\t0\n")
        fout.write(f"{L}\t{r * L * 100}\n")


def get_LD_stats(rep_ii, r_bins):
    vcf_file = f"./data/split_mig.{ii}.vcf.gz"
    time1 = time.time()
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_file,
        rec_map_file="./data/flat_map.txt",
        pop_file="./data/samples.txt",
        pops=["deme0", "deme1"],
        r_bins=r_bins,
        report=False,
    )
    time2 = time.time()
    print("  finished rep", ii, "in", int(time2 - time1), "seconds")
    return ld_stats


if __name__ == "__main__":
    num_reps = 100
    # define the bin edges
    r_bins = np.concatenate(([0], np.logspace(-6, -3, 16)))

    try:
        print("loading data if pre-computed")
        with open(f"./data/means.varcovs.split_mig.{num_reps}_reps.bp", "rb") as fin:
            mv = pickle.load(fin)
        with open(f"./data/bootstrap_sets.split_mig.{num_reps}_reps.bp", "rb") as fin:
            all_boot = pickle.load(fin)
    except IOError:
        print("running msprime and writing vcfs")
        run_msprime_replicates(num_reps=num_reps)

        print("writing samples and recombination map")
        write_samples_and_rec_map()

        print("parsing LD statistics")
        # Note: I usually would do this in parallel on cluster - is the slowest step
        ld_stats = {}
        for ii in range(num_reps):
            ld_stats[ii] = get_LD_stats(ii, r_bins)

        print("computing mean and varcov matrix from LD statistics sums")
        mv = moments.LD.Parsing.bootstrap_data(ld_stats)
        with open(f"./data/means.varcovs.split_mig.{num_reps}_reps.bp", "wb+") as fout:
            pickle.dump(mv, fout)
        print(
            "computing bootstrap replicates of mean statistics (for confidence intervals"
        )
        all_boot = moments.LD.Parsing.get_bootstrap_sets(ld_stats)
        with open(f"./data/bootstrap_sets.split_mig.{num_reps}_reps.bp", "wb+") as fout:
            pickle.dump(all_boot, fout)
        os.system("rm ./data/*.vcf.gz")
        os.system("rm ./data/*.h5")

    print("computing expectations under the model")
    g = demographic_model()
    y = moments.Demes.LD(g, sampled_demes=["deme0", "deme1"], rho=4 * 10000 * r_bins)
    y = moments.LD.LDstats(
        [(y_l + y_r) / 2 for y_l, y_r in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops,
        pop_ids=y.pop_ids,
    )
    y = moments.LD.Inference.sigmaD2(y)

    # plot simulated data vs expectations under the model
    fig = moments.LD.Plotting.plot_ld_curves_comp(
        y,
        mv["means"][:-1],
        mv["varcovs"][:-1],
        rs=r_bins,
        stats_to_plot=[
            ["DD_0_0"],
            ["DD_0_1"],
            ["DD_1_1"],
            ["Dz_0_0_0"],
            ["Dz_0_1_1"],
            ["Dz_1_1_1"],
            ["pi2_0_0_1_1"],
            ["pi2_0_1_0_1"],
            ["pi2_1_1_1_1"],
        ],
        labels=[
            [r"$D_0^2$"],
            [r"$D_0 D_1$"],
            [r"$D_1^2$"],
            [r"$Dz_{0,0,0}$"],
            [r"$Dz_{0,1,1}$"],
            [r"$Dz_{1,1,1}$"],
            [r"$\pi_{2;0,0,1,1}$"],
            [r"$\pi_{2;0,1,0,1}$"],
            [r"$\pi_{2;1,1,1,1}$"],
        ],
        #statistics=stats,
        rows=3,
        plot_vcs=True,
        show=False,
        fig_size=(6, 4),
        output="split_mig_comparison.pdf",
    )

    print("running inference")
    # Run inference using the parsed data
    demo_func = moments.LD.Demographics2D.split_mig
    # Set up the initial guess
    # The split_mig function takes four parameters (nu0, nu1, T, m), and we append
    # the last parameter to fit Ne, which doesn't get passed to the function but
    # scales recombination rates so can be simultaneously fit
    p_guess = [0.1, 2, 0.075, 2, 10000]
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)

    opt_params, LL = moments.LD.Inference.optimize_log_fmin(
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins
    )

    physical_units = moments.LD.Util.rescale_params(
        opt_params, ["nu", "nu", "T", "m", "Ne"]
    )

    print("Simulated parameters:")
    print(f"  N(deme0)         :  {g.demes[1].epochs[0].start_size:.1f}")
    print(f"  N(deme1)         :  {g.demes[2].epochs[0].start_size:.1f}")
    print(f"  Div. time (gen)  :  {g.demes[1].epochs[0].start_time:.1f}")
    print(f"  Migration rate   :  {g.migrations[0].rate:.6f}")
    print(f"  N(ancestral)     :  {g.demes[0].epochs[0].start_size:.1f}")

    print("best fit parameters:")
    print(f"  N(deme0)         :  {physical_units[0]:.1f}")
    print(f"  N(deme1)         :  {physical_units[1]:.1f}")
    print(f"  Div. time (gen)  :  {physical_units[2]:.1f}")
    print(f"  Migration rate   :  {physical_units[3]:.6f}")
    print(f"  N(ancestral)     :  {physical_units[4]:.1f}")

    print("computing confidence intervals for parameters")

    # This is somewhat ugly and poor API in moments, and we prefer to use
    # demes for inference and CI calculations in any case. But this at
    # least gets this example to work.
    means, varcovs = moments.LD.Inference.remove_normalized_data(
        mv["means"], mv["varcovs"], num_pops=2
    )
    stats = mv["stats"]
    stats[0].pop(stats[0].index("pi2_0_0_0_0"))
    stats[1].pop(stats[1].index("H_0_0"))

    uncerts = moments.LD.Godambe.GIM_uncert(
        demo_func,
        all_boot,
        opt_params,
        means,
        varcovs,
        r_edges=r_bins,
        statistics=stats,
    )

    lower = opt_params - 1.96 * uncerts
    upper = opt_params + 1.96 * uncerts

    lower_pu = moments.LD.Util.rescale_params(lower, ["nu", "nu", "T", "m", "Ne"])
    upper_pu = moments.LD.Util.rescale_params(upper, ["nu", "nu", "T", "m", "Ne"])

    print("95% CIs:")
    print(f"  N(deme0)         :  {lower_pu[0]:.1f} - {upper_pu[0]:.1f}")
    print(f"  N(deme1)         :  {lower_pu[1]:.1f} - {upper_pu[1]:.1f}")
    print(f"  Div. time (gen)  :  {lower_pu[2]:.1f} - {upper_pu[2]:.1f}")
    print(f"  Migration rate   :  {lower_pu[3]:.6f} - {upper_pu[3]:.6f}")
    print(f"  N(ancestral)     :  {lower_pu[4]:.1f} - {upper_pu[4]:.1f}")

    ## Below shows an example using a LRT, with the test statistic adjusted
    ## using bootstrapped data
    def IM_nomig(params, rho=None, theta=0.001, pop_ids=None):
        full_params = np.concatenate((params, [0]))
        return moments.LD.Demographics2D.split_mig(
            full_params, rho=rho, theta=theta, pop_ids=pop_ids)

    p_guess_nomig = [1, 1, 0.1, 10000] # nu1, nu2, T, Ne
    p_guess_nomig = moments.LD.Util.perturb_params(p_guess_nomig)

    opt_params_nomig, LL_nomig = moments.LD.Inference.optimize_log_lbfgsb(
        p_guess_nomig, [mv["means"], mv["varcovs"]], [IM_nomig], rs=r_bins
    )

    physical_units = moments.LD.Util.rescale_params(
        opt_params_nomig, ["nu", "nu", "T", "Ne"]
    )

    print("best fit parameters:")
    print(f"  N(deme0)         :  {physical_units[0]:.1f}")
    print(f"  N(deme1)         :  {physical_units[1]:.1f}")
    print(f"  Div. time (gen)  :  {physical_units[2]:.1f}")
    print(f"  N(ancestral)     :  {physical_units[3]:.1f}")

    p0 = np.concatenate((opt_params_nomig[:-1], [0], [opt_params_nomig[-1]]))
    adj = moments.LD.Godambe.LRT_adjust(
        demo_func,
        all_boot,
        p0,
        [3],
        means,
        varcovs,
        r_edges=r_bins,
        statistics=stats
    )

    D = LL_nomig - LL
    D_adj = D * adj

    p_val_nomig = moments.Godambe.sum_chi2_ppf(D_adj, weights=(0.5, 0.5))
    
    print()
    print("Simple model with no migration")
    print("Log-likelihoods:")
    print(f"  True model      :", LL)
    print(f"  No migration    :", LL_nomig)
    print("Test statistics:")
    print(f"  No adjustment   :", D)
    print(f"  Bootstrap adj   :", D_adj)
    print(f"P value           :", p_val_nomig)

