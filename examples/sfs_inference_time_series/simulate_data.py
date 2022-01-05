import numpy as np
import pickle
import msprime
import tskit
import demes
import moments


def build_model():
    # builds the demes model, with start parameters
    b = demes.Builder()
    b.add_deme(
        name="X",
        epochs=[
            dict(start_size=50000, end_time=4000),
            dict(start_size=30000, end_time=3000),
            dict(start_size=100000, end_time=0),
        ],
    )
    g = b.resolve()
    return g


def run_model(demog, sample_sizes, sample_times, L=1000000, r=1e-8, u=1e-8, nreps=500):
    sample_sets = [
        msprime.SampleSet(n, time=t) for n, t in zip(sample_sizes, sample_times)
    ]
    ts_reps = msprime.sim_ancestry(
        samples=sample_sets,
        demography=demog,
        sequence_length=L,
        recombination_rate=r,
        num_replicates=nreps,
        discrete_genome=False,
    )
    sfs_samples = [
        range(
            2 * sum(sample_sizes[:i]), 2 * sum(sample_sizes[:i]) + 2 * sample_sizes[i]
        )
        for i in range(len(sample_sizes))
    ]
    spectra = {}
    for ii, ts in enumerate(ts_reps):
        ts = msprime.sim_mutations(ts, rate=u, discrete_genome=False)
        spectra[ii] = ts.allele_frequency_spectrum(
            sample_sets=sfs_samples, span_normalise=False, polarised=True
        )
    return spectra


if __name__ == "__main__":
    g = build_model()

    demog = msprime.Demography.from_demes(g)
    sample_sizes = [10, 8, 40]
    sample_times = [4100, 2900, 0]

    u = 1e-8
    L = 1000000
    nreps = 500

    print("simulating data, nreps =", nreps)
    try:
        data = moments.Spectrum.from_file("data.fs")
        spectra = pickle.load(open("rep_spectra.bp", "rb"))
    except IOError:
        spectra = run_model(demog, sample_sizes, sample_times, L=L, u=u, nreps=nreps)
        pickle.dump(spectra, open("rep_spectra.bp", "wb+"))
        data = sum(spectra.values())

        data = moments.Spectrum(data)
        data.pop_ids = moments.Demes.SFS(
            g,
            sample_sizes=sample_sizes,
            sampled_demes=["X", "X", "X"],
            sample_times=sample_times,
        ).pop_ids
        data.to_file("data.fs")

    ## reinfer population sizes, knowing the sampling times
    options = "options.yaml"
    graph_init = "model.yaml"

    uL = nreps * L * u
    
    print("running inference")
    ret = moments.Demes.Inference.optimize(
        graph_init,
        options,
        data,
        perturb=1,
        uL=uL,
        method="fmin",
        output="fit_model.yaml",
    )

    print("LL:", -ret[2])
    print("param\tvalue")
    for p, v in zip(ret[0], ret[1]):
        print(f"{p}\t{int(v):,}")

    ## use replicates to create bootstrap data and compute SEs
    bs_data = []
    for _ in range(nreps):
        choices = np.random.randint(500, size=500)
        bs_data.append(moments.Spectrum(sum([spectra[i] for i in choices])))

    print("running uncert analysis")
    uncerts = moments.Demes.Inference.uncerts(
        "fit_model.yaml",
        options,
        data,
        bootstraps=bs_data,
        uL=uL,
        method="GIM",
    )
    
    print("param\tvalue\tstd err")
    for p, v, u in zip(ret[0], ret[1], uncerts):
        print(f"{p}\t{int(v):,}\t{int(u):,}")
