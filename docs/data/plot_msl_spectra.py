import pickle
import matplotlib.pylab as plt

def plot_fs(ax, fs, lw=0.7, ms=3, label=None):
    ax.plot(fs, ".-", lw=lw, ms=2, label=label)

if __name__ == "__main__":
    data = pickle.load(open("msl_data.bp", "rb"))

    fs_syn = data["spectra"]["syn"]
    fs_mis = data["spectra"]["mis"]
    fs_lof = data["spectra"]["lof"]
    fs_lof.mask[fs_lof == 0] = True # for plotting niceness

    fig = plt.figure(figsize=(10, 4))
    
    ax1 = plt.subplot(1, 2, 1)

    plot_fs(ax1, fs_syn, label="Synonymous")
    plot_fs(ax1, fs_mis, label="Missense")
    plot_fs(ax1, fs_lof, label="Loss of function")

    ax1.set_yscale("log")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Derived allele frequency")
    ax1.legend(frameon=False)

    ax1 = plt.subplot(1, 2, 2)
    
    plot_fs(ax1, fs_syn / fs_syn.S(), label="Synonymous")
    plot_fs(ax1, fs_mis / fs_mis.S(), label="Missense")
    plot_fs(ax1, fs_lof / fs_lof.S(), label="Loss of function")
    
    ax1.set_yscale("log")
    ax1.set_ylabel("Proportion")
    ax1.set_xlabel("Derived allele frequency")

    plt.tight_layout()
    plt.savefig("../figures/msl_spectra.png")

