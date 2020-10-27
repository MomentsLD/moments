# This script takes the n-dimensional SFS output from ANGSD
# and creates and (optionally) saves a moments Spectrum object.
# Requires the ANGSD SFS file, a sample-population file where each line
# has 'sample_id pop_id'. We also need to specify a list of population ids
# in the order that was input to ANGSD. If the ANGSD SFS is folded, specify
# with the --folded flag.
# This script assumes just a single line in the input SFS file, and checks
# that the dimensions match what is expected from the number of samples for
# each population in the population file.

import moments
import numpy as np
import sys
import argparse


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sfs_file", "-i", required=True, help="n-dimensional SFS from ANGSD"
    )
    parser.add_argument(
        "--pop_file",
        "-p",
        type=str,
        required=True,
        help="File with population labels for samples",
    )
    parser.add_argument(
        "--pop_ids",
        type=str,
        nargs="+",
        required=True,
        help="List of populations ids, matching order of ANGSD output",
    )
    parser.add_argument(
        "--folded",
        "-f",
        action="store_false",
        help="Defaults to False. If spectrum is folded, specify --folded",
    )
    parser.add_argument(
        "--out_file",
        "-o",
        type=str,
        required=False,
        help="Saves a moments Spectrum object if file name is specified",
    )
    parser.add_argument(
        "--mask_corners",
        action="store_false",
        help="Specify if the corners (fixed bins) should be unmasked",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If plot flag is given, plots resulting SFS for visual inspections",
    )
    return parser


def set_font_sizes():
    import matplotlib

    # Set fontsize
    matplotlib.rc(
        "font",
        **{
            "family": "sans-serif",
            "sans-serif": ["Helvetica"],
            "style": "normal",
            "size": 8,
        }
    )
    # Set label tick sizes
    matplotlib.rc("xtick", labelsize=7)
    matplotlib.rc("ytick", labelsize=7)
    matplotlib.rc("legend", fontsize=7)
    matplotlib.rc("legend", frameon=False)


def plot_fs(fs):
    """
    Function to plot and show the resulting frequency spectrum.
    By default, this is not done. But not a bad idea to visually inspect
    the resulting frequency spetrum to ensure that everything looks
    resonable.
    """
    assert (
        moments.__version__ >= "1.0.3"
    ), "Plotting function relies on changes made in\
moments version 1.0.3"
    import matplotlib.pylab as plt

    cmap = plt.cm.cubehelix_r
    set_font_sizes()
    # depending on dimension of SFS, plot marginal spectra
    if fs.ndim == 1:
        # single axis to plot 1D sfs
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        moments.Plotting.plot_1d_fs(fs, ax=ax)
        fig.tight_layout()
        plt.show()
    elif fs.ndim == 2:
        # three axes (two marginal SFS and the joint SFS heatmap)
        fig = plt.figure()
        ax1 = plt.subplot(1, 3, 1)
        moments.Plotting.plot_1d_fs(fs.marginalize([1]), ax=ax1)
        ax1.set_title(fs.pop_ids[0])
        ax2 = plt.subplot(2, 3, 2, sharey=ax1)
        moments.Plotting.plot_1d_fs(fs.marginalize([0]), ax=ax2)
        ax2.set_title(fs.pop_ids[1], fontsize=8)
        # plot heatmap SFS
        ax3 = plt.subplot(2, 3, 3)
        moments.Plotting.plot_single_2d_sfs(fs.marginalize([2]), ax=ax3, cmap=cmap)
        plt.show()
    elif fs.ndim == 3:
        # six axes (three marginal SFS, and three marg joint SFS heatmaps)
        fig = plt.figure(1, figsize=(6, 3))
        ax1 = plt.subplot(2, 3, 1)
        moments.Plotting.plot_1d_fs(fs.marginalize([1, 2]), ax=ax1)
        ax1.set_title(fs.pop_ids[0], fontsize=8)
        ax2 = plt.subplot(2, 3, 2, sharey=ax1)
        moments.Plotting.plot_1d_fs(fs.marginalize([0, 2]), ax=ax2)
        ax2.set_title(fs.pop_ids[1], fontsize=8)
        ax3 = plt.subplot(2, 3, 3, sharey=ax1)
        moments.Plotting.plot_1d_fs(fs.marginalize([0, 1]), ax=ax3)
        ax3.set_title(fs.pop_ids[2], fontsize=8)
        # plot heatmaps of pairwise joint frequency spectra
        joint_margins = [fs.marginalize([i]) for i in range(3)]
        min_ = min(*(m.min() for m in joint_margins))
        max_ = max(*(m.max() for m in joint_margins))
        ax4 = plt.subplot(2, 3, 4)
        moments.Plotting.plot_single_2d_sfs(
            joint_margins[2], ax=ax4, cmap=cmap, vmin=min_, vmax=max_
        )
        ax5 = plt.subplot(2, 3, 5)
        moments.Plotting.plot_single_2d_sfs(
            joint_margins[1], ax=ax5, cmap=cmap, vmin=min_, vmax=max_
        )
        ax6 = plt.subplot(2, 3, 6)
        moments.Plotting.plot_single_2d_sfs(
            joint_margins[0], ax=ax6, cmap=cmap, vmin=min_, vmax=max_
        )
        fig.tight_layout()
        plt.show()
    else:
        print("plotting only available for up to three dimensional sfs")


def get_fs_from_angsd(sfs_file, pop_file, pop_ids, fold):
    """
    Get sample sizes from the pop_file, then import the ANGSD data and reshape
    the spectrum to the correct dimensions.
    """
    pop_ids = args.pop_ids
    ns = [0 for _ in pop_ids]
    with open(args.pop_file) as f_pop:
        for line in f_pop:
            pop_id = line.split()[1]
            if pop_id in pop_ids:
                ns[pop_ids.index(pop_id)] += 2
    # get the SFS data
    with open(args.sfs_file) as f_data:
        data_line = f_data.readline()

    data = np.array([float(d) for d in data_line.split()])

    # check that the length of data matches the number of bins in the SFS
    assert np.prod([n + 1 for n in ns]) == len(data), "data does not match sample sizes"

    data = np.reshape(data, [n + 1 for n in ns])

    # turn into moments object
    fs = moments.Spectrum(data, mask_corners=args.mask_corners, pop_ids=pop_ids)

    if fold is True:
        fs = fs.fold()

    return fs


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    if args.folded is True:
        print(
            "The output SFS is folded, but we assume the input ANGSD file is unfolded"
        )

    fs = get_fs_from_angsd(args.sfs_file, args.pop_file, args.pop_ids, args.folded)

    # save the file if an output filename is given
    if args.out_file is not None:
        fs.to_file(args.out_file)

    # plot marginal frequency spectra for visual inspection
    if args.plot is True:
        plot_fs(fs)
