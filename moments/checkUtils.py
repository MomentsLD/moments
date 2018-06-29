import numpy as np
import Jackknife as JK
import Spectrum_mod


def get_error_threshold(n):
    """
    Given the length of a 1D AFS, return the acceptable error threshold of jk1 and jk2.
    e.g.
    >>> get_error_threshold(3)
    0.15, 0.35
    Error threshold should be 1e-2, 1e-2 for jk1, jk2.
    """

    jk1_lookup = {1: 0.5, 2: 0.15, 3: 0.004, 4: 6e-5}
    jk2_lookup = {1: 4.5, 2: 0.35, 3: 0.012, 4: 0.00016}

    if n in jk1_lookup:
        return jk1_lookup[n], jk2_lookup[n]
    else:
        return 7.5 * 0.1 ** (n + 2), 2.5 * 0.1 ** (n + 1)


def mre(orig, approx):
    """
    Given the approximated AFS and the original AFS, return the max relative error.
    """
    err = np.abs(orig - approx)/orig
    err = err[err != np.inf]
    return np.nanmax(err)


def check_negative(approx):
    """
    Check number of negative elements in AFS approx.
    """
    neg_count = sum(1 for x in approx if x < 0)
    if neg_count > 0:
        print "Warning: encounter negative elements when integrating!"


def check_relative_error(orig, approx, prev_error, threshold=0.00001):
    """
    Check whether the max relative error in approximated AFS
    is above threshold.
    """
    max_err = mre(orig, approx)
    if max_err > threshold:
        print "Warning: max relative error {} is above threshold {}!".format(str(max_err), str(threshold))

    if max_err > prev_error:
        print "Warning: relative error is increasing!"


def check_jk1(phi, prev_error=100):
    """
    Given a 1D AFS phi, project it to sample n-1 and approximate it use Jackknife.
    Check whether there's negative element and relative error in the recovered AFS.

    :param phi: a Spectrum object
    :param prev_error: max relative error in the previous timestamp
    :return: approximated AFS, of type Spectrum
    """
    n = len(phi) - 1

    projected = phi.project([n - 1])
    J = JK.calcJK13(n - 1)
    approx = J.dot(projected[1:-1])

    check_negative(approx)
    threshold = get_error_threshold(len(str(n)))[0]
    check_relative_error(phi[1:-1], approx, prev_error, threshold=threshold)

    return approx


def check_jk2(phi, prev_error=100):
    """
        Given a 1D AFS phi, project it to sample n-2 and approximate it use Jackknife.
        Check whether there's negative element and relative error in the recovered AFS.

        :param phi: a Spectrum object
        :param prev_error: max relavie error in the previous timestamp
        :return: approximated AFS, of type Spectrum
        """
    n = len(phi) - 1

    projected = phi.project([n - 2])
    J = JK.calcJK23(n - 2)
    approx = J.dot(projected[1:-1])

    check_negative(approx)
    threshold = get_error_threshold(len(str(n)))[1]
    check_relative_error(phi[1:-1], approx, prev_error, threshold=threshold)

    return approx


def check_nD_jk1(sfs):
    """
    Given any dimensional AFS, check whether step 1 Jackknife causes error.
    :param sfs:
    :return:
    """
    shape = sfs.shape
    dim = len(shape)

    # Fix other dimension to check 1D AFS in dimension i
    for i in range(dim):
        shape_cpy = list(shape[:])
        shape_cpy.pop(i)

        # Create all possible combinations of other dimensions
        # e.g. check sfs[:, a, b] for possible combinations of a and b.
        grid_cmd = "np.meshgrid("
        for j in range(len(shape_cpy)):
            grid_cmd += "np.arange(shape_cpy[{}]), ".format(j)

        grid_cmd = grid_cmd[:-2] + ")"
        # print "Command to create grid: " + str(grid_cmd)
        grid = eval(grid_cmd)

        # Convert the grid to tuple
        # e.g. To check sfs[:, 10, 9, 2], generate tuple (10, 9, 2) in this step.
        positions = np.vstack(map(np.ravel, grid))
        for j in range(positions.shape[1]):
            index = list(positions[:,  j])
            index.insert(i, 99)
            slice_cmd = "sfs["
            for k in range(len(index)):
                if k == i:
                    slice_cmd += ":, "
                else:
                    slice_cmd += "{}, ".format(index[k])
            slice_cmd = slice_cmd[:-2] + "]"

            # Slice the sfs to generate a 1D AFS, which can checked by check_jk1.
            phi = eval(slice_cmd)
            check_jk1(Spectrum_mod.Spectrum(phi))


def check_nD_jk2(sfs):
    """
    Given any dimensional AFS, check whether step 2 Jackknife causes error.

    :param sfs:
    :return:
    """
    shape = sfs.shape
    dim = len(shape)

    # Fix other dimension to check 1D AFS in dimension i
    for i in range(dim):
        shape_cpy = list(shape[:])
        shape_cpy.pop(i)

        # Create all possible combinations of other dimensions
        # e.g. check sfs[:, a, b] for possible combinations of a and b.
        grid_cmd = "np.meshgrid("
        for j in range(len(shape_cpy)):
            grid_cmd += "np.arange(shape_cpy[{}]), ".format(j)

        grid_cmd = grid_cmd[:-2] + ")"
        # print "Command to create grid: " + str(grid_cmd)
        grid = eval(grid_cmd)

        # Convert the grid to tuple
        # e.g. To check sfs[:, 10, 9, 2], generate tuple (10, 9, 2) in this step.
        positions = np.vstack(map(np.ravel, grid))
        for j in range(positions.shape[1]):
            index = list(positions[:,  j])
            index.insert(i, 99)
            slice_cmd = "sfs["
            for k in range(len(index)):
                if k == i:
                    slice_cmd += ":, "
                else:
                    slice_cmd += "{}, ".format(index[k])
            slice_cmd = slice_cmd[:-2] + "]"

            # Slice the sfs to generate a 1D AFS, which can checked by check_jk1.
            phi = eval(slice_cmd)
            check_jk2(Spectrum_mod.Spectrum(phi))
