"""
Numerically useful functions, including extrapolation and default grid.
"""
import logging

logger = logging.getLogger("Numerics")

import functools, os
import numpy
from scipy.special import gammaln


def reverse_array(arr):
    """
    Reverse an array along all axes, so arr[i,j] -> arr[-(i+1),-(j+1)].
    """
    reverse_slice = tuple([slice(None, None, -1) for ii in arr.shape])
    return arr[reverse_slice]


def intersect_masks(m1, m2):
    """
    Versions of m1 and m2 that are masked where either m1 or m2 were masked.

    If neither m1 or m2 is masked, just returns m1 and m2. Otherwise returns
    m1 and m2 wrapped as masked_arrays with identical masks.
    """
    ma = numpy.ma
    if ma.isMaskedArray(m1) and ma.isMaskedArray(m2) and numpy.all(m1.mask == m2.mask):
        return m1, m2

    if ma.isMaskedArray(m1) or ma.isMaskedArray(m2):
        joint_mask = ma.mask_or(ma.getmask(m1), ma.getmask(m2))

        import moments

        m1 = moments.Spectrum(m1, mask=joint_mask.copy())
        m2 = moments.Spectrum(m2, mask=joint_mask.copy())
    return m1, m2


def compute_N_effective(f, t0, t1):
    """
    Function to compute the effective population sizes considering drift
    between 2 time steps.

    N is a function of time (1 scalar argument), it can be multi-dimensions.

    t0 < t1 are scalars giving the time bounds of the integration step.
    """
    # Number of points for the integration.
    nb_pts = 10
    step = float(t1 - t0) / nb_pts
    values = numpy.array(
        [1.0 / numpy.array(f(t0 + i * step)) for i in range(nb_pts + 1)]
    )
    res = numpy.sum(0.5 * (values[0:-1] + values[1::]) * step, axis=0)
    return (t1 - t0) / res


def trapz(yy, xx=None, dx=None, axis=-1):
    """
    Integrate yy(xx) along given axis using the composite trapezoidal rule.

    xx must be one-dimensional and len(xx) must equal yy.shape[axis].

    This is modified from the SciPy version to work with n-D yy and 1-D xx.
    """
    if (xx is None and dx is None) or (xx is not None and dx is not None):
        raise ValueError("One and only one of xx or dx must be specified.")
    elif (xx is not None) and (dx is None):
        dx = numpy.diff(xx)
    yy = numpy.asanyarray(yy)
    nd = yy.ndim

    if yy.shape[axis] != (len(dx) + 1):
        raise ValueError(
            "Length of xx must be equal to length of yy along "
            "specified axis. Here len(xx) = %i and "
            "yy.shape[axis] = %i." % (len(dx) + 1, yy.shape[axis])
        )

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    sliceX = [numpy.newaxis] * nd
    sliceX[axis] = slice(None)

    return numpy.sum(dx[sliceX] * (yy[slice1] + yy[slice2]) / 2.0, axis=axis)


_projection_cache = {}


def _lncomb(N, k):
    """
    Log of N choose k.
    """
    return gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)


def _cached_projection(proj_to, proj_from, hits):
    """
    Coefficients for projection from a different fs size.

    proj_to: Numper of samples to project down to.
    proj_from: Numper of samples to project from.
    hits: Number of derived alleles projecting from.
    """
    key = (proj_to, proj_from, hits)
    try:
        return _projection_cache[key]
    except KeyError:
        pass

    if numpy.isscalar(proj_to) and numpy.isscalar(proj_from) and proj_from < proj_to:
        # Short-circuit calculation.
        contrib = numpy.zeros(proj_to + 1)
    else:
        # We set numpy's error reporting so that it will ignore underflows,
        # because those just imply that contrib is 0.
        previous_err_state = numpy.seterr(
            under="ignore", divide="raise", over="raise", invalid="raise"
        )
        proj_hits = numpy.arange(proj_to + 1)
        # For large sample sizes, we need to do the calculation in logs, and it
        # is accurate enough for small sizes as well.
        lncontrib = _lncomb(proj_to, proj_hits)
        lncontrib += _lncomb(proj_from - proj_to, hits - proj_hits)
        lncontrib -= _lncomb(proj_from, hits)
        contrib = numpy.exp(lncontrib)
        numpy.seterr(**previous_err_state)
    _projection_cache[key] = contrib
    return contrib


def array_from_file(fid, return_comments=False):
    """
    Read array from file.

    fid: string with file name to read from or an open file object.
    return_comments: If True, the return value is (fs, comments), where
                     comments is a list of strings containing the comments
                     from the file (without #'s).

    The file format is:
        # Any number of comment lines beginning with a '#'
        A single line containing N integers giving the dimensions of the fs
          array. So this line would be '5 5 3' for an SFS that was 5x5x3.
          (That would be 4x4x2 *samples*.)
        A single line giving the array elements. The order of elements is
          e.g.: fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1] ...
    """
    newfile = False
    # Try to read from fid. If we can't, assume it's something that we can
    # use to open a file.
    if not hasattr(fid, "read"):
        newfile = True
        fid = open(fid, "r")

    line = fid.readline()
    # Strip out the comments
    comments = []
    while line.startswith("#"):
        comments.append(line[1:].strip())
        line = fid.readline()

    # Read the shape of the data
    shape = tuple([int(d) for d in line.split()])

    data = numpy.fromfile(fid, count=numpy.prod(shape), sep=" ")
    # fromfile returns a 1-d array. Reshape it to the proper form.
    data = data.reshape(*shape)

    # If we opened a new file, clean it up.
    if newfile:
        fid.close()

    if not return_comments:
        return data
    else:
        return data, comments


def array_to_file(data, fid, precision=16, comment_lines=[]):
    """
    Write array to file.

    data: array to write
    fid: string with file name to write to or an open file object.
    precision: precision with which to write out entries of the SFS. (They
               are formated via %.<p>g, where <p> is the precision.)
    comment lines: list of strings to be used as comment lines in the header
                   of the output file.

    The file format is:
        # Any number of comment lines beginning with a '#'
        A single line containing N integers giving the dimensions of the fs
          array. So this line would be '5 5 3' for an SFS that was 5x5x3.
          (That would be 4x4x2 *samples*.)
        A single line giving the array elements. The order of elements is
          e.g.: fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1] ...
    """
    # Open the file object.
    newfile = False
    if not hasattr(fid, "write"):
        newfile = True
        fid = open(fid, "w")

    # Write comments
    for line in comment_lines:
        fid.write("# ")
        fid.write(line.strip())
        fid.write(os.linesep)

    # Write out the shape of the fs
    for elem in data.shape:
        fid.write("%i " % elem)
    fid.write(os.linesep)

    if hasattr(data, "filled"):
        # Masked entries in the fs will go in as 'nan'
        data = data.filled()
    # Write to file
    data.tofile(fid, " ", "%%.%ig" % precision)
    fid.write(os.linesep)

    # Close file
    if newfile:
        fid.close()


def check_function_regularity(function, t):
    """
    Method to check the regularity and monotony of a function.
    May be usefull to warn the user if the population size evolves
    to fast or with monotony changes.

    function: callable scalar function with taking 1 scalar argument.
    t: scalar, final simulation time.
    """
    assert callable(function)
    nb_eval = 1000
    dt = float(t) / (nb_eval - 1)
    ls = numpy.linspace(0, t, nb_eval)
    # array of increments
    relative_increase = numpy.array(
        [(function(ls[i + 1]) - function(ls[i])) / dt for i in range(nb_eval - 1)]
    )

    if (relative_increase > 0).any() and (relative_increase < 0).any():
        logger.warn("Warning: non monotonic function!")

    if (abs(relative_increase) > 5).any():
        logger.warn("Warning: rapid changes in your function!")
