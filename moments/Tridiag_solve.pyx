import numpy
cimport numpy

#------------------------------------------------------------------
# Solver for tridiagonal systems Ax=b
# This code is adapted from Jonathan Senning's code available here: 
# http://www.cs.gordon.edu/courses/mat342/python/tridiagonal.py
#------------------------------------------------------------------


def mat_to_diag(numpy.ndarray[numpy.float64_t, ndim=2] mat):
    """Extracts the 3 diagonals from a tridiagonal matrix mat
    INPUT: 
        mat         - tridiagonal matrix (numpy array)

    OUTPUT:
        a, d, c     - lists (or numpy arrays) a is the subdiagonal, d the
                      main diagonal and c the superdiagonal.
    """
    cdef int i
    cdef int n = mat.shape[0]
    cdef numpy.ndarray[numpy.float64_t] a = numpy.zeros(n - 1)
    cdef numpy.ndarray[numpy.float64_t] c = numpy.zeros(n - 1)
    cdef numpy.ndarray[numpy.float64_t] d = numpy.zeros(n)
    for i in xrange(n - 1):
        a[i] = mat[i + 1, i]
        c[i] = mat[i, i + 1]
        d[i] = mat[i, i]
    d[n - 1] = mat[n - 1, n - 1]
    return a, d, c

def factor(numpy.ndarray[numpy.float64_t] a, numpy.ndarray[numpy.float64_t] d, 
           numpy.ndarray[numpy.float64_t] c):
    """Performs LU factorization on tridiagonal matrix A

    USAGE:
        factor( a, d, c )

    INPUT:
        a, d, c    - lists or NumPy arrays specifying the diagonals of the
                     tridiagonal matrix A.  a is the subdiagonal with a[0]
                     being the A[1,0] value, d is the main diagonal with
                     d[0] being the A[0,0] value and c is the superdiagonal
                     with c[0] being the A[0,1] value.

    OUTPUT:
        a, d, c    - arrays containing the data for the factored matrix

    NOTE:
        For this to be sure to work A should be strictly diagonally
        dominant, meaning that |d(i)| > |a(i-1)| + |c(i)| for each i.
        This ensures that pivoting will not be necessary.
    """
    cdef int i
    cdef int n = len(d)
    for i in xrange(1, n):
        a[i - 1] = a[i - 1] / d[i - 1]
        d[i] = d[i] - a[i - 1] * c[i - 1]

    return

cpdef numpy.ndarray[numpy.float64_t] solve(numpy.ndarray[numpy.float64_t] a, numpy.ndarray[numpy.float64_t] d,
                          numpy.ndarray[numpy.float64_t] c, numpy.ndarray b):
    """Solves Ax=b for x with factored tridigonal A having diagonals a, d, c

    USAGE:
        x = solve( a, d, c, b )

    INPUT:
        a, d, c    - lists or NumPy arrays specifying the diagonals of the
                     factored tridiagonal matrix A.  These are produced by
                     factor().
        b          - right-hand-side vector

    OUTPUT:
        x          - float list: solution vector
    """
    cdef int i
    cdef int n = len(d)

    cdef numpy.ndarray[numpy.float64_t] x = numpy.zeros(n)
    x[0] = b[0]

    for i in xrange(1, n):
        x[i] = b[i] - a[i - 1] * x[i - 1]

    x[n - 1] = x[n - 1] / d[n - 1]

    for i in xrange(n-2, -1, -1):
        x[i] = (x[i] - c[i] * x[i + 1]) / d[i]

    return x