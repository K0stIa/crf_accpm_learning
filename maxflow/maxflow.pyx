cimport cython
cimport numpy
import numpy
from libcpp.vector cimport vector

cdef extern from "maxflow.cc":
    vector[int] Solve2LabelProblem(double *, const int, double *, const int, int *, const int,);
    vector[int] solveMultiLabelProblem(const int, double*, const int, double *, const int, int *, const int);

# @cython.boundscheck(False)
def solve_mincut(numpy.ndarray[int, mode="c", ndim = 2] edges, numpy.ndarray[double, ndim = 2] unary, numpy.ndarray[double, ndim = 3] pairwise):
    # TODO: check if arrays are C-ordered
    assert edges.shape[0] == 3

    cdef int n_edges = edges.shape[1]
    cdef int n_labels = unary.shape[0]
    cdef int n_unaries = unary.shape[1]

    cdef numpy.ndarray[int, mode="c", ndim=2] c_edges = numpy.ascontiguousarray(edges)
    cdef numpy.ndarray[double, mode="c", ndim=2] c_unary = numpy.ascontiguousarray(unary)
    cdef numpy.ndarray[double, mode="c", ndim=3] c_pairwise = numpy.ascontiguousarray(pairwise)

    assert n_labels == pairwise.shape[1]
    assert n_labels == pairwise.shape[2]

    cdef int n_pairwise_cliques = pairwise.shape[0]

    assert n_labels >= 2

    cdef vector[int] v_labelling

    if n_labels == 2:
        assert False
        v_labelling = Solve2LabelProblem(<double*>c_unary.data, n_unaries, \
            <double*>c_pairwise.data, n_pairwise_cliques, \
            <int*>c_edges.data, n_edges)
    else:
        v_labelling = solveMultiLabelProblem(n_labels, <double*>c_unary.data, n_unaries, \
            <double*>c_pairwise.data, n_pairwise_cliques, \
            <int*>c_edges.data, n_edges)

    cdef numpy.ndarray[int, mode="c", ndim=1] labelling = numpy.asarray(v_labelling).astype(numpy.int32)
    return labelling