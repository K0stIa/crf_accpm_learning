__author__ = 'kostia'

import numpy as np
from scipy import weave
from maxflow.pygraphcut import solve_mincut


class Inference(object):
    def __init__(self):
        pass

    def __call__(self, item, model, loss=None):
        raise NotImplementedError()


class MinCutInference(Inference):
    def __init__(self):
        super(self.__class__, self).__init__()

    def __call__(self, item, model, loss=None):
        unary = -1 * np.dot(model.get_unary().T, item.features.T)
        pairwise = -1 * model.get_pairwise()

        if loss is not None:
            for label in xrange(item.n_classes):
                unary[label, :] -= loss(item.labelling, label)

        return solve_mincut(item.graph.edges, unary, pairwise)

class PixelWiseInference(Inference):
    def __init__(self):
        super(self.__class__, self).__init__()

    def __call__(self, item, model, loss=None):
        unary = -1 * np.dot(model.get_unary().T, item.features.T)
        pairwise = -1 * model.get_pairwise()

        if loss is not None:
            for label in xrange(item.n_classes):
                unary[label, :] -= loss(item.labelling, label)

        labelling = item.labelling
        edges = item.graph.edges
        n_classes = item.n_classes
        n_edges = edges.shape[1]
        cpp_code = """
        for (int e = 0; e < n_edges; ++e) {
            const int u = edges(0, e);
            const int v = edges(1, e);
            const int eid = edges(2, e);
            for (int k = 0; k < n_classes; ++k) {
                unary(k, u) += pairwise(eid, k, labelling(u));
                unary(k, v) += pairwise(eid, labelling(v), k);
            }
        }
        """
        weave.inline(cpp_code, ['edges', 'n_edges', 'n_classes', 'unary', 'pairwise', 'labelling'], \
                     type_converters=weave.converters.blitz, compiler='gcc')

        return unary.argmin(axis=0)