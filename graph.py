__author__ = 'kostia'

import numpy as np


class Graph(object):
    def __init__(self, shape):
        self.edges, self.n_vertex_cliques, self.n_edge_cliques = \
            self._build(shape)

    def _build(self, shape):
        raise NotImplementedError()


class GridGraph(Graph):
    def __init__(self, shape):
        super(self.__class__, self).__init__(shape)

    def _build(self, shape):
        if len(shape) > 2:
            shape = shape[:2]
        h, w = shape
        edges = []
        for i in xrange(h):
            for j in xrange(w):
                if i + 1 < h:
                    edges.append([i * w + j, (i + 1) * w + j, 0])
                if j + 1 < w:
                    edges.append([i * w + j, i * w + j + 1, 1])

        edges = np.ascontiguousarray(np.array(edges, dtype=np.int32).T)
        n_edge_cliques = 2
        n_vertex_cliques = 1

        return edges, n_vertex_cliques, n_edge_cliques