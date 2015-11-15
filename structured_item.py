__author__ = 'kostia'

import numpy as np


class StructuredItem(object):
    def __init__(self, item_id, graph, features, labelling, n_classes):
        """
        """
        self.item_id = item_id
        self.graph = graph
        self.features = features
        self.labelling = labelling
        self.n_classes = n_classes

    def get_graound_true_labeling_psi(self):
        if self.ground_true_psi is None:
            self.ground_true_psi = self.get_labeling_psi(self.labelling)
        return self.ground_true_psi

    def get_labeling_psi(self, labelling):
        n_classes = self.n_classes
        edge_id = self.graph.edges[2, :]
        edge = self.graph.edges[:2, :]
        n_edge_cliques = self.graph.n_edge_cliques
        pair_wise = np.zeros((n_edge_cliques, n_classes, n_classes))
        for cl in xrange(n_edge_cliques):
            for k1 in xrange(n_classes):
                for k2 in xrange(n_classes):
                    edge_idx = np.nonzero(edge_id == cl)[0]
                    if edge_idx.size == 0:
                        continue
                    labeling_mask = (labelling[edge[0, edge_idx]] == k1) & \
                                    (labelling[edge[1, edge_idx]] == k2)
                    pair_wise[cl, k1, k2] += np.sum(labeling_mask)

        features = self.features
        n_examples, feat_dim = features.shape
        unary = np.zeros((feat_dim, n_classes))
        for k in xrange(n_classes):
            idx = np.nonzero(labelling == k)[0]
            if idx.size == 0:
                continue
            unary[:, k] += features[idx, :].sum(axis=0)

        return unary, pair_wise