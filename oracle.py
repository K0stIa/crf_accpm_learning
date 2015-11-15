__author__ = 'kostia'

import numpy as np


class Oracle(object):
    def __init__(self, param_dim, inference_solver, data, model, loss):
        self.data = data
        self.model = model
        self.loss = loss
        self.dimm = param_dim
        self.inference_solver = inference_solver

    def __call(self, weights):
        raise NotImplementedError()


class SupervisedOracle(Oracle):
    def __init__(self, param_dim, inference_solver, data, model, loss):
        super(self.__class__, self).__init__(param_dim, inference_solver, data, model, loss)

    def __call__(self, weights):
        self.model.set_weights(weights)

        obj = 0
        unary_psi = np.zeros(self.model.get_unary_shape())
        pairwise_psi = np.zeros(self.model.get_pairwise_shape())

        for item in self.data:
            pred_labelling = self.inference_solver(item, self.model, self.loss)
            obj += np.sum(self.loss(pred_labelling, item.labelling))
            (u, p) = item.get_labeling_psi(pred_labelling)
            unary_psi += u
            pairwise_psi += p
            (u, p) = item.get_labeling_psi(item.labelling)
            unary_psi -= u
            pairwise_psi -= p

        self.model.set_potentials(unary_psi, pairwise_psi)
        subgradient = self.model.get_weights()
        n = len(self.data)

        obj += (subgradient * weights).sum()

        return obj / n, subgradient / n

class PixelWiseSuperwisedOracle(Oracle):
    def __init__(self, param_dim, inference_solver, data, model, loss):
        super(self.__class__, self).__init__(param_dim, inference_solver, data, model, loss)

    def __call__(self, weights):
        self.model.set_weights(weights)

        obj = 0
        unary_psi = np.zeros(self.model.get_unary_shape())
        pairwise_psi = np.zeros(self.model.get_pairwise_shape())

        for item in self.data:
            pred_labelling = self.inference_solver(item, self.model, self.loss)
            obj += np.sum(self.loss(pred_labelling, item.labelling))
            (u, p) = self.get_psi(pred_labelling, item)
            unary_psi += u
            pairwise_psi += p
            (u, p) = self.get_psi(item.labelling, item)
            unary_psi -= u
            pairwise_psi -= p

        self.model.set_potentials(unary_psi, pairwise_psi)
        subgradient = self.model.get_weights()
        n = len(self.data)

        obj += (subgradient * weights).sum()

        return obj / n, subgradient / n

    def get_psi(self, prediected_labelling, item):
        n_classes = item.n_classes
        edge_id = item.graph.edges[2, :]
        edge = item.graph.edges[:2, :]
        n_edge_cliques = item.graph.n_edge_cliques
        pair_wise = np.zeros((n_edge_cliques, n_classes, n_classes))
        for cl in xrange(n_edge_cliques):
            for k1 in xrange(n_classes):
                for k2 in xrange(n_classes):
                    edge_idx = np.nonzero(edge_id == cl)[0]
                    if edge_idx.size == 0:
                        continue
                    labeling_mask = (prediected_labelling[edge[0, edge_idx]] == k1) & \
                                    (item.labelling[edge[1, edge_idx]] == k2)
                    pair_wise[cl, k1, k2] += np.sum(labeling_mask)

                    labeling_mask = (item.labelling[edge[0, edge_idx]] == k1) & \
                                    (prediected_labelling[edge[1, edge_idx]] == k2)
                    pair_wise[cl, k1, k2] += np.sum(labeling_mask)

        features = item.features
        n_examples, feat_dim = features.shape
        unary = np.zeros((feat_dim, n_classes))
        for k in xrange(n_classes):
            idx = np.nonzero(prediected_labelling == k)[0]
            if idx.size == 0:
                continue
            unary[:, k] += features[idx, :].sum(axis=0)

        return unary, pair_wise