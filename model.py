__author__ = 'kostia'

import numpy as np

class Model(object):
    def __init__(self, n_classes, feat_dim, n_vertex_cliques, n_pairwise_cliques):
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.n_vertex_cliques = n_vertex_cliques
        self.n_pairwise_cliques = n_pairwise_cliques
        self.weights = None
        self.unary = None
        self.pairwise = None

    def set_weights(self, weights):
        self.weights = weights
        self.unary = None
        self.pairwise = None

    def get_unary(self):
        if self.unary is None:
            unary_shape = self.get_unary_shape()
            unary_size = self.feat_dim * self.n_classes
            self.unary = self.weights[:unary_size, :].reshape(unary_shape, order='C')
        return self.unary

    def get_pairwise(self):
        if self.pairwise is None:
            pairwise_shape = self.get_pairwise_shape()
            unary_size = self.feat_dim * self.n_classes
            self.pairwise = self.weights[unary_size:, :].reshape(pairwise_shape, order='C')
        return self.pairwise

    def set_potentials(self, unary, pairwise):
        self.pairwise = pairwise
        self.unary = unary
        self.weights = np.zeros(unary.size + pairwise.size)
        self.weights[:unary.size] = unary.reshape(-1, order='C')
        self.weights[unary.size:] = pairwise.reshape(-1, order='C')
        self.weights = self.weights.reshape((-1, 1), order='C')

    def get_weights(self):
        return self.weights

    def get_unary_shape(self):
        return self.feat_dim, self.n_classes

    def get_pairwise_shape(self):
        return self.n_pairwise_cliques, self.n_classes, self.n_classes