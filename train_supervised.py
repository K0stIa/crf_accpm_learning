__author__ = 'kostia'

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

from optimization.accpm.accpm import accpm
from optimization.pybmrm.bmrm import BMRM, BMRMOptionns
from structured_item import StructuredItem as Item
from feature_extractor import SimpleRgbFeatureExtractor
from graph import GridGraph
from loss import Loss
from model import Model
from oracle import SupervisedOracle, PixelWiseSuperwisedOracle
from inference import MinCutInference, PixelWiseInference


def create_supermodular_constraint_matrix(n_classes):
    a = np.zeros(((n_classes - 1) ** 2, n_classes ** 2), dtype=np.double)
    cnt = 0

    for i in xrange(n_classes - 1):
        for j in xrange(n_classes - 1):
            g = np.zeros((n_classes, n_classes), dtype=np.double)
            g[i, j + 1] = 1
            g[i + 1, j] = 1
            g[i, j] = -1
            g[i + 1, j + 1] = -1
            a[cnt, :] = g.flatten(1)
            cnt += 1
    return a

def get_oracle_submodular_constraints(n_classes, n_pairwise_cliques, feature_dim):
    a = create_supermodular_constraint_matrix(n_classes)
    A = a
    for i in xrange(n_pairwise_cliques - 1):
        A = block_diag(A, a)
    A = np.c_[np.zeros((A.shape[0], feature_dim * n_classes), dtype=np.double), A]
    b = np.zeros((A.shape[0], 1))
    return A, b

M = sio.loadmat('data/mscows_data.mat')
TRN = np.arange(0, 3)
N_CLASSES = 3

#build train data
data = []
feature_dim = None
n_pairwise_cliques = None
for idx in TRN:
    image = M["images"][0, idx]
    image_labelling = M["labelings"][0, idx]
    print "labels:", np.unique(image_labelling)
    features, feature_labels = SimpleRgbFeatureExtractor().extract(image, image_labelling)
    graph = GridGraph(image.shape)
    item = Item(idx, graph, features, feature_labels, N_CLASSES)
    data.append(item)
    feature_dim = features.shape[1]
    n_pairwise_cliques = graph.n_edge_cliques

#numder of parameters to train
param_dim = N_CLASSES * feature_dim + n_pairwise_cliques * N_CLASSES ** 2
# get constraints on submodular part
A, b = get_oracle_submodular_constraints(N_CLASSES, n_pairwise_cliques, feature_dim)

C = np.r_[np.eye(param_dim), -np.eye(param_dim)]
d = 10 * np.ones((2 * param_dim, 1), dtype=np.double)
A = np.r_[A, C]
b = np.r_[b, d]


model = Model(N_CLASSES, feature_dim, 1, n_pairwise_cliques)

inference_solver = MinCutInference()
oracle = SupervisedOracle(param_dim, inference_solver, data, model, Loss())

# inference_solver = PixelWiseInference()
# oracle = PixelWiseSuperwisedOracle(param_dim, inference_solver, data, model, Loss())


import time
tic = time.time()
x_opt, fobj = accpm(oracle, A, b, Lambda=1.0)
# bmrm_options = BMRMOptionns(oracle.dimm, max_cp=750, max_iter=3000)
# bmrm = BMRM(oracle, bmrm_options)
# weights = bmrm.learn(0.1)[0]
print '\ntime %f\n' % (time.time() - tic)