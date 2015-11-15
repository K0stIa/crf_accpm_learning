__author__ = 'kostia'

import numpy as np


class FeatureExtractor(object):
    def __init__(self):
        pass

    def extract(self, image, labelling):
        raise NotImplementedError()


class SimpleRgbFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super(self.__class__, self).__init__()

    def extract(self, image, labelling):
        shape = image.shape
        if len(shape) <= 2:
            for i in xrange(0, 3 - shape.size):
                shape = shape + (1,)
        features = image.reshape((shape[0] * shape[1], shape[2]), order='C').astype(np.double)
        labelling = labelling.astype(np.int32).reshape(-1, order='C')
        features -= features.min()
        features /= features.max()

        return features, labelling