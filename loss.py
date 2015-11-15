__author__ = 'kostia'

import numpy as np

class Loss(object):
    def __init__(self):
        pass

    def __call__(self, labelling1, labelling2):
        return (labelling1 != labelling2).astype(np.double) / labelling1.size