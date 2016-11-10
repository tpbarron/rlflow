from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from rlcore.policies.f_approx.f_approximator import FunctionApproximator

class Network(FunctionApproximator):

    def __init__(self, keras_model, prediction_preprocessors=None, prediction_postprocessors=None):
        super(Network, self).__init__(keras_model)
        self.prediction_preprocessors = prediction_preprocessors
        self.prediction_postprocessors = prediction_postprocessors


    def predict(self, input):
        input = input[np.newaxis,:]
        return self.model.predict(input) #.flatten()
        # action = np.random.choice(6, 1, p=out/np.sum(out))[0]
        # return action
        # return out


    def update(self, gradient):
        pass
