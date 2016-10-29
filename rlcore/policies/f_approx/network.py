from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from rlcore.policies.f_approx.f_approximator import FunctionApproximator

class Network(FunctionApproximator):

    def __init__(self, keras_model, prediction_preprocessors=None, prediction_postprocessors=None):
        self.keras_model = keras_model
        self.prediction_preprocessors = prediction_preprocessors
        self.prediction_postprocessors = prediction_postprocessors


    def predict(self, input):
        input = input[np.newaxis,:]
        out = self.keras_model.predict(input)
        return out


    def update(self, gradient):
        pass
