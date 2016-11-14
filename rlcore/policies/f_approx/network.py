from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from rlcore.policies.f_approx.f_approximator import FunctionApproximator

class Network(FunctionApproximator):

    def __init__(self, keras_model, **kwargs):
        super(Network, self).__init__(keras_model, **kwargs)


    def predict(self, input):
        input = input[np.newaxis,:]
        return self.model.predict(input)
