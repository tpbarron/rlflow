from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from .f_approximator import FunctionApproximator

class Network(FunctionApproximator):

    def __init__(self, inputs, outputs, scope=None):
        super(Network, self).__init__(inputs, outputs, scope=scope)


    def predict(self, x):
        x = x[np.newaxis,:]
        # return self.model.predict(x)
        return self.sess.run(self.outputs[0], feed_dict={self.inputs[0]: x})
