from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from .f_approximator import FunctionApproximator

class Network(FunctionApproximator):

    def __init__(self, inputs, model, pol_type):
        super(Network, self).__init__(inputs,
                                      model,
                                      pol_type)


    def predict(self, x):
        x = x[np.newaxis,:]
        return self.sess.run(self.prediction, feed_dict={self.inputs[0]: x})
