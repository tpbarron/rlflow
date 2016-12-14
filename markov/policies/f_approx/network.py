from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from .f_approximator import FunctionApproximator

class Network(FunctionApproximator):

    def __init__(self, input_tensor, model, session, pol_type, use_clone_net=False):
        super(Network, self).__init__(input_tensor,
                                      model,
                                      session,
                                      pol_type,
                                      use_clone_net)


    def predict(self, x):
        x = x[np.newaxis,:]
        return self.sess.run(self.prediction_model, feed_dict={self.input_tensor: x})
