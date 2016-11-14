import sys
import numpy as np
import tensorflow as tf
import rlcore.core.tensorflow_utils as tf_utils
from .f_approximator import FunctionApproximator

class LinearApproximator(FunctionApproximator):

    def __init__(self, in_d, out_d, **kwargs):
        self.in_d = in_d # input dim
        self.out_d = out_d # output dim

        self.W = tf.Variable(tf.random_normal([out_d, in_d]), name="W")
        self.obs = tf.placeholder(tf.float32, [in_d, 1], name="obs")
        self.f_predict = tf.mul(self.W, self.obs)

        super(LinearApproximator, self).__init__(self.f_predict, **kwargs)

        self.init = tf.initialize_all_variables()
        self.init.run()


    def get_num_weights(self):
        return tf.size(self.W).eval()

    #
    # def update(self, gradient):
    #     self.f_weight_update.eval(feed_dict={self.grad: gradient.reshape(self.grad.get_shape())})


    def predict(self, input):
        return self.f_predict.eval(feed_dict={self.obs: input.reshape(self.in_d,1)})
