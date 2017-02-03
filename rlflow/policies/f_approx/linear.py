
import tensorflow as tf
from .f_approximator import FunctionApproximator

class LinearApproximator(FunctionApproximator):

    def __init__(self, linear, pol_type):
        super(LinearApproximator, self).__init__(linear,
                                                 pol_type)


    def predict(self, x):
        return self.sess.run(self.prediction_model, feed_dict={self.input_tensor: x.reshape(1, len(x))})
