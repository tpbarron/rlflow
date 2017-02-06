
import tensorflow as tf
from .f_approximator import FunctionApproximator

class LinearApproximator(FunctionApproximator):

    def __init__(self, inputs, model, pol_type):
        super(LinearApproximator, self).__init__(inputs,
                                                 model,
                                                 pol_type)


    def predict(self, x):
        return self.sess.run(self.prediction, feed_dict={self.inputs[0]: x.reshape(1, len(x))})
