
import tensorflow as tf
from .f_approximator import FunctionApproximator

class LinearApproximator(FunctionApproximator):

    def __init__(self, input_tensor, linear, session, pol_type):
        super(LinearApproximator, self).__init__(input_tensor,
                                                 linear,
                                                 session,
                                                 pol_type)


    def get_num_weights(self):
        return tf.size(self.W).eval()


    def predict(self, x):
        return self.sess.run(self.prediction_model, feed_dict={self.input_tensor: x.reshape(1, len(x))})
