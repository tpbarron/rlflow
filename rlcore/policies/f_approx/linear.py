
import tensorflow as tf
from .f_approximator import FunctionApproximator

class LinearApproximator(FunctionApproximator):

    def __init__(self, input_tensor, linear, session, **kwargs):
        super(LinearApproximator, self).__init__(input_tensor, linear, session, **kwargs)


    def get_num_weights(self):
        return tf.size(self.W).eval()


    def predict(self, x):
        return self.sess.run(self.model, feed_dict={self.input_tensor: x.reshape(1, len(x))})
