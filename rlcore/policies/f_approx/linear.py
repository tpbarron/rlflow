
import tensorflow as tf
import tflearn
from .f_approximator import FunctionApproximator

class LinearApproximator(FunctionApproximator):

    def __init__(self, input_tensor, linear, session, **kwargs):
        super(LinearApproximator, self).__init__(input_tensor, linear, session, **kwargs)


    def get_num_weights(self):
        return tf.size(self.W).eval()


    def get_max_action(self, next_state):
        # now return the max action given the state
        # the representation could either be
        # (state + action) -> value
        # or
        # (state) -> [set of action values]
        return 0


    def get_value(self, state):
        return 0


    def predict(self, x):
        return self.sess.run(self.model, feed_dict={self.input_tensor: x.reshape(1, len(x))})
