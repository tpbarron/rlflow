
import tensorflow as tf
from .f_approximator import FunctionApproximator

class LinearApproximator(FunctionApproximator):

    def __init__(self, in_d, out_d, **kwargs):
        self.in_d = in_d # input dim
        self.out_d = out_d # output dim

        self.W = tf.Variable(tf.random_normal((out_d, in_d)), name="W")
        self.obs = tf.placeholder(tf.float32, shape=(in_d, 1), name="obs")
        self.f_predict = tf.squeeze(tf.matmul(self.W, self.obs))

        super(LinearApproximator, self).__init__(self.f_predict, **kwargs)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


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


    def predict(self, pattern):
        return self.sess.run(self.f_predict, feed_dict={self.obs: pattern.reshape(self.in_d, 1)})
