import sys
import numpy as np
import tensorflow as tf
import rlcore.core.tensorflow_utils as tf_utils
from .f_approximator import FunctionApproximator

class LinearApproximator(FunctionApproximator):

    def __init__(self, in_d, out_d, lr=0.1, weight_variance=1.0):
        super(LinearApproximator, self).__init__()
        self.in_d = in_d # input dim
        self.out_d = out_d # output dim

        self.lr = tf.constant(lr, shape=[1], name="learning_rates")
        self.W = tf.Variable(tf.random_normal([out_d, in_d], 1.0, weight_variance), name="W")
        self.Wbackup = tf.Variable(tf.zeros([out_d, in_d]), name="Wbackup")

        self.deltas = tf.placeholder(tf.float32, [out_d, in_d], name="deltas")
        self.grad = tf.placeholder(tf.float32, [out_d, in_d], name="grad")
        self.obs = tf.placeholder(tf.float32, [in_d, 1], name="obs")

        self.f_weight_backup = self.Wbackup.assign(self.W)
        self.f_weight_restore = self.W.assign(self.Wbackup)
        self.f_weight_update = self.W.assign_add(self.W + self.lr * self.grad)
        self.f_gen_deltas = tf.random_normal(self.W.get_shape(), 0.0, 0.1*tf_utils.stddev(self.W))
        self.f_weight_variation = self.W.assign(tf.add(self.Wbackup, self.deltas))
        self.f_predict = tf.matmul(self.W, self.obs)

        self.init = tf.initialize_all_variables()
        self.init.run()



    def backup_weights(self):
        self.f_weight_backup.eval()


    def restore_weights(self):
        self.f_weight_restore.eval()


    def get_num_weights(self):
        return tf.size(self.W).eval()


    def get_weight_variation(self, stdcoeff=0.1):
        # define distribution based on given weights
        # TODO use variable for stdcoeff
        deltas = self.f_gen_deltas.eval()
        self.f_weight_variation.eval(feed_dict={self.deltas: deltas})
        return deltas


    def update(self, gradient):
        self.f_weight_update.eval(feed_dict={self.grad: gradient.reshape(self.grad.get_shape())})


    def predict(self, input):
        return self.f_predict.eval(feed_dict={self.obs: input.reshape(4,1)})
