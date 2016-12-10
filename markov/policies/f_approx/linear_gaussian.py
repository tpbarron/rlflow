from __future__ import print_function
import sys
import numpy.matlib
import numpy as np
from .f_approximator import FunctionApproximator

class LinearGaussianApproximator(FunctionApproximator):

    def __init__(self, n, m, num_f, lr=0.1):
        super(LinearGaussianApproximator, self).__init__()
        self.n = n #num inputs, ie num functions
        self.m = m # num points to approximate
        self.num_f = num_f # number of bases
        self.lr = lr

        self.c = np.random.uniform(-1, 1, (self.n, self.num_f))
        self.w = np.random.normal(0, 0.01, self.num_f)


    def gaussian(self, c, x):
        assert len(x) == self.n
        return np.exp(-np.linalg.norm(c-x)**2.0)


    def compute_activations(self, input):
        acts = np.zeros((self.num_f,))
        for i in range(self.num_f):
            c = self.c[:,i]
            acts[i] = self.gaussian(c, input)
        return acts


    def get_num_weights(self):
        return self.w.size


    def get_weight_variation(self, dist='gaussian'):
        # define distribution based on given weights
        mu, sigma = 0.0, np.std(self.w)
        deltas = np.random.normal(mu, sigma, self.w.shape)
        varied_weights = np.add(np.copy(self.w), deltas)
        policy_variation = LinearGaussianApproximator(self.n,
                                                      self.m,
                                                      self.num_f,
                                                      lr=self.lr)
        policy_variation.w = varied_weights
        return policy_variation, deltas


    def update(self, gradient):
        self.w += self.lr * gradient


    def predict(self, input):
        activations = self.compute_activations(input)
        action = np.dot(activations, self.w)
        return action
