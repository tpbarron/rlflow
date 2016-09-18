from __future__ import print_function
import sys
import numpy.matlib
import numpy as np
from .f_approximator import FunctionApproximator

class LinearGaussianApproximator(FunctionApproximator):

    def __init__(self, n, gaussians=None, num_gaussians=20, discretization=100, lr=0.1):
        super(LinearGaussianApproximator, self).__init__()
        self.n = n
        self.num_gaussians = num_gaussians
        self.discretization = discretization
        self.lr = lr
        self.w = np.random.normal(0, 0.01, self.num_gaussians)

        # if (gaussians != None):
        #     self.gaussians = gaussians
        # else:
        self.gen_gaussians()


    def gen_gaussians(self):
        """
        This function requires:
            num_gaussians - the number of basis functions
            discretization - the number of timesteps
        """
        c, h = self.get_basis_func_params()
        z = np.linspace(0, 1, self.discretization)

        # compute the basis function activations at each timestep
        b = np.exp(-1 * np.square(
                    (np.matlib.repmat(z, self.num_gaussians, 1) - np.matlib.repmat(c[np.newaxis].T, 1, self.discretization))
                    ) / (2.0 * h))
        sum_b = np.matlib.repmat(np.sum(b, axis=0), self.num_gaussians, 1)
        b_norm = np.divide(b, sum_b)
        self.gaussians = b_norm


    def get_basis_func_params(self):
        c, h = None, None
        if (self.num_gaussians > 5):
            p = 1.0 / (self.num_gaussians - 3)
            a1 = np.array([0.0 - p])
            a2 = np.array([1.0 + p])
            mid = np.linspace(0.0, 1.0, self.num_gaussians - 2)
            c = np.concatenate((a1, mid, a2))
            h = (2 * 0.5 * (c[1] - c[0])) ** 2.0
        else:
            c = np.linspace(0, 1, self.num_gaussians)
            h = (2 * 0.5 * (c[1] - c[0])) ** 2.0 #1.0/self.num_gaussian
        return c, h


    def get_num_weights(self):
        return self.w.size


    def get_weight_variation(self, dist='gaussian'):
        # define distribution based on given weights
        mu, sigma = 0.0, 2*np.std(self.w)
        deltas = np.random.normal(mu, sigma, self.w.shape)
        varied_weights = np.add(np.copy(self.w), deltas)
        policy_variation = LinearGaussianApproximator(self.n,
                                                      gaussians=self.gaussians,
                                                      num_gaussians=self.num_gaussians,
                                                      discretization=self.discretization,
                                                      lr=self.lr)
        policy_variation.w = varied_weights
        return policy_variation, deltas


    def update(self, gradient):
        self.w += self.lr * gradient


    def predict(self, input, t=0):
        action = np.dot(self.gaussians[:,t], self.w)
        # print ("Action = ", action)
        # print ("w = ", self.w)
        # if (np.isnan(action)):
        #     print (t)
        #     print (self.gaussians[:,t])
        #     print (self.w)
        #     sys.exit(1)
        return np.dot(self.gaussians[:,t], self.w) #, input)
