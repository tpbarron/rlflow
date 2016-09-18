import numpy as np
from .f_approximator import FunctionApproximator

class LinearApproximator(FunctionApproximator):

    def __init__(self, n, lr=0.1):
        super(LinearApproximator, self).__init__()
        self.n = n
        self.lr = lr
        self.w = np.random.normal(0, 0.01, n)


    def get_num_weights(self):
        return self.w.size


    def get_weight_variation(self, dist='gaussian'):
        # define distribution based on given weights
        mu, sigma = 0.0, np.std(self.w)
        deltas = np.random.normal(mu, sigma, self.w.shape)
        varied_weights = np.add(np.copy(self.w), deltas)
        policy_variation = LinearApproximator(self.n, lr=self.lr)
        policy_variation.w = varied_weights
        return policy_variation, deltas


    def update(self, gradient):
        self.w += self.lr * gradient


    def predict(self, input):
        return np.dot(self.w, input)
