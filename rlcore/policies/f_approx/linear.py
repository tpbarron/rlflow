from __future__ import print_function

import numpy as np
from .f_approximator import FunctionApproximator

class LinearFunctionApproximator(FunctionApproximator):

    def __init__(self, n, lr=0.1):
        super(LinearFunctionApproximator, self).__init__()
        self.n = n
        self.lr = lr
        self.w = np.random.normal(0, 0.01, n)


    def update(self, gradient):
        self.w += self.lr * gradient


    def predict(self, input):
        return np.dot(self.w, input)
