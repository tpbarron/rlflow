from __future__ import print_function

class FunctionApproximator(object):

    def __init__(self):
        pass


    def update(self, gradient):
        raise NotImplementedError


    def predict(self, input):
        raise NotImplementedError
