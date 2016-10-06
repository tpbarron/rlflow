from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


class Policy(object):


    def __init__(self):
        pass


    @property
    def is_deterministic(self):
        raise NotImplementedError
        

    def predict(self, input):
        raise NotImplementedError
