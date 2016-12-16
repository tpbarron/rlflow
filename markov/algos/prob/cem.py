from __future__ import print_function
import numpy as np


class CrossEntropyMethod:

    def __init__(self, env):
        self.env = env


    def optimize(self):
        raise NotImplementedError
