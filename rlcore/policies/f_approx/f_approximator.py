from __future__ import print_function

from rlcore.policies.policy import Policy

class FunctionApproximator(Policy):

    def __init__(self, model, **kwargs):
        super(FunctionApproximator, self).__init__(**kwargs)
        self.model = model


    def get_num_weights(self):
        raise NotImplementedError


    def get_weight_variation(self):
        raise NotImplementedError


    def update(self, gradient):
        raise NotImplementedError


    def predict(self, obs):
        raise NotImplementedError
