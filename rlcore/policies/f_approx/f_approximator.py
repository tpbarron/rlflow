from __future__ import print_function

from rlcore.policies.policy import Policy

class FunctionApproximator(Policy):

    def __init__(self, input_tensor, model, session, **kwargs):
        super(FunctionApproximator, self).__init__(**kwargs)
        self.input_tensor = input_tensor
        self.model = model
        self.sess = session


    def get_num_weights(self):
        raise NotImplementedError


    def get_weight_variation(self):
        raise NotImplementedError


    def update(self, gradient):
        raise NotImplementedError


    def predict(self, obs):
        raise NotImplementedError
