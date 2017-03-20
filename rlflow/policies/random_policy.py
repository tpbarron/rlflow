from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from rlflow.policies.policy import Policy

class RandomPolicy(Policy):

    def __init__(self, action_space, mode=Policy.TRAIN):
        super(RandomPolicy, self).__init__(mode)
        self.action_space = action_space


    def predict(self, x):
        return self.action_space.sample()
