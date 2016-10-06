from __future__ import print_function
import numpy as np


class NaturalPG:

    def __init__(self, env):
        self.env = env


    def optimize(self, policy, episode_len=np.inf):
        # run episode with initial weights to get J_ref
        J_ref = self.env.rollout_with_policy(policy, episode_len)

        grad = 0
        return grad
