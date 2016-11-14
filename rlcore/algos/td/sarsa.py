from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from rlcore.algos.td.td_algo import RLTDAlgorithm

class SARSA(RLTDAlgorithm):

    def __init__(self, env, policy, step=0.1, discount=0.9, episode_len=np.inf):
        self.env = env
        self.policy = policy
        self.step = step
        self.discount = discount
        self.episode_len = episode_len


    def sarsa_update(self, s, a, r, sp, ap):
        """
        Q(s, a) = Q(s, a) + alpha * (r + discount * Q(s', a') - Q(s, a))
        """
        qsa = self.policy.get_value(s, a)
        qspap = self.policy.get_value(sp, ap)
        val = qsa + self.step * (r + self.discount * qspap - qsa)
        self.policy.set_value(s, a, val)


    def optimize(self):
        """
        Run a single episode and perform SARSA updates along the way
        """
        ep_steps = 0
        done = False
        state = self.env.reset()
        action = self.policy.predict(state)

        while not done and ep_steps < self.episode_len:
            next_state, reward, done, info = self.env.step(action)
            next_action = self.policy.predict(next_state)

            self.sarsa_update(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

            ep_steps += 1
