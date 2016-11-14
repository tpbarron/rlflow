from __future__ import print_function
import numpy as np

from rlcore.algos.td.td_algo import RLTDAlgorithm

class QLearning(RLTDAlgorithm):

    def __init__(self, env, policy, step=0.1, discount=0.9, episode_len=np.inf):
        self.env = env
        self.policy = policy
        self.step = step
        self.discount = discount
        self.episode_len = episode_len


    def ql_update(self, s, a, r, sp, max_a):
        """
        Q(s, a) = Q(s, a) + alpha * (r + discount * max_a' Q(s', a') - Q(s, a))
        """
        qsa = self.policy.get_value(s, a)
        qspap_max = self.policy.get_value(sp, max_a)
        val = qsa + self.step * (r + self.discount * qspap_max - qsa)
        self.policy.set_value(s, a, val)


    def optimize(self):
        """
        Run a single episode and perform SARSA updates along the way
        """
        ep_steps = 0
        done = False
        state = self.env.reset()

        while not done and ep_steps < self.episode_len:
            action = self.policy.predict(state)
            next_state, reward, done, info = self.env.step(action)
            max_action = self.policy.get_max_action(next_state)

            self.ql_update(state, action, reward, next_state, max_action)
            state = next_state

            ep_steps += 1
