from __future__ import print_function
import numpy as np

from markov.algos.td.td_algo import RLTDAlgorithm
from markov.core import rl_utils

class QLearning(RLTDAlgorithm):

    def __init__(self, env, policy, step=0.1, discount=0.9, episode_len=np.inf):
        super(QLearning, self).__init__(env, policy)
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
        Run a single episode and perform QL updates along the way
        """
        ep_steps = 0
        done = False
        state = self.env.reset()

        while not done and ep_steps < self.episode_len:
            state = rl_utils.apply_prediction_preprocessors(self.policy, state)
            action = self.policy.predict(state)
            print ("Action: " + str(action), end='\n\n')
            action = rl_utils.apply_prediction_postprocessors(self.policy, action)
            print ("Action: " + str(action), end='\n\n')
            next_state, reward, done, _ = self.env.step(action)
            max_action = self.policy.get_max_action(next_state)

            self.ql_update(state, action, reward, next_state, max_action)
            state = next_state

            ep_steps += 1


    def step(self):
        pass


    def reset(self):
        pass
