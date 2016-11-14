from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from rlcore.algos.td.td_algo import RLTDAlgorithm
from rlcore.policies.tab import TabularValue
from rlcore.policies.tab import TabularPolicy
from rlcore.core import rl_utils

class TDLambda(RLTDAlgorithm):
    """
    TD Lambda implementation from Sutton's Intro to RL
    """

    def __init__(self,
                 env,
                 lmda=0.0,
                 discount=0.9,
                 step=0.1,
                 episode_len=np.inf,
                 evaluation_episodes=10):
        self.env = env
        self.lmda = lmda
        self.discount = discount
        self.step = step
        self.episode_len = episode_len
        self.evaluation_episodes = evaluation_episodes

        self.values_tab = TabularValue(self.env.observation_space, self.env.action_space)
        self.policy = self.get_policy()
        self.traces = np.zeros((self.env.observation_space.n))

        self.epsilon = 0.1


    def td_update(self, state_t, state_tp, r):
        """
        Update the value of state_t given the reward and next state value
        """
        delta = r + self.discount * self.values_tab.get_value(state_tp) - \
            self.values_tab.get_value(state_t)
        self.values_tab.set_value(state_t, self.values_tab.get_value(state_t) + \
            self.step * delta)


    def optimize(self):
        """
        Run a single iteration of policy evaluation and then a policy update.
        """
        for j in range(self.evaluation_episodes):
            ep_steps = 0
            done = False
            state = self.env.reset()
            while not done and ep_steps < self.episode_len:
                prev_state = state
                action = int(self.policy.predict(state))
                state, reward, done, info = self.env.step(action)

                # TODO: traces

                # update trace
                # self.traces[last_state]
                # do td update with last two states
                self.td_update(prev_state, state, reward)
                ep_steps += 1

        # update policy to be greedy
        self.policy = self.get_policy()


    def get_policy(self):
        # generate deterministic policy
        det_pol = TabularPolicy(self.env.observation_space, self.env.action_space, prediction_postprocessors=[rl_utils.cast_int])
        for s in range(self.env.observation_space.n):
            action = self.values_tab.get_action(s)
            det_pol.set_action(s, action)
        return det_pol
