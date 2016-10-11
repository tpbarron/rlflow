from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from rlcore.policies.tab import TabularValue
from rlcore.policies.tab import TabularPolicy

class TDLambda(object):

    def __init__(self, env, lmda=0.0, discount=0.9, step=0.1):
        self.env = env
        self.lmda = lmda
        self.discount = discount
        self.step = step
        self.values = TabularValue(self.env.observation_space.n)
        self.traces = np.zeros((self.env.observation_space.n))


    def td_update(self, state_t, state_tp, r):
        """
        Update the value of state_t given the reward and next state value
        """
        delta = r + self.discount * self.values.get_value(state_tp) - \
            self.values.get_value(state_t)
        self.values.set_value(state_t, self.values.get_value(state_t) + \
            self.step * delta)


    def iterate(self, eps, episode_len=np.inf):
        """
        Run eps episodes of the environment, updating the value function in the process
        """
        for i in range(eps):
            print ("Iteration:", i)
            done = False
            prev_state = None
            state = self.env.reset()
            steps = 0

            # reset traces
            self.traces = {}

            while not done and steps < episode_len:
                last_state = state
                action = self.env.action_space.sample() #self.values.get_action(self.env.P[state])
                #env.action_space.sample() # policy.get_action(state)
                state, reward, done, info = self.env.step(action)

                # update trace
                self.traces[last_state]
                # do td update with last two states
                self.td_update(prev_state, state, reward)
                total_reward += reward
                steps += 1

        # generate deterministic policy
        det_pol = TabularPolicy(self.env.observation_space.n)
        for s in range(self.env.observation_space.n):
            action = self.values.get_action(self.env.P[s])
            det_pol.set_action(s, action)
        return det_pol
