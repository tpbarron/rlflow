from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from rlcore.policies.tab import TabularValue

class TDLambda(object):

    def __init__(self, env, lmda, discount=0.9, step=0.1):
        self.env = env
        self.obs_space = env.observation_space.n
        self.lmda = lmda
        self.discount = discount
        self.step = step
        self.values = TabularValue(self.obs_space)


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
        Run eps episodes of the environment, updating the value in the process
        """
        total_reward = 0

        for i in range(eps):
            print ("Iteration:", i)
            done = False
            prev_state = None
            state = self.env.reset()
            steps = 0
            while not done and steps < episode_len:
                self.env.render()
                last_state = state
                # TODO: should the policy be updated after each iteration?
                action = self.env.action_space.sample() #self.values.get_action(self.env.P[state])
                #env.action_space.sample() # policy.get_action(state)
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                # do td update with last two states
                self.td_update(prev_state, state, reward)
                steps += 1

            last_state = None
        print ("Total reward: ", total_reward)
