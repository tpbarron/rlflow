from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import copy
from markov.policies.tab import TabularPolicy


class ValueIteration(object):


    def __init__(self, env, policy, discount=0.9, delta_thresh=0.1):
        self.env = env
        self.policy = policy
        self.discount = discount
        self.delta_thresh = delta_thresh


    def iterate(self):
        # (*) dictionary dict of dicts of lists, where
        #    P[s][a] == [(probability, nextstate, reward, done), ...]
        # for each state, s, in environment
        while True:
            delta = 0.0
            values_copy = copy.deepcopy(self.policy.values)
            for s in range(self.env.observation_space.n):
                max_a = 0.0
                # for each possible action a
                for a in range(self.env.action_space.n):
                    sum_a = 0.0
                    next_state_probs = self.env.P[s][a]
                    # print (next_state_probs)
                    # compute Sum(Prob(s, a, s') * (Reward(s, a, s') + discount*V(s')))
                    # for each possible next state s'
                    for next in next_state_probs:
                        p, sp, rew, done = next
                        sum_a += p * (rew + self.discount*values_copy[sp])

                    # print ("Sum a, max a: ", sum_a, max_a)
                    if (sum_a > max_a):
                        max_a = sum_a

                # update V(s) to be the value generated from the best action
                self.policy.set_value(s, max_a)
                delta = max(delta, abs(values_copy[s] - self.policy.get_value(s)))

            if (delta < self.delta_thresh):
                break

        # genrate deterministic policy
        det_pol = TabularPolicy(self.env.observation_space, self.env.action_space)
        for s in range(self.env.observation_space.n):
            action = self.policy.get_action(s) #self.env.P[s])
            det_pol.set_action(s, action)
        return det_pol
