from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from rlcore.policies.tab import TabularValue


class PolicyIteration(object):


    def __init__(self, env, policy, discount=0.9, delta_thresh=0.1):
        self.env = env
        self.policy = policy
        self.values = TabularValue(env.observation_space.n)
        self.discount = discount
        self.delta_thresh = delta_thresh


    def evaluate(self):
        # (*) dictionary dict of dicts of lists, where
        #    P[s][a] == [(probability, nextstate, reward, done), ...]
        # for each state, s, in environment

        while True:
            delta = 0.0
            for s in range(self.env.observation_space.n):
                vtmp = self.values.get_value(s)

                a = self.policy.get_action(s)
                next_state_probs = self.env.P[s][a]
                vsum = 0.0
                for next in next_state_probs:
                    p, sp, rew, done = next
                    vsum += p * (rew + self.discount*self.values.get_value(sp))
                self.values.set_value(s, vsum)
                delta = max(delta, abs(vtmp - self.values.get_value(s)))

            if (delta < self.delta_thresh):
                break


    def improve(self):
        stable = True
        for s in range(self.env.observation_space.n):
            ptmp = self.policy.get_action(s)

            # find best action
            max_aval = 0.0
            max_a = 0
            # for each possible action a
            for a in range(self.env.action_space.n):
                sum_a = 0.0
                next_state_probs = self.env.P[s][a]
                # print (next_state_probs)
                # compute Sum(Prob(s, a, s') * (Reward(s, a, s') + discount*V(s')))
                # for each possible next state s'
                for next in next_state_probs:
                    p, sp, rew, done = next
                    sum_a += p * (rew + self.discount*self.values.get_value(sp))

                if (sum_a > max_aval):
                    max_aval = sum_a
                    max_a = a

            self.policy.set_action(s, max_a)
            if (ptmp != max_a):
                stable = False

        return stable



    def iterate(self, to_convergence=True):
        pstable = False
        while not pstable:
            self.evaluate()
            pstable = self.improve()

            if not to_convergence:
                return pstable
        return pstable
