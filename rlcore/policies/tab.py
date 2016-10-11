from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from .policy import Policy


class Tabular(Policy):

    def __init__(self, num_states):
        self.num_states = num_states
        self.values = np.zeros((num_states,))


    @property
    def is_deterministic(self):
        return True


    def prettyprint(self):
        n = int(np.sqrt(self.num_states)) # assume square
        for r in range(n):
            print (self.values[r*n:(r+1)*n])
        print ('\n')
        # print (self.values[0:4])
        # print (self.values[4:8])
        # print (self.values[8:12])
        # print (self.values[12:])
        # print ('\n')



class TabularValue(Tabular):

    def __init__(self, num_states):
        super(TabularValue, self).__init__(num_states)


    def get_value(self, state):
        return self.values[state]


    def set_value(self, state, value):
        self.values[state] = value


    def get_action(self, probs):
        """
        This method takes in the probability of successive states: env.P[state]
        """
        best_action = 0
        best_value = -np.inf
        for k, v in probs.items():
            for vals in v:
                # print (vals)
                if (vals[2] == 1.0): # if goal
                    return k
                if (self.values[vals[1]] > best_value):
                    best_action = k
                    best_value = self.values[vals[1]]
        return best_action



class TabularPolicy(Tabular):

    def __init__(self, num_states):
        super(TabularPolicy, self).__init__(num_states)


    def get_action(self, state):
        return self.values[state]


    def set_action(self, state, action):
        self.values[state] = action
