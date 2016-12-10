from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import gym
from .policy import Policy


"""
Defines tabular policies, all epsilon greedy
"""

class Tabular(Policy):

    def __init__(self, observation_space, action_space, **kwargs):
        super(Tabular, self).__init__(**kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.epsilon = 0.1
        if ('epsilon' in kwargs):
            self.epsilon = kwargs['epsilon']

        self.num_states = np.inf if isinstance(self.observation_space, gym.spaces.box.Box) else self.observation_space.n
        self.num_actions = np.inf if isinstance(self.action_space, gym.spaces.box.Box) else self.action_space.n

        self.edge = np.inf if self.num_states == np.inf else int(np.sqrt(self.num_states))
        self.values = {}


    @property
    def is_deterministic(self):
        return True


    def init(self, mode='random'):
        assert self.num_states is not np.inf
        if (mode == 'random'):
            # randomize initial values
            for s in range(self.num_states):
                self.values[s] = np.random.random()
        elif (mode == 'zeros'):
            for s in range(self.num_states):
                self.values[s] = 0


    def prettyprint(self):
        for r in range(self.edge):
            for c in range(self.edge):
                s = r*self.edge+c
                if not s in self.values:
                    print ("-1", end=' ')
                else:
                    print (self.values[r*self.edge+c], end=' ')
            print ()



class TabularValue(Tabular):
    """
    Tabular value function representation as a python dictionary

    Both the observation and action spaces must be finite and discrete
    """

    def __init__(self, observation_space, action_space, **kwargs):
        assert isinstance(observation_space, gym.spaces.discrete.Discrete)
        assert isinstance(action_space, gym.spaces.discrete.Discrete)
        assert action_space.n == 4 # grid world restriction
        super(TabularValue, self).__init__(observation_space, action_space, **kwargs)
        self.init(mode='random')


    def get_value(self, state):
        return self.values[state]


    def set_value(self, state, value):
        self.values[state] = value


    def get_action(self, state):
        """
        The method takes in the current state and returns the best action to take
        given the current values. This assumes a square grid world.
        """

        best_action = None
        best_value = -np.inf
        actions = [0, 1, 2, 3] # left, down, right, up
        for a in actions:
            row = state // self.edge
            col = state % self.edge
            # print (row, col)
            if a == 0:
                col = max(col-1, 0)
            elif a == 1:
                row = min(row+1, self.edge-1)
            elif a == 2:
                col = min(col+1, self.edge-1)
            elif a == 3:
                row = max(row-1, 0)
            # print (row, col)

            new_state = row * self.edge + col
            # print (new_state)
            if (self.values[new_state] > best_value or new_state == self.num_states-1): #goal
                best_value = 1.0 if new_state == self.num_states-1 else self.values[new_state]
                best_action = a
        return best_action


    def predict(self, state):
        if self.mode == Policy.TEST:
            return self.get_action(state)

        if np.random.random() < self.epsilon():
            return self.action_space.sample()
        return self.get_action(state)



class TabularActionValue(Tabular):
    """
    Tabular Q value representation as a python dictionary:

    action_values[state] -> { action1: value1, action2: value2, ... , action_n: value_n }

    The action space must be discrete and finite but the observation space may
    be infinite and/or continous.
    """

    def __init__(self, observation_space, action_space, **kwargs):
        assert isinstance(action_space, gym.spaces.discrete.Discrete)
        super(TabularActionValue, self).__init__(observation_space,  action_space, init='random', **kwargs)

        # can index action values by [state, action] -> value
        self.action_values = {}


    def get_value(self, state, action):
        if not state in self.action_values:
            self.action_values[state] = {}
        if not action in self.action_values[state]:
            # if state action pair not yet seen, declare value to be 0
            self.action_values[state][action] = 0.0
        return self.action_values[state][action]


    def set_value(self, state, action, value):
        if not state in self.action_values:
            self.action_values[state] = {}
        self.action_values[state][action] = value


    def get_max_action(self, state):
        if not state in self.action_values:
            # if haven't seen this state, just sample a random action
            return self.action_space.sample()

        action_dict = self.action_values[state]
        a = max(action_dict, key=lambda key: action_dict[key])
        return a


    def predict(self, state):
        # return highest action value with ep greedy
        if self.mode == Policy.TEST:
            return self.get_max_action(state)

        if (np.random.random() < self.epsilon or (not state in self.action_values)):
            return self.action_space.sample()
        return self.get_max_action(state)



    def prettyprint(self):
        for i in range(self.edge):
            for j in range(self.edge):
                print (self.get_max_action(i*self.edge+j), end=' ')
            print ()
        # for s in self.action_values:
        #     print ("s:", self.get_max_action(s))
        # for s in self.action_values:
        #     avs = self.action_values[s]
        #     for a in avs:
        #         print ("av["+str(s)+"]["+str(a)+"] = " + str(avs[a]), end=' ')
        #     print ()




class TabularPolicy(Tabular):

    def __init__(self, observation_space, action_space, **kwargs):
        assert isinstance(observation_space, gym.spaces.discrete.Discrete)
        assert isinstance(action_space, gym.spaces.discrete.Discrete)
        assert action_space.n == 4 # grid world restriction
        super(TabularPolicy, self).__init__(observation_space, action_space, **kwargs)
        self.init(mode='zeros')


    def get_action(self, state):
        return self.values[state]


    def set_action(self, state, action):
        self.values[state] = action


    def predict(self, state):
        if self.mode == Policy.TEST:
            return self.get_action(state)

        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        return self.get_action(state)
