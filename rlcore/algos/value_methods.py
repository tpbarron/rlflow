from __future__ import print_function
import numpy as np


class ValueMethod(object):

    def __init__(self,
                nactions,
                init=None,
                init_mu=0.0,
                init_std=0.0001,
                optimistic_init=False,
                step_update=False,
                alpha=.5):
        self.nactions = nactions
        self.starting_action_values = init
        self.init_mu = init_mu
        self.init_std = init_std
        self.optimistic_init = optimistic_init
        if (self.optimistic_init):
            self.init_mu = 5.0
        self.step_update = step_update
        self.alpha = alpha
        self.reset()


    def reset(self):
        self.action_values = self.starting_action_values
        if (self.starting_action_values == None):
            self.action_values = np.random.normal(self.init_mu, self.init_std, self.nactions)
        self.action_counts = np.ones((self.nactions,))


    def get_action(self):
        raise NotImplementedError


    def update_values(self, a, r):
        """
        a - action taken
        r - reward for action a
        """
        # Maintain average of rewards for each state over time where each
        # state is weighted equally. Distributions are stationary.
        # Incremental update: Q_k+1 = Q_k + 1/k * (R_k - Q_k)
        # If using step update: Q_k+1 = Q_k + alpha * (R_k - Q_k)
        if (self.step_update):
            self.action_values[a] = self.action_values[a] + self.alpha * (r - self.action_values[a])
        else:
            self.action_values[a] = self.action_values[a] + 1.0/self.action_counts[a] * (r - self.action_values[a])
        self.action_counts[a] += 1



class EpsilonGreedy(ValueMethod):

    def __init__(self,
                epsilon,
                nactions,
                **kwargs):
        super(EpsilonGreedy, self).__init__(nactions, **kwargs)
        self.epsilon = epsilon


    def get_action(self):
        v = np.random.random()
        if (v < self.epsilon):
            action = np.random.randint(0, self.nactions)
        else:
            action = np.argmax(self.action_values)
        return action



class SoftmaxActionSelection(ValueMethod):

    def __init__(self,
                temp,
                nactions,
                **kwargs):
        super(SoftmaxActionSelection, self).__init__(nactions, **kwargs)
        self.temp = temp


    def get_action(self):
        action_probs = np.exp(self.action_values / self.temp)
        action_probs /= np.sum(action_probs)
        action = np.random.choice(len(self.action_values), 1, p=action_probs)
        return action
