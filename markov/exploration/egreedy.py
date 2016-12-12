import random
from .exploration import Exploration

class EpsilonGreedy(Exploration):

    def __init__(self,
                 initial_epsilon,
                 decay_to_value,
                 decay_to_iteration):
        """
        Epsilon between [0, 1]
        """
        self.initial_epsilon = initial_epsilon
        self.decay_to_value = decay_to_value
        self.decay_to_iteration = decay_to_iteration

        self.current_iteration = 0


    def increment_iteration(self):
        self.current_iteration += 1
        

    def explore(self):
        """
        Returns true if should explore,
        else False
        """
        if self.current_iteration < self.decay_to_iteration:
            epsilon = 1.0 - self.initial_epsilon * (float(self.current_iteration) / self.decay_to_iteration)
        else:
            epsilon = self.decay_to_value

        if random.random() < epsilon:
            return True

        return False
