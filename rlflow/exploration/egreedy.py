import random
from .exploration import Exploration

class EpsilonGreedy(Exploration):

    def __init__(self,
                 initial_epsilon,
                 decay_to_value=None,
                 decay_to_iteration=None):
        """
        Epsilon between [0, 1]
        """
        super(EpsilonGreedy, self).__init__(Exploration.CONDITIONAL_EXPLORATION)
        self.initial_epsilon = initial_epsilon
        self.decay_to_value = decay_to_value
        self.decay_to_iteration = decay_to_iteration
        # self.current_iteration = 0


    # def increment_iteration(self):
    #     self.current_iteration += 1


    def explore(self, current_iteration):
        """
        Returns true if should explore, else False
        """
        if self.decay_to_value is None or self.decay_to_iteration is None:
            epsilon = self.initial_epsilon
        else:
            if current_iteration < self.decay_to_iteration:
                epsilon = 1.0 - self.initial_epsilon * (float(current_iteration) / self.decay_to_iteration)
            else:
                epsilon = self.decay_to_value

        # print ("epsilon: ", epsilon)

        return random.random() < epsilon
