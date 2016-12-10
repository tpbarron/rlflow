
from .exploration import Exploration

class EpsilonGreedy(Exploration):

    def __init__(self,
                 epsilon,
                 decay_to_value,
                 decay_until_iteration,
                 decay_method='linear'):
        self.epsilon = epsilon


    def explore(self):
        pass
