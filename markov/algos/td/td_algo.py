
import numpy as np
from markov.algos.algo import RLAlgorithm

class RLTDAlgorithm(RLAlgorithm):

    def __init__(self, env, policy, discount, episode_len):
        """
        TODO: either unify these or separate completely
        """
        super(RLTDAlgorithm, self).__init__(env, policy, None, episode_len, discount, False, None)
        # pass


    def optimize(self):
        """
        Optimize method that subclasses implement
        """
        raise NotImplementedError
