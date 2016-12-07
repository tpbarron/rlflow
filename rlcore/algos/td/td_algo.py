
from rlcore.algos.algo import RLAlgorithm

class RLTDAlgorithm(RLAlgorithm):

    def __init__(self, env, policy):
        super(RLTDAlgorithm, self).__init__(env, policy)


    def optimize(self):
        """
        Optimize method that subclasses implement
        """
        raise NotImplementedError
