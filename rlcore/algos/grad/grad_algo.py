from rlcore.algos.algo import RLAlgorithm

class RLGradientAlgorithm(RLAlgorithm):

    def __init__(self, env, policy):
        super(RLGradientAlgorithm, self).__init__(env, policy)


    def optimize(self):
        """
        Optimize method that subclasses implement
        """
        raise NotImplementedError
