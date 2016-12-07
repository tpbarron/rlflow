
from rlcore.algos.algo import RLAlgorithm
import tflearn

class RLGradientAlgorithm(RLAlgorithm):

    def __init__(self, env, policy, session, episode_len, discount, standardize, optimizer, learning_rate):
        super(RLGradientAlgorithm, self).__init__(env, policy, session, episode_len, discount, standardize)

        self.learning_rate = learning_rate
        self.tfl_opt = tflearn.optimizers.get(optimizer)(learning_rate)
        self.tfl_opt.build()
        self.opt = self.tfl_opt.get_tensor()


    def optimize(self):
        """
        Optimize method that subclasses implement
        """
        raise NotImplementedError
