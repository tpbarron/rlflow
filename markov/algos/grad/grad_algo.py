
from markov.algos.algo import RLAlgorithm
import tflearn

class RLGradientAlgorithm(RLAlgorithm):
    """
    Parent class for any algorithm that uses gradient based optimization. This
    includes methods like DQN since they are based on neural network approximators
    even though traditionally DQN would classify as a temporal difference algorithm.
    """

    def __init__(self,
                 env,
                 policy,
                 session,
                 episode_len,
                 discount,
                 standardize,
                 input_processor,
                 optimizer,
                 learning_rate,
                 clip_gradients):

        super(RLGradientAlgorithm, self).__init__(env,
                                                  policy,
                                                  session,
                                                  episode_len,
                                                  discount,
                                                  standardize,
                                                  input_processor)

        self.learning_rate = learning_rate
        self.tfl_opt = tflearn.optimizers.get(optimizer)(learning_rate)
        self.tfl_opt.build()
        self.opt = self.tfl_opt.get_tensor()

        self.clip_gradients = clip_gradients


    def optimize(self):
        """
        Optimize method that subclasses implement
        """
        return super(RLGradientAlgorithm, self).optimize()
