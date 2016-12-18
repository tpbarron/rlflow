
import tflearn
from ..algo import RLAlgorithm

class RLTDAlgorithm(RLAlgorithm):


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

        super(RLTDAlgorithm, self).__init__(env,
                                            policy,
                                            session,
                                            episode_len,
                                            discount,
                                            standardize,
                                            input_processor)

        self.learning_rate = learning_rate

        if isinstance(optimizer, tflearn.optimizers.Optimizer):
            self.tfl_opt = optimizer
        else:
            self.tfl_opt = tflearn.optimizers.get(optimizer)(learning_rate)

        self.tfl_opt.build()
        self.opt = self.tfl_opt.get_tensor()

        self.clip_gradients = clip_gradients


    def optimize(self):
        """
        Optimize method that subclasses implement
        """
        return super(RLTDAlgorithm, self).optimize()
