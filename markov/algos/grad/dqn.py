from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn

from markov.core import rl_utils
from markov.algos.grad.grad_algo import RLGradientAlgorithm

class DQN(RLGradientAlgorithm):
    """
    Basic deep q network implementation based on tensorflow network
    """

    def __init__(self,
                 env,
                 policy,
                 session,
                 memory,
                 episode_len=np.inf,
                 discount=1.0,
                 standardize=True,
                 learning_rate=0.01,
                 optimizer='sgd',
                 clip_gradients=(None, None)):

        super(DQN, self).__init__(env,
                                 policy,
                                 session,
                                 episode_len,
                                 discount,
                                 standardize,
                                 optimizer,
                                 learning_rate,
                                 clip_gradients)

        self.memory = memory

        self.states = self.policy.input_tensor
        self.action_values = self.policy.model

        # vanilla gradient = mul(sum(logprobs * rewards))
        self.L = -tf.reduce_sum(tf.mul(self.logprobs, self.rewards))
        self.grads_and_vars = self.opt.compute_gradients(self.L)

        if None not in self.clip_gradients:
            self.clipped_grads_and_vars = [(tf.clip_by_value(gv[0], clip_gradients[0], clip_gradients[1]), gv[1])
                                            for gv in self.grads_and_vars]
            self.update = self.opt.apply_gradients(self.clipped_grads_and_vars)
        else:
            self.update = self.opt.apply_gradients(self.grads_and_vars)

        self.sess.run(tf.initialize_all_variables())
