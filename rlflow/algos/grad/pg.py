from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn

from rlflow.core import rl_utils
from rlflow.algos.algo import RLAlgorithm

class PolicyGradient(RLAlgorithm):
    """
    Basic stochastic policy gradient implementation based on Keras network
    """

    def __init__(self,
                 env,
                 policy,
                 episode_len=np.inf,
                 discount=1.0,
                 standardize=True,
                 input_processor=None,
                 optimizer='sgd',
                 clip_gradients=(None, None),
                 baseline=None):

        super(PolicyGradient, self).__init__(env,
                                             policy,
                                             episode_len,
                                             discount,
                                             standardize,
                                             input_processor,
                                             optimizer,
                                             clip_gradients)

        self.states = self.policy.inputs[0]
        self.actions = tf.placeholder(tf.float32, shape=(None, env.action_space.n))
        self.rewards = tf.placeholder(tf.float32, shape=(None))

        self.probs = self.policy.output
        self.action_probs = tf.mul(self.probs, self.actions)
        self.reduced_action_probs = tf.reduce_sum(self.action_probs, reduction_indices=[1])
        self.logprobs = tf.log(self.reduced_action_probs)

        # vanilla gradient = mul(sum(logprobs * rewards))
        self.L = -tf.reduce_sum(tf.mul(self.logprobs, self.rewards))

        self.grads_and_vars = self.opt.compute_gradients(self.L)

        if None not in self.clip_gradients:
            self.clipped_grads_and_vars = [(tf.clip_by_value(gv[0], clip_gradients[0], clip_gradients[1]), gv[1])
                                            for gv in self.grads_and_vars]
            self.update = self.opt.apply_gradients(self.clipped_grads_and_vars)
        else:
            self.update = self.opt.apply_gradients(self.grads_and_vars)

        # TODO: if baseline is set, learn a critic
        self.baseline = baseline


    def optimize(self):
        ep_states, ep_actions, ep_rewards, ep_infos = super(PolicyGradient, self).optimize()

        formatted_rewards = np.array(ep_rewards)

        if self.discount != 1.0:
            formatted_rewards = rl_utils.discount_rewards(formatted_rewards, gamma=self.discount)

        formatted_actions = np.zeros((len(ep_actions), self.env.action_space.n))
        for i in range(len(ep_actions)):
            formatted_actions[i][ep_actions[i]] = 1.0

        if self.standardize:
            formatted_rewards = rl_utils.standardize_rewards(formatted_rewards)

        self.sess.run(self.update, feed_dict={self.actions: formatted_actions,
                                              self.states: ep_states,
                                              self.rewards: formatted_rewards})
        return ep_states, ep_actions, ep_rewards, ep_infos
