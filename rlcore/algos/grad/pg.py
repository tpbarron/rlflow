from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn

from rlcore.core import rl_utils
from rlcore.algos.grad.grad_algo import RLGradientAlgorithm

class PolicyGradient(RLGradientAlgorithm):
    """
    Basic stochastic policy gradient implementation based on Keras network
    """

    def __init__(self,
                 env,
                 policy,
                 session,
                 episode_len=100,
                 discount=False,
                 standardize=True,
                 learning_rate=0.01,
                 optimizer='sgd',
                 clip_gradients=(None, None)):

        super(PolicyGradient, self).__init__(env,
                                             policy,
                                             session,
                                             episode_len,
                                             discount,
                                             standardize,
                                             optimizer,
                                             learning_rate)

        self.states = self.policy.input_tensor
        self.actions = tf.placeholder(tf.float32, shape=(None, env.action_space.n))
        self.rewards = tf.placeholder(tf.float32, shape=(None))

        self.probs = self.policy.model
        self.action_probs = tf.mul(self.probs, self.actions)
        self.reduced_action_probs = tf.reduce_sum(self.action_probs, reduction_indices=[1])
        self.logprobs = tf.log(self.reduced_action_probs)

        # vanilla gradient = mul(sum(logprobs * rewards))
        self.L = -tf.reduce_sum(tf.mul(self.logprobs, self.rewards))

        self.grads_and_vars = self.opt.compute_gradients(self.L)

        if None not in clip_gradients:
            self.clipped_grads_and_vars = [(tf.clip_by_value(gv[0], clip_gradients[0], clip_gradients[1]), gv[1])
                                            for gv in self.grads_and_vars]
            self.update = self.opt.apply_gradients(self.clipped_grads_and_vars)
        else:
            self.update = self.opt.apply_gradients(self.grads_and_vars)

        self.sess.run(tf.initialize_all_variables())


    def optimize(self):
        ep_states, ep_raw_actions, ep_processed_actions, ep_rewards = rl_utils.rollout_env_with_policy(self.env,
                                                                                                       self.policy,
                                                                                                       episode_len=self.episode_len)

        if self.discount:
            ep_rewards = rl_utils.discount_rewards(np.array(ep_rewards))

        formatted_actions = np.zeros((len(ep_raw_actions), self.env.action_space.n))
        for i in range(len(ep_processed_actions)):
            formatted_actions[i][ep_processed_actions[i]] = 1.0

        formatted_rewards = ep_rewards
        if self.standardize:
            formatted_rewards = rl_utils.standardize_rewards(formatted_rewards)

        self.sess.run(self.update, feed_dict={self.actions: formatted_actions,
                                              self.states: ep_states,
                                              self.rewards: formatted_rewards})
