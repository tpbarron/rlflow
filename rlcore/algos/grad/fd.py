from __future__ import print_function
import numpy as np
import tensorflow as tf

from rlcore.algos.grad.grad_algo import RLGradientAlgorithm
from rlcore.core import rl_utils

class FiniteDifference(RLGradientAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 num_passes=2,
                 episode_len=100):

        self.env = env
        self.policy = policy
        self.num_passes = num_passes
        self.episode_len = episode_len

        # TODO: compute number of weights in policy model
        self.num_weights = pass
        self.num_variations = self.num_passes * self.num_weights

        self.dJs = tf.placeholder(tf.float32,
            shape=(self.num_passes * self.num_weights, 1))
        self.dTs = tf.placeholder(tf.float32,
            shape=(self.num_passes * self.num_weights, self.num_weights))
        self.f_grad = tf.matrix_solve_ls(self.dTs, self.dJs)


    def optimize(self):
        # TODO: save initial weight values

        # run episode with initial weights to get J_ref
        ep_states, ep_raw_actions, ep_processed_actions, ep_rewards = rl_utils.rollout_env_with_policy(self.env,
                                                                                                       self.policy,
                                                                                                       episode_len=self.episode_len)
        J_ref = sum(ep_rewards)

        deltaJs = np.empty((self.num_passes*self.num_weights, 1))
        deltaTs = np.empty((self.num_passes*self.num_weights, self.num_weights))

        for i in range(self.num_variations):
            # adjust policy
            # TODO vary weights and apply
            deltas = pass

            # run one episode with new policy
            ep_states, ep_raw_actions, ep_processed_actions, ep_rewards = rl_utils.rollout_env_with_policy(self.env,
                                                                                                           self.policy,
                                                                                                           episode_len=self.episode_len)
            total_reward = sum(ep_rewards)
            print ("FD variation", i, "of", (num_variations), ", reward =", total_reward)
            deltaJs[i] = total_reward - J_ref
            deltaTs[i,:] = deltas.reshape((self.num_weights,))

        # TODO: restore initial weights
        grad = self.f_grad.eval(feed_dict={self.dTs: deltaTs, self.dJs: deltaJs})
        # TODO: apply gradient through tensorflow ops
