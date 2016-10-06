from __future__ import print_function
import numpy as np
import tensorflow as tf

class FiniteDifference:

    def __init__(self, env, policy, num_passes=2):
        self.env = env
        self.policy = policy
        self.num_passes = num_passes

        self.dJs = tf.placeholder(tf.float32,
            shape=(self.num_passes * self.policy.get_num_weights(), 1))
        self.dTs = tf.placeholder(tf.float32,
            shape=(self.num_passes * self.policy.get_num_weights(),
            self.policy.get_num_weights()))

        self.f_grad = tf.matrix_solve_ls(self.dTs, self.dJs)


    def optimize(self, num_variations=None, episode_len=np.inf):
        self.policy.backup_weights()
        self.num_weights = self.policy.get_num_weights()
        if (num_variations == None):
            num_variations = self.num_passes * self.num_weights

        # run episode with initial weights to get J_ref
        J_ref = self.env.rollout_with_policy(self.policy, episode_len)

        deltaJs = np.empty((self.num_passes*self.num_weights, 1))
        deltaTs = np.empty((self.num_passes*self.num_weights, self.num_weights))

        for i in range(num_variations):
            # adjust policy
            deltas = self.policy.get_weight_variation()

            # run one episode with new policy
            total_reward = self.env.rollout_with_policy(self.policy, episode_len)
            print ("FD variation", i, "of", (num_variations), ", reward =", total_reward)
            deltaJs[i] = total_reward - J_ref
            deltaTs[i,:] = deltas.reshape((self.num_weights,))

        self.policy.restore_weights()
        grad = self.f_grad.eval(feed_dict={self.dTs: deltaTs, self.dJs: deltaJs})
        return grad
