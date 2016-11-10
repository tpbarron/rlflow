from __future__ import print_function

import numpy as np
import tensorflow as tf

from rlcore.core import rl_utils
from rlcore.algos.grad.grad_algo import RLGradientAlgorithm

class PolicyGradient(RLGradientAlgorithm):
    """
    Basic stochastic policy gradient implementation based on Keras network
    """

    def __init__(self,
                 env,
                 policy,
                 episode_len=100,
                 discount=False,
                 optimizer='sgd'):

        self.env = env
        self.policy = policy
        self.episode_len = episode_len
        self.discount = discount

        self.states = tf.placeholder(tf.float32, shape=(None, 4))
        self.actions = tf.placeholder(tf.float32, shape=(None, 2))
        self.rewards = tf.placeholder(tf.float32, shape=(None))
        self.probs = self.policy.model(self.states)

        self.logprobs = tf.log(tf.reduce_sum(tf.mul(self.probs, self.actions), reduction_indices=[1]))
        self.eligibility = self.logprobs * self.rewards
        self.L = -tf.reduce_sum(self.eligibility)
        # self.L = -tf.reduce_sum(tf.mul(tf.mul(tf.log(self.probs), self.actions), self.rewards))

        # TODO: gen optimizer based on param
        self.opt = tf.train.AdamOptimizer(0.01).minimize(self.L)

        # do gradient update separately so do apply custom function to gradients?
        # self.grads_and_vars = self.opt.compute_gradients(self.L)
        # self.apply_grads = self.opt.apply_gradients(self.grads_and_vars)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def optimize(self):
        ep_states, ep_raw_actions, ep_processed_actions, ep_rewards = rl_utils.rollout_env_with_policy(self.env,
                                                                                                       self.policy,
                                                                                                       episode_len=self.episode_len)

        # print ("Actions: ", ep_processed_actions)
        # print ("States: ", ep_states)
        # print ("Rewards: ", ep_rewards)
        if (self.discount):
            ep_rewards = rl_utils.discount_rewards(np.array(ep_rewards))

        formatted_actions = np.zeros((len(ep_raw_actions), 2))
        for i in range(len(ep_processed_actions)):
            formatted_actions[ep_processed_actions[i]] = 1.0

        formatted_rewards = np.zeros((len(ep_rewards),))
        # R_t is the reward from time t to the end
        running_sum = 0.0
        for t in range(len(ep_rewards)-1, -1, -1):
            running_sum += ep_rewards[t]
            formatted_rewards[t] = running_sum

        formatted_rewards -= np.mean(formatted_rewards)
        formatted_rewards /= np.std(formatted_rewards)

        self.sess.run(self.opt, feed_dict={self.actions: formatted_actions,
                                   self.states: ep_states,
                                   self.rewards: formatted_rewards})

        # grad_vals = self.sess.run([g for (g,v) in self.grads_and_vars], feed_dict={self.actions: formatted_actions,
        #                                                                            self.states: ep_states,
        #                                                                            self.rewards: formatted_rewards})
        # # print (len(grad_vals))
        # self.sess.run(self.apply_grads, feed_dict={self.actions: formatted_actions,
        #                                            self.states: ep_states,
        #                                            self.rewards: formatted_rewards})
