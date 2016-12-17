from __future__ import print_function

import numpy as np
import tensorflow as tf

from markov.core import rl_utils
from markov.algos.grad.grad_algo import RLGradientAlgorithm


class NaturalPG(RLGradientAlgorithm):
    """
    Natural policy gradient implementation based on Keras network
    """

    def __init__(self,
                 env,
                 policy,
                 episode_len=100,
                 discount=False,
                 optimizer='sgd'):

        raise NotImplementedError

        self.env = env
        self.policy = policy
        self.episode_len = episode_len
        self.discount = discount

        self.states = tf.placeholder(tf.float32, shape=(None, 4))
        self.actions = tf.placeholder(tf.float32, shape=(None, 2))
        self.rewards = tf.placeholder(tf.float32, shape=(None))
        self.probs = self.policy.model(self.states)

        self.action_probs = tf.mul(self.probs, self.actions)
        self.reduced_action_probs = tf.reduce_sum(self.action_probs, reduction_indices=[1])
        self.logprobs = tf.log(self.reduced_action_probs)
        self.eligibility = self.logprobs * self.rewards
        self.L = -tf.reduce_sum(self.eligibility)

        # fisher matrix
        self.F = tf.mul(self.logprobs, tf.transpose(self.logprobs))



        # TODO: gen optimizer based on param
        self.opt = tf.train.AdamOptimizer(0.005).minimize(self.L)

        # do gradient update separately so do apply custom function to gradients?
        # self.grads_and_vars = self.opt.compute_gradients(self.L)
        # self.apply_grads = self.opt.apply_gradients(self.grads_and_vars)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def optimize(self):
        ep_states, ep_raw_actions, ep_processed_actions, ep_rewards = rl_utils.rollout_env_with_policy(self.env,
                                                                                                       self.policy,
                                                                                                       episode_len=self.episode_len)

        # print ("raw actions: ", ep_raw_actions)
        # print ("processed actions: ", ep_processed_actions)

        if self.discount:
            ep_rewards = rl_utils.discount_rewards(np.array(ep_rewards))

        formatted_actions = np.zeros((len(ep_raw_actions), 2))
        for i in range(len(ep_processed_actions)):
            formatted_actions[i][ep_processed_actions[i]] = 1.0

        # formatted_rewards = ep_rewards
        formatted_rewards = np.zeros((len(ep_rewards),))
        # R_t is the reward from time t to the end
        running_sum = 0.0
        for t in range(len(ep_rewards)-1, -1, -1):
            running_sum += ep_rewards[t]
            formatted_rewards[t] = running_sum

        formatted_rewards -= np.mean(formatted_rewards)
        formatted_rewards /= np.std(formatted_rewards)

        # print ("formatted_actions: ", formatted_actions)
        # print ("States: ", ep_states)
        # print ("Rewards: ", formatted_rewards)
        #
        # probs = self.sess.run(self.action_probs, feed_dict={self.actions: formatted_actions,
        #                            self.states: ep_states,
        #                            self.rewards: formatted_rewards})
        #
        # print ("Probs: ", probs)
        # print ("Reduced: ", self.sess.run(self.reduced_action_probs, feed_dict={self.actions: formatted_actions,
        #                            self.states: ep_states,
        #                            self.rewards: formatted_rewards}))
        # print ("logprobs: ", self.sess.run(self.logprobs, feed_dict={self.actions: formatted_actions,
        #                            self.states: ep_states,
        #                            self.rewards: formatted_rewards}))
        # print ("eligibility: ", self.sess.run(self.eligibility, feed_dict={self.actions: formatted_actions,
        #                            self.states: ep_states,
        #                            self.rewards: formatted_rewards}))
        self.sess.run(self.opt, feed_dict={self.actions: formatted_actions,
                                   self.states: ep_states,
                                   self.rewards: formatted_rewards})

        # import sys
        # sys.exit()

        # grad_vals = self.sess.run([g for (g,v) in self.grads_and_vars], feed_dict={self.actions: formatted_actions,
        #                                                                            self.states: ep_states,
        #                                                                            self.rewards: formatted_rewards})
        # # print (len(grad_vals))
        # self.sess.run(self.apply_grads, feed_dict={self.actions: formatted_actions,
        #                                            self.states: ep_states,
        #                                            self.rewards: formatted_rewards})
