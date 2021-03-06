from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn

from rlflow.core import rl_utils
from rlflow.core.output import output_processors
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
                 optimizer=None,
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

        self.probs = self.policy.outputs[0]
        self.action_probs = tf.multiply(self.probs, self.actions)
        self.reduced_action_probs = tf.reduce_sum(self.action_probs, reduction_indices=[1])
        self.logprobs = tf.log(self.reduced_action_probs)

        # vanilla gradient = mul(sum(logprobs * rewards))
        self.L = -tf.reduce_sum(tf.multiply(self.logprobs, self.rewards))
        self.grads_and_vars = self.opt.compute_gradients(self.L)

        if None not in self.clip_gradients:
            self.clipped_grads_and_vars = [(tf.clip_by_value(gv[0], clip_gradients[0], clip_gradients[1]), gv[1])
                                            for gv in self.grads_and_vars]
            self.update = self.opt.apply_gradients(self.clipped_grads_and_vars)
        else:
            self.update = self.opt.apply_gradients(self.grads_and_vars)


        # sampling
        self.policy_output = tf.placeholder(tf.float32, shape=(None, env.action_space.n))
        self.policy_sample = output_processors.pg_sample(self.policy_output)

        # TODO: if baseline is set, learn a critic
        self.baseline = baseline



    def act(self, obs, mode):
        """
        Override action procedure to take policy output and sample as probabilities
        """
        pol_out = np.squeeze(super(PolicyGradient, self).act(obs, mode))
        # print (pol_out)
        # print (pol_out, sum(pol_out))
        return np.random.choice(range(len(pol_out)), p=np.squeeze(pol_out))
        # return self.sess.run(self.policy_sample, feed_dict={self.policy_output: pol_out})


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

        # print (formatted_actions)
        # print (ep_states[4:])
        # print (np.array(ep_states[4:]).shape)
        # print (formatted_rewards)
        self.sess.run(self.update, feed_dict={self.actions: formatted_actions,
                                              self.states: np.array(ep_states),
                                              self.rewards: formatted_rewards})
        return ep_states, ep_actions, ep_rewards, ep_infos
