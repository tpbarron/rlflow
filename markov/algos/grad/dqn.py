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

        # vars to hold state updates
        self.last_state = None

        self.states = self.policy.input_tensort
        self.q_values = self.policy.model
        self.a = tf.placeholder(tf.int64, shape=[None])
        self.y = tf.placeholder(tf.float32, shape=[None])

        a_one_hot = tf.one_hot(self.a, self.env.action_space.n, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=[1])
        self.L = tflearn.mean_square(self.y, q_value)

        self.grads_and_vars = self.opt.compute_gradients(self.L)

        if None not in self.clip_gradients:
            self.clipped_grads_and_vars = [(tf.clip_by_value(gv[0], clip_gradients[0], clip_gradients[1]), gv[1])
                                            for gv in self.grads_and_vars]
            self.update = self.opt.apply_gradients(self.clipped_grads_and_vars)
        else:
            self.update = self.opt.apply_gradients(self.grads_and_vars)

        self.sess.run(tf.initialize_all_variables())



    def reset(self):
        return super(DQN, self).reset()


    def act(self, obs):
        """
        Overriding act so can do proper exploration processing,
        add to memory and sample from memory for updates
        """
        if self.exploration.explore():
            return self.env.action_space.sample()
        else:
            # find max action
            return super(DQN, self).act(obs)


    def step_callback(self, obs, action, reward, done, info):
        """
        Receive data from the last step, add to memory
        """
        if self.last_state is not None:
            # then this is not the first state seen
            self.memory.add_element([self.last_state, action, reward, obs])
        else:
            # this is the first state in the episode
            self.last_state = obs

        if done:
            # if this is the end of an episode mark that
            self.last_state = None


    def optimize(self):
        ep_states, ep_actions, ep_rewards, _ = self.run_episode()

        if self.discount != 1.0:
            ep_rewards = rl_utils.discount_rewards(np.array(ep_rewards), gamma=self.discount)

        formatted_actions = np.zeros((len(ep_actions), self.env.action_space.n))
        for i in range(len(ep_actions)):
            formatted_actions[i][ep_actions[i]] = 1.0

        formatted_rewards = ep_rewards
        if self.standardize:
            formatted_rewards = rl_utils.standardize_rewards(formatted_rewards)

        self.sess.run(self.update, feed_dict={self.actions: formatted_actions,
                                              self.states: ep_states,
                                              self.rewards: formatted_rewards})
