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
                 exploration,
                 episode_len=np.inf,
                 discount=1.0,
                 standardize=True,
                 input_processor=None,
                 learning_rate=0.01,
                 optimizer='sgd',
                 clip_gradients=(None, None),
                 sample_size=32,
                 memory_init_size=1000):

        super(DQN, self).__init__(env,
                                 policy,
                                 session,
                                 episode_len,
                                 discount,
                                 standardize,
                                 input_processor,
                                 optimizer,
                                 learning_rate,
                                 clip_gradients)

        self.memory = memory
        self.exploration = exploration
        self.sample_size = sample_size
        self.memory_init_size = memory_init_size

        # vars to hold state updates
        self.last_state = None

        self.states = self.policy.input_tensor
        self.q_value = self.policy.model
        self.a = tf.placeholder(tf.int64, shape=[None])
        self.y = tf.placeholder(tf.float32, shape=[None])

        # # TODO: it would be nice to put the next two lines into the policy
        # a_one_hot = tf.one_hot(self.a, self.env.action_space.n, 1.0, 0.0)
        # q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=[1])

        self.L = tflearn.mean_square(self.y, self.q_value)
        self.grads_and_vars = self.opt.compute_gradients(self.L)
        print (self.grads_and_vars)
        
        if None not in self.clip_gradients:
            self.clipped_grads_and_vars = [(tf.clip_by_value(gv[0], clip_gradients[0], clip_gradients[1]), gv[1])
                                            for gv in self.grads_and_vars]
            self.update = self.opt.apply_gradients(self.clipped_grads_and_vars)
        else:
            self.update = self.opt.apply_gradients(self.grads_and_vars)

        self.sess.run(tf.global_variables_initializer())


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
        # mark that we have done another step for epsilon decrease
        self.exploration.increment_iteration()
        if self.last_state is not None:
            # then this is not the first state seen
            self.memory.add_element([self.last_state, action, reward, obs])

        # else this is the first state in the episode, either way
        # keep track of last state
        self.last_state = obs

        if done:
            # if this is the end of an episode mark that
            self.last_state = None

        if self.memory.size() > self.memory_init_size:
            sample = self.memory.sample(self.sample_size)
            print (sample)
            # self.sess.run(self.update, feed_dict={self.states: sample,
            #                                       })


    def optimize(self):
        """
        In this case all the work happens in the callbacks
        """
        self.run_episode()
