from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn

from .td_algo import RLTDAlgorithm

class DQN(RLTDAlgorithm):
    """
    Basic deep q network implementation based on TFLearn network
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
                 memory_init_size=5000,
                 clone_frequency=10000):

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
        self.clone_frequency = clone_frequency
        self.steps = 0

        # vars to hold state updates
        self.last_state = None

        self.states = self.policy.input_tensor
        self.q_values = self.policy.model

        self.target_states = self.policy.clone_input_tensor
        self.target_q_values = self.policy.clone_model

        self.actions = tf.placeholder(tf.int64, shape=[None])
        self.a_one_hot = tf.one_hot(self.actions, self.env.action_space.n, 1.0, 0.0)
        self.q_value = tf.reduce_sum(tf.mul(self.q_values, self.a_one_hot))
        self.y = tf.placeholder(tf.float32, shape=[None])

        self.L = tflearn.objectives.mean_square(self.q_value, self.y)
        self.grads_and_vars = self.opt.compute_gradients(self.L, var_list=tf.trainable_variables())

        if None not in self.clip_gradients:
            self.clipped_grads_and_vars = [(tf.clip_by_value(gv[0], clip_gradients[0], clip_gradients[1]), gv[1])
                                           for gv in self.grads_and_vars]
            self.update = self.opt.apply_gradients(self.clipped_grads_and_vars)
        else:
            self.update = self.opt.apply_gradients(self.grads_and_vars)

        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()


    def act(self, obs):
        """
        Overriding act so can do proper exploration processing,
        add to memory and sample from memory for updates
        """
        if self.memory.size() < self.memory_init_size:
            return self.env.action_space.sample()

        if self.exploration.explore():
            return self.env.action_space.sample()
        else:
            # find max action
            return super(DQN, self).act(obs)


    def on_step_completion(self, obs, action, reward, done, info, mode):
        """
        Receive data from the last step, add to memory
        """
        if mode == DQN.TRAIN:
            # last state is none if this is the start of an episode
            # obs is None until the input processor provides valid processing
            if self.last_state is not None and obs is not None:
                # then this is not the first state seen
                self.memory.add_element([self.last_state, action, reward, obs, done])

            # else this is the first state in the episode, either way
            # keep track of last state, if this is the end of an episode mark it
            self.last_state = obs if not done else None

            if self.memory.size() >= self.memory_init_size:
                # mark that we have done another step for epsilon decrease
                self.exploration.increment_iteration()

                samples = self.memory.sample(self.sample_size)
                states, actions, rewards, next_states, terminals = [], [], [], [], []
                for s in samples:
                    states.append(s.S1)
                    actions.append(s.A)
                    rewards.append(s.R)
                    next_states.append(s.S2)
                    terminals.append(s.T)

                terminals = np.array(terminals) + 0
                next_states = np.stack(next_states)

                with self.policy.clone_graph.as_default():
                    target_qs = self.policy.clone_sess.run(self.target_q_values, feed_dict={self.target_states: next_states})

                ys = rewards + (1 - terminals) * self.discount * np.max(target_qs, axis=1)

                self.sess.run(self.update,
                              feed_dict={self.states: states,
                                         self.actions: actions,
                                         self.y: ys})

                # if at desired step, clone model
                if self.steps % self.clone_frequency == 0:
                    self.policy.clone()


    def optimize(self):
        """
        In this case all the work happens in the callbacks, just run an episode
        """
        return super(DQN, self).optimize()
