from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn

from rlflow.algos.algo import RLAlgorithm

class SARSA(RLAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 session,
                 exploration,
                 episode_len=np.inf,
                 discount=1.0,
                 standardize=True,
                 input_processor=None,
                 optimizer='adam',
                 learning_rate=0.1,
                 clip_gradients=(None, None)):

        super(SARSA, self).__init__(env,
                                    policy,
                                    session,
                                    episode_len,
                                    discount,
                                    standardize,
                                    input_processor,
                                    optimizer,
                                    learning_rate,
                                    clip_gradients)

        self.last_state = None
        self.last_action = None

        self.batch_size = 500
        self.current_batch = 0
        self.inputs = []
        self.estimates = []

        self.exploration = exploration

        self.states = self.policy.input_tensor
        self.q_values = self.policy.model

        self.actions = tf.placeholder(tf.int64, shape=[None])
        self.a_one_hot = tf.one_hot(self.actions, self.env.action_space.n, 1.0, 0.0)
        self.q_value = tf.reduce_sum(tf.mul(self.q_values, self.a_one_hot))

        self.td_estimate = tf.placeholder(tf.float32, shape=(None, 1))

        # MSE of action-value estimation
        self.L = tflearn.objectives.mean_square(self.q_values, self.td_estimate)
        self.grads_and_vars = self.opt.compute_gradients(self.L)

        if None not in self.clip_gradients:
            self.clipped_grads_and_vars = [(tf.clip_by_value(gv[0], self.clip_gradients[0], self.clip_gradients[1]), gv[1])
                                            for gv in self.grads_and_vars]
            self.update = self.opt.apply_gradients(self.clipped_grads_and_vars)
        else:
            self.update = self.opt.apply_gradients(self.grads_and_vars)

        self.sess.run(tf.initialize_all_variables())
        self.sess.graph.finalize()


    def act(self, obs):
        """
        Overriding act so can do proper exploration processing
        """
        if self.exploration.explore():
            return self.env.action_space.sample()
        else:
            # find max action
            return super(SARSA, self).act(obs)


    def on_step_completion(self, obs, action, reward, done, info, mode):
        """
        Receive data from the last step, add to memory
        """
        if mode == SARSA.TRAIN:

            if self.last_state is not None and self.last_action is not None:
                # use current step to do sarsa update
                qspap = np.squeeze(self.sess.run(self.q_values, feed_dict={self.states: obs.reshape(1, 4)}))[action]
                est = reward + self.discount * qspap

                # self.inputs.append(self.last_state)
                # self.estimates.append(est)

                inp = self.last_state.reshape(1, 4)
                est = np.array(est).reshape(1, 1)

                self.sess.run(self.update, feed_dict={self.states: inp, #self.inputs,
                                                      self.td_estimate: est,
                                                      self.actions: np.array([action])}) #self.estimates})
                self.inputs = []
                self.estimates = []


            # else this is the first state in the episode, either way
            # keep track of last state, if this is the end of an episode mark it
            self.last_state = obs if not done else None
            self.last_action = action if not done else None


    def optimize(self):
        """
        In this case all the work happens in the callbacks, just run an episode
        """
        return super(SARSA, self).optimize()
