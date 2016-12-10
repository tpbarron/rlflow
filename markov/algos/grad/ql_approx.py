from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn

from markov.algos.grad.grad_algo import RLGradientAlgorithm
from markov.core import rl_utils

class QLearning(RLGradientAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 session,
                 epsilon=0.1,
                 episode_len=np.inf,
                 discount=1.0,
                 standardize=True,
                 optimizer='adam',
                 learning_rate=0.1,
                 clip_gradients=(None, None)):

        super(QLearning, self).__init__(env,
                                        policy,
                                        session,
                                        episode_len,
                                        discount,
                                        standardize,
                                        optimizer,
                                        learning_rate,
                                        clip_gradients)

        self.epsilon = epsilon
        self.state_action = self.policy.input_tensor
        self.values = self.policy.model
        self.td_value_estimate = tf.placeholder(tf.float32, shape=(None, 1))

        # MSE of action-value estimation
        self.L = tflearn.objectives.mean_square(self.values, self.td_value_estimate)
        self.grads_and_vars = self.opt.compute_gradients(self.L)

        if None not in self.clip_gradients:
            self.clipped_grads_and_vars = [(tf.clip_by_value(gv[0], self.clip_gradients[0], self.clip_gradients[1]), gv[1])
                                            for gv in self.grads_and_vars]
            self.update = self.opt.apply_gradients(self.clipped_grads_and_vars)
        else:
            self.update = self.opt.apply_gradients(self.grads_and_vars)

        self.batch_size = 500
        self.current_batch = 0
        self.inputs = []
        self.estimates = []

        self.sess.run(tf.initialize_all_variables())



    def ql_estimate(self, s, a, r, sp, max_a):
        """
        Q(s, a) = Q(s, a) + alpha * (r + discount * max_a' Q(s', a') - Q(s, a))
        """
        # qsa = self.policy.predict(np.append(s, [a]))
        qspap_max = self.policy.predict(np.append(sp, [max_a]))

        est = r + self.discount * qspap_max
        return est


    def max_action(self, state):
        # now return the max action given the state
        # the representation could either be
        # (state + action) -> value
        # or
        # (state) -> [set of action values]
        values = []
        for a in range(self.env.action_space.n):
            i = np.append(state, [a])
            val = self.policy.predict(i)
            values.append(float(val))

        index = max(xrange(len(values)), key = lambda x: values[x])
        # print (index, values)
        return index


    def optimize(self):
        """
        Run a single episode and perform QL updates along the way
        """
        ep_steps = 0
        done = False
        state = self.env.reset()

        while not done and ep_steps < self.episode_len:
            self.env.render()
            state = rl_utils.apply_prediction_preprocessors(self.policy, state)
            if np.random.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.max_action(state)

            action = rl_utils.apply_prediction_postprocessors(self.policy, action)
            next_state, reward, done, _ = self.env.step(action)
            max_action = self.max_action(next_state)

            td_estimate = self.ql_estimate(state, action, reward, next_state, max_action)
            st_act_pair = np.append(state, [action])#.reshape(1, 5)

            self.inputs.append(st_act_pair)
            self.estimates.extend(td_estimate)
            self.current_batch += 1

            if self.current_batch >= self.batch_size:
                print ("Doing update")
                self.sess.run(self.update, feed_dict={self.state_action: self.inputs,
                                                      self.td_value_estimate: self.estimates})
                self.current_batch = 0
                self.inputs = []
                self.estimates = []

            # self.sess.run(self.update, feed_dict={self.state_action: st_act_pair,
            #                                       self.td_value_estimate: td_estimate})

            state = next_state
            ep_steps += 1


    def step(self):
        pass


    def reset(self):
        pass
