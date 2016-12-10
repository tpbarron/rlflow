from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn

from rlcore.algos.grad.grad_algo import RLGradientAlgorithm
from rlcore.core import rl_utils

class SARSA(RLGradientAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 session,
                 episode_len=np.inf,
                 discount=1.0,
                 standardize=True,
                 optimizer='adam',
                 learning_rate=0.1,
                 clip_gradients=(None, None)):

        super(SARSA, self).__init__(env,
                                        policy,
                                        session,
                                        episode_len,
                                        discount,
                                        standardize,
                                        optimizer,
                                        learning_rate,
                                        clip_gradients)

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



    def sarsa_estimate(self, s, a, r, sp, max_a):
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
        # import sys
        # sys.exit()
        return index


    def optimize(self):
        """
        Run a single episode and perform QL updates along the way
        """
        ep_steps = 0
        done = False

        state = self.env.reset()
        state = rl_utils.apply_prediction_preprocessors(self.policy, state)
        action = self.max_action(state)
        action = rl_utils.apply_prediction_postprocessors(self.policy, action)

        while not done and ep_steps < self.episode_len:
            self.env.render()
            next_state, reward, done, info = self.env.step(action)
            next_state = rl_utils.apply_prediction_preprocessors(self.policy, next_state)

            next_action = self.max_action(next_state)
            next_action = rl_utils.apply_prediction_postprocessors(self.policy, next_action)

            td_estimate = self.sarsa_estimate(state, action, reward, next_state, next_action)
            st_act_pair = np.append(state, [action])#.reshape(1, 5)

            self.inputs.append(st_act_pair)
            self.estimates.extend(td_estimate)

            state = next_state
            action = next_action

            ep_steps += 1

        self.sess.run(self.update, feed_dict={self.state_action: self.inputs,
                                              self.td_value_estimate: self.estimates})
        self.inputs = []
        self.estimates = []


    def step(self):
        pass


    def reset(self):
        pass
