from __future__ import print_function

import numpy as np
# from keras.layers import Input
# from keras import backend as K
# from keras.optimizers import RMSprop
import tensorflow as tf
# import tensorlayer as tl

from rlflow.algos.algo import RLAlgorithm
from rlflow.core import tf_utils

class DQN(RLAlgorithm):
    """
    Basic deep q network implementation based on TFLearn network
    """

    def __init__(self,
                 env,
                 train_policy,
                 clone_policy,
                 memory,
                 exploration,
                 episode_len=np.inf,
                 discount=1.0,
                 standardize=True,
                 input_processor=None,
                 optimizer='rmsprop',
                 clip_gradients=(None, None),
                 sample_size=32,
                 memory_init_size=5000,
                 clone_frequency=10000,
                 test_epsilon=0.05):

        super(DQN, self).__init__(env,
                                  clone_policy, # pass clone policy to super since that is the default for action selection
                                  episode_len,
                                  discount,
                                  standardize,
                                  input_processor,
                                  optimizer,
                                  clip_gradients)

        self.train_policy = train_policy
        self.memory = memory
        self.exploration = exploration
        self.sample_size = sample_size
        self.memory_init_size = memory_init_size
        self.clone_frequency = clone_frequency
        self.test_epsilon = test_epsilon
        self.steps = 0

        self.clone_ops = tf_utils.build_policy_copy_ops(self.train_policy, self.policy)

        # vars to hold state updates
        self.last_state = None

        self.train_states = self.train_policy.inputs[0] # self.train_policy.inputs[0]
        self.train_q_values = self.train_policy.outputs[0] # self.train_policy.output

        self.target_states = self.policy.inputs[0] #self.policy.inputs[0]
        self.target_q_values = self.policy.outputs[0] # self.self.policy.output

        # self.S1 = Input(shape=(84, 84, 4))
        # self.S2 = Input(shape=(84, 84, 4))
        # self.A = Input(shape=(1,), dtype='int32')
        # self.R = Input(shape=(1,), dtype='float32')
        # self.T = Input(shape=(1,), dtype='float32')

        # VS = self.train_q_values #self.train_policy.model(self.S1)
        # VNS = self.target_q_values #self.policy.model(self.S2)
        #
        # future_value = (1-self.T) * K.max(VNS, axis=1, keepdims=True)
        #
        # print ("Past max")
        # discounted_future_value = self.discount * future_value
        # target = self.R + discounted_future_value
        #
        # cost = (VS[:, self.A] - target)#**2).mean()
        # opt = RMSprop(0.0001)
        # updates = opt.get_updates(self.policy.model.trainable_weights, [], cost)
        # self.update = K.function([self.train_states, self.target_states, self.A, self.R, self.T], cost, updates=updates)


        self.actions = tf.placeholder(tf.int64, shape=[None])
        self.a_one_hot = tf.one_hot(self.actions, self.env.action_space.n, 1.0, 0.0)

        # This used to reduce the q-value to a single number!
        # I don't think that is what I want. I want a list of q-values and a list of targets
        # This should be bettwe with axis=1
        self.q_estimates = tf.reduce_sum(tf.multiply(self.train_q_values, self.a_one_hot), axis=1)
        self.q_targets = tf.placeholder(tf.float32, shape=[None])


        self.delta = self.q_targets - self.q_estimates
        self.clipped_error = tf.where(tf.abs(self.delta) < 1.0,
                                    0.5 * tf.square(self.delta),
                                    tf.abs(self.delta) - 0.5, name='clipped_error')
        self.L = tf.reduce_mean(self.clipped_error, name='loss')

        self.grads_and_vars = self.opt.compute_gradients(self.L, var_list=self.train_policy.get_params())
        # for idx, (grad, var) in enumerate(self.grads_and_vars):
        #     if grad is not None:
        #         self.grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
        self.update = self.opt.apply_gradients(self.grads_and_vars)

        # self.L = tf_utils.mean_square(self.q_value, self.y)
        # self.grads_and_vars = self.opt.compute_gradients(self.L, var_list=self.train_policy.get_params())
        #
        # if None not in self.clip_gradients:
        #     self.clipped_grads_and_vars = [(tf.clip_by_value(gv[0], clip_gradients[0], clip_gradients[1]), gv[1])
        #                                    for gv in self.grads_and_vars]
        #     self.update = self.opt.apply_gradients(self.clipped_grads_and_vars)
        # else:
        #     self.update = self.opt.apply_gradients(self.grads_and_vars)


    def clone(self):
        """
        Run the clone ops
        """
        # print ("Cloning")
        self.sess.run(self.clone_ops)

        # self.train_policy.model.get_weights()
        # self.policy.model.set_weights(self.train_policy.model.get_weights())
        # v1 = self.sess.run(self.train_policy.get_params()[0])
        # v2 = self.sess.run(self.policy.get_params()[0])
        # print (np.allclose(v1, v2))
        # import sys
        # sys.exit()


    def on_train_start(self):
        """
        Run the clone ops to make networks same at start
        """
        self.clone()


    def max_action(self, obs, mode):
        actions = super(DQN, self).act(obs, mode)
        return np.argmax(actions)


    def act(self, obs, mode):
        """
        Overriding act so can do proper exploration processing,
        add to memory and sample from memory for updates
        """
        if mode == RLAlgorithm.TRAIN:
            if self.memory.size() < self.memory_init_size:
                return self.env.action_space.sample()

            if self.exploration.explore(self.steps):
                return self.env.action_space.sample()
            else:
                # find max action
                return self.max_action(obs, mode)
                # return super(DQN, self).act(obs, mode)
        else:
            if np.random.random() < self.test_epsilon:
                return self.env.action_space.sample()
            else:
                return self.max_action(obs, mode)
                # return super(DQN, self).act(obs, mode)


    def on_step_finish(self, obs, action, reward, done, info, mode):
        """
        Receive data from the last step, add to memory
        """
        if mode == RLAlgorithm.TRAIN:
            # clip reward between [-1, 1]
            reward = reward if abs(reward) <= 1.0 else float(reward)/reward

            # last state is none if this is the start of an episode
            # obs is None until the input processor provides valid processing
            if self.last_state is not None and obs is not None:
                # then this is not the first state seen
                self.memory.add_element(self.last_state, action, reward, obs, done)

            # else this is the first state in the episode, either way
            # keep track of last state, if this is the end of an episode mark it
            self.last_state = obs if not done else None

            if self.memory.size() >= self.memory_init_size:
                # mark that we have done another step for epsilon decrease
                # self.exploration.increment_iteration()

                states, actions, rewards, next_states, terminals = self.memory.sample(self.sample_size)

                # his takes about 0.01 seconds on my laptop
                # print ("next states: ", next_states.shape)
                target_qs = self.sess.run(self.target_q_values, feed_dict={self.target_states: next_states})
                ys = rewards + (1 - terminals) * self.discount * np.max(target_qs, axis=1)

                # Is there a performance issue here? this takes about 0.07 seconds on my laptop
                self.sess.run(self.update,
                              feed_dict={self.train_states: states,
                                         self.actions: actions,
                                         self.q_targets: ys})

                # if at desired step, clone model
                if self.steps % self.clone_frequency == 0:
                    # print ("Step ", self.steps, ", cloning model")
                    self.clone()

                self.steps += 1
        else: # TEST mode
            pass


    def optimize(self):
        """
        In this case all the work happens in the callbacks, just run an episode
        """
        print ("Current step: ", self.steps)
        return super(DQN, self).optimize()
