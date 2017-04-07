from __future__ import print_function

import gym
import tensorflow as tf
import numpy as np
from rlflow.core import tf_utils
from rlflow.policies.f_approx import Network
from rlflow.algos.grad import PolicyGradient
from rlflow.core.input import InputStreamDownsamplerProcessor, InputStreamSequentialProcessor, InputStreamProcessor


def build_network(name_scope, env):
    w_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d()
    w_init_dense = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(value=0.1)

    with tf.variable_scope(name_scope):
        input_tensor = tf.placeholder(tf.float32,
                                      shape=[None, 84, 84, 4],
                                      name='policy_input_'+name_scope)
        net = tf.contrib.layers.convolution2d(input_tensor,
                                              16, (8, 8), (4, 4),
                                              "VALID",
                                              activation_fn=tf.nn.relu,
                                              weights_initializer=w_init_conv2d,
                                              scope='conv1_'+name_scope)
        net = tf.contrib.layers.convolution2d(net,
                                              16, (8, 8), (4, 4),
                                              "VALID",
                                              activation_fn=tf.nn.relu,
                                              weights_initializer=w_init_conv2d,
                                              scope='conv2_'+name_scope)
        net = tf.contrib.layers.flatten(net,
                                        scope='flatten1_'+name_scope)
        net = tf.contrib.layers.fully_connected(net,
                                                1024,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=w_init_dense,
                                                scope='dense1_'+name_scope)
        net = tf.contrib.layers.fully_connected(net,
                                                env.action_space.n,
                                                weights_initializer=w_init_dense,
                                                scope='dense2_'+name_scope)
        net = tf.contrib.layers.softmax(net)

    return [input_tensor], [net]


if __name__ == "__main__":
    env = gym.make("Pong-v0")

    inputs, outputs = build_network("policy", env)
    network = Network(inputs, outputs)

    downsampler = InputStreamDownsamplerProcessor((84, 84), gray=True, scale=True)
    sequential = InputStreamSequentialProcessor(observations=4)
    input_processor = InputStreamProcessor(processor_list=[downsampler, sequential])

    # initialize algorithm with env, policy, session and other params
    pg = PolicyGradient(env,
                        network,
                        episode_len=np.inf,
                        discount=0.99,
                        input_processor=input_processor,
                        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001))

    # start the training process
    pg.train(max_episodes=5000)
    rewards = pg.test(episodes=10)
    print ("Average: ", float(sum(rewards)) / len(rewards))
