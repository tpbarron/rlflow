from __future__ import print_function

import gym
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Flatten, Convolution2D
import keras.backend as K

from rlflow.policies.f_approx import Network
from rlflow.algos.td import DQN
from rlflow.memories import ExperienceReplay
from rlflow.exploration.egreedy import EpsilonGreedy
from rlflow.core.input import InputStreamDownsamplerProcessor, InputStreamSequentialProcessor, InputStreamProcessor
from rlflow.core import tf_utils

K.set_session(tf_utils.get_tf_session())

def build_network(env, name_scope):
    """
    See this Keras blog post (https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html)
    for more information
    """
    with tf.variable_scope(name_scope):
        inputs = tf.placeholder(tf.float32, shape=tf_utils.get_input_tensor_shape(env))
        net = Dense(32, activation='sigmoid')(inputs)
        q_values = Dense(env.action_space.n)(net)
        return [inputs], [q_values]


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    inputs, outputs = build_network(env, "train_policy")
    network = Network(inputs, outputs, scope="train_policy")

    clone_inputs, clone_outputs = build_network(env, "clone_policy")
    clone_network = Network(clone_inputs, clone_outputs, scope="clone_policy")

    memory = ExperienceReplay(state_shape=tf_utils.get_input_tensor_shape(env)[1:], max_size=100000)
    egreedy = EpsilonGreedy(0.9, 0.1, 100000)

    opt = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)

    dqn = DQN(env,
              network,
              clone_network,
              memory,
              egreedy,
              discount=0.99,
              optimizer=opt,
              memory_init_size=5000,
              clip_gradients=(-10.0, 10.0),
              clone_frequency=5000)

    dqn.train(max_episodes=1000000, test_frequency=250, save_frequency=250)

    rewards = dqn.test(episodes=10)
    print ("Avg test reward: ", float(sum(rewards)) / len(rewards))
