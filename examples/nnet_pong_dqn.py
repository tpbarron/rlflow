from __future__ import print_function

import gym
import tensorflow as tf

from rlflow.policies.f_approx import Network
from rlflow.algos.td import DQN
from rlflow.memories import ExperienceReplay
from rlflow.exploration.egreedy import EpsilonGreedy
from rlflow.core.input import InputStreamDownsamplerProcessor, InputStreamSequentialProcessor, InputStreamProcessor


def build_network(name_scope, env):
    w_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d()
    w_init_dense = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(value=0.0)

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

    return [input_tensor], [net]


if __name__ == "__main__":
    env = gym.make("Pong-v0")

    inputs, outputs = build_network("train_policy", env)
    # Note the scope on the Network params. This is important, otherwise
    # we will get trainable params from the clone net in our updates also
    network = Network(inputs, outputs, scope="train_policy")

    clone_inputs, clone_outputs = build_network("clone_policy", env)
    clone_network = Network(clone_inputs, clone_outputs, scope="clone_policy")

    memory = ExperienceReplay(state_shape=(84, 84, 4), max_size=50000)
    egreedy = EpsilonGreedy(0.9, 0.1, 1000000)

    downsampler = InputStreamDownsamplerProcessor((84, 84), gray=True)
    sequential = InputStreamSequentialProcessor(observations=4)
    input_processor = InputStreamProcessor(processor_list=[downsampler, sequential])

    opt = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)

    dqn = DQN(env,
              network,
              clone_network,
              memory,
              egreedy,
              input_processor=input_processor,
              discount=0.99,
              optimizer=opt,
              memory_init_size=50000,
              clip_gradients=(-10.0, 10.0),
              clone_frequency=5000)

    dqn.train(max_episodes=1000000, test_frequency=10, save_frequency=100)

    rewards = dqn.test(episodes=10)
    print ("Avg test reward: ", float(sum(rewards)) / len(rewards))
