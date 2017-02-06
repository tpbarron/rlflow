from __future__ import print_function

import gym
import tensorflow as tf
import tensorlayer as tl
from rlflow.core import tf_utils
from rlflow.policies.f_approx import Network
from rlflow.algos.grad import PolicyGradient
from rlflow.core.input import InputStreamDownsamplerProcessor, InputStreamSequentialProcessor, InputStreamProcessor

if __name__ == "__main__":
    env = gym.make("Pong-v0")

    w_init = tf.truncated_normal_initializer(stddev=0.05)
    b_init = tf.constant_initializer(value=0.0)

    name_scope = 'network'
    with tf.name_scope(name_scope) as scope:
        input_tensor = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name='policy_input_'+name_scope)
        net = tl.layers.InputLayer(input_tensor, name='input1_'+name_scope)
        net = tl.layers.Conv2d(net, 16, (8, 8), (4, 4), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_'+name_scope)
        net = tl.layers.Conv2d(net, 32, (4, 4), (2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_'+name_scope)
        net = tl.layers.FlattenLayer(net, name='flatten1_'+name_scope)
        net = tl.layers.DenseLayer(net, 1024, act=tf.nn.sigmoid, name='dense1_'+name_scope)
        net = tl.layers.DenseLayer(net, env.action_space.n, act=tf.nn.softmax, name='dense2_'+name_scope)

    downsampler = InputStreamDownsamplerProcessor((84, 84), gray=True)
    sequential = InputStreamSequentialProcessor(observations=4)
    input_processor = InputStreamProcessor(processor_list=[downsampler, sequential])


    # initialize policy with network
    policy = Network([input_tensor],
                     net,
                     Network.TYPE_PG)

    # initialize algorithm with env, policy, session and other params
    pg = PolicyGradient(env,
                        policy,
                        episode_len=1000,
                        discount=True,
                        input_processor=input_processor,
                        optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

    # start the training process
    pg.train(max_episodes=5000)
    rewards = pg.test(episodes=10)
    print ("Average: ", float(sum(rewards)) / len(rewards))
