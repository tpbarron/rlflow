from __future__ import print_function

import gym
import tensorflow as tf
import tflearn
from markov.policies.f_approx import Network
from markov.algos.grad import PolicyGradient

if __name__ == "__main__":
    env = gym.make("Pong-v0")

    with tf.Session() as sess:
        # Build neural network
        input_tensor = tflearn.input_data(shape=[None, 80, 80, 1])
        net = tflearn.conv_2d(input_tensor, 16, 4, strides=2, activation='relu')
        net = tflearn.conv_2d(input_tensor, 64, 4, strides=2, activation='relu')
        net = tflearn.flatten(net)
        net = tflearn.fully_connected(net, 16, activation='sigmoid')
        net = tflearn.fully_connected(net, env.action_space.n, activation='softmax')

        # initialize policy with network
        policy = Network(net,
                         sess,
                         Network.TYPE_PG)

        # initialize algorithm with env, policy, session and other params
        pg = PolicyGradient(env,
                            policy,
                            session=sess,
                            episode_len=1000,
                            discount=True,
                            optimizer='adam')

        # start the training process
        pg.train(max_episodes=5000)
        rewards = pg.test(episodes=10)
        print ("Average: ", float(sum(rewards)) / len(rewards))
