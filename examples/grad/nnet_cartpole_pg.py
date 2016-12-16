from __future__ import print_function

import gym
import tensorflow as tf
import tflearn

from markov.core import tf_utils
from markov.policies.f_approx import Network
from markov.algos.grad import PolicyGradient

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    with tf.Session() as sess:
        # Build neural network
        input_tensor = tflearn.input_data(shape=tf_utils.get_input_tensor_shape(env))
        net = tflearn.fully_connected(input_tensor, 4, activation='sigmoid')
        net = tflearn.fully_connected(net, env.action_space.n, activation='softmax')

        policy = Network(net,
                         sess,
                         Network.TYPE_PG)

        pg = PolicyGradient(env,
                            policy,
                            session=sess,
                            episode_len=1000,
                            discount=True,
                            optimizer='adam')

        pg.train(max_episodes=5000)
        rewards = pg.test(episodes=10)
        print ("Average: ", float(sum(rewards)) / len(rewards))
