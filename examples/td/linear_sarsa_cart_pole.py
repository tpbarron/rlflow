from __future__ import print_function

import gym
import tensorflow as tf
import tflearn

from markov.policies.f_approx import LinearApproximator
from markov.algos.td import SARSA
from markov.exploration import EpsilonGreedy


if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    with tf.Session() as sess:
        in_dimen = env.observation_space.shape[0] # + 1 #env.action_space.n
        out_dimen = env.action_space.n

        input_tensor = tflearn.input_data(shape=[None, in_dimen])
        linear = tflearn.fully_connected(input_tensor, out_dimen)

        lin_approx = LinearApproximator(linear,
                                        sess,
                                        LinearApproximator.TYPE_DQN)

        egreedy = EpsilonGreedy(0.9)

        sarsa = SARSA(env,
                      lin_approx,
                      sess,
                      egreedy,
                      episode_len=100,
                      discount=0.9,
                      learning_rate=0.001,
                      optimizer='adam',
                      clip_gradients=(-1, 1))

        sarsa.train(max_episodes=10000)
        rewards = sarsa.test(episodes=10)
        print ("Average: ", float(sum(rewards))/len(rewards))
