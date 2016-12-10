from __future__ import print_function

import gym
import tensorflow as tf
import tflearn

from markov.policies.f_approx import LinearApproximator
from markov.algos.grad.ql_approx import QLearning
from markov.algos.grad.sarsa_approx import SARSA
from markov.core import rl_utils


if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    with tf.Session() as sess:
        in_dimen = env.observation_space.shape[0] + 1 #env.action_space.n
        out_dimen = 1

        input_tensor = tflearn.input_data(shape=[None, in_dimen])
        linear = tflearn.fully_connected(input_tensor, out_dimen)
        # tflearn.single_unit(input_tensor)

        lin_approx = LinearApproximator(input_tensor,
                                        linear,
                                        sess)

        ql = SARSA(env,
                       lin_approx,
                       sess,
                       episode_len=100,
                       discount=0.9,
                       learning_rate=0.01,
                       optimizer='adam',
                       clip_gradients=(-1, 1))

        ql.train(max_iterations=10000, gym_record=False)

        # average_reward = rl_utils.average_test_episodes(env,
        #                                                 lin_approx,
        #                                                 10,
        #                                                 episode_len=ql.episode_len)
        # print ("Average: ", average_reward)
