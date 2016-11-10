from __future__ import print_function

import sys
import numpy as np

import gym
from rlcore.core import rl_utils
from rlcore.policies.f_approx import Network
from rlcore.algos.grad import PolicyGradient

from keras.models import Sequential
from keras.layers import Dense

if __name__ == "__main__":
    env = gym.make("Pong-v0")

    model = Sequential()
    model.add(Dense(200, input_dim=6400, activation='relu'))
    model.add(Dense(env.action_space.n, activation='sigmoid'))
    network = Network(model, prediction_preprocessors=[rl_utils.prepro]) #,
                            #  prediction_postprocessors=[rl_utils.sample_outputs]) #,
                                                        # rl_utils.cast_int,
                                                        # rl_utils.pong_outputs])

    pg = PolicyGradient(env, network)

    max_itr = 2500
    max_episode_len = np.inf
    for i in range(max_itr):
        print ("Episode: ", i)
        pg.optimize(network, episode_len=max_episode_len)

        if (i % 100 == 0):
            reward = rl_utils.run_test_episode(env, network, episode_len=max_episode_len)
            print ("Reward: " + str(reward) + ", on iteration " + str(i))
