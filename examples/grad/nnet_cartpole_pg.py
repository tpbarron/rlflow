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
    env = gym.make("CartPole-v0")

    model = Sequential()
    model.add(Dense(4, input_dim=env.observation_space.shape[0], activation='sigmoid'))
    model.add(Dense(env.action_space.n, activation='softmax'))
    network = Network(model, prediction_postprocessors=[rl_utils.sample, rl_utils.cast_int])
    pg = PolicyGradient(env, network, episode_len=1000, discount=True)
    pg.train(max_iterations=2500, gym_record=False)
