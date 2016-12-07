from __future__ import print_function

import gym
from rlcore.core import rl_utils
from rlcore.policies.f_approx import Network
from rlcore.algos.grad import PolicyGradient

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten

if __name__ == "__main__":
    env = gym.make("Breakout-v0")

    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            input_shape=env.observation_space.shape,
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dense(env.action_space.n, activation='softmax'))

    policy = Network(model, prediction_postprocessors=[rl_utils.sample, rl_utils.cast_int])
    pg = PolicyGradient(env, policy, episode_len=1000, discount=True)
    pg.train(max_iterations=5000, gym_record=False)

    average_reward = rl_utils.average_test_episodes(env, policy, 10, episode_len=1000)
    print ("Average: ", average_reward)
