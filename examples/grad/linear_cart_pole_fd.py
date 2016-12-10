from __future__ import print_function

import sys
import numpy as np

import gym
from markov.core import rl_utils
from markov.policies.f_approx import LinearApproximator
from markov.algos.grad import FiniteDifference

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    lin_approx = LinearApproximator(env.observation_space.shape[0],
                                    1,
                                    lr=0.01,
                                    prediction_postprocessor=rl_utils.sign)

    fd = FiniteDifference(env, num_passes=2)

    max_itr = 2500
    max_episode_len = 1000
    for i in range(max_itr):
        grad = fd.optimize(lin_approx, episode_len=max_episode_len)
        lin_approx.update(grad)
        reward = rl_utils.run_test_episode(env, lin_approx, episode_len=max_episode_len)
        print ("Reward: " + str(reward) + ", on iteration " + str(i))
