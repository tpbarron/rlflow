from __future__ import print_function

import sys, time
import numpy as np
import tensorflow as tf
import gym
from rlcore.policies.f_approx.linear_tf import LinearApproximator
from rlcore.algos.grad.finite_diff_tf import FiniteDifference


if __name__ == "__main__":
    env = gym.make("CartPole-v0") #normalize(CartpoleEnv())

    max_itr = 2500
    max_episode_len = 100

    lin_approx = LinearApproximator(env.observation_space.flat_dim, env.action_dim, lr=0.0001)
    fd = FiniteDifference(env, lin_approx, num_passes=2)
    for i in range(max_itr):
        grad = fd.optimize(episode_len=max_episode_len)
        lin_approx.update(grad)
        run_test_episode(env, lin_approx, episode_len=max_episode_len)
