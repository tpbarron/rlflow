from __future__ import print_function

import sys
import numpy as np

import gym
from rlcore.core import rl_utils
from rlcore.policies.f_approx.linear import LinearApproximator
from rlcore.algos.grad.finite_diff import FiniteDifference


def run_test_episode(env, lin_approx, episode_len=np.inf):
    episode_itr = 0
    total_reward = 0.0
    done = False
    obs = env.reset()
    while not done and episode_itr < episode_len:
        env.render()
        action = lin_approx.predict(obs)
        state, reward, done, info = env.step(action)
        total_reward += reward
        episode_itr += 1
    print ("Reward: " + str(total_reward) + ", on iteration " + str(i))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    lin_approx = LinearApproximator(env.observation_space.shape[0],
                                    1,
                                    lr=0.001,
                                    prediction_postprocessor=rl_utils.sign)

    fd = FiniteDifference(env, num_passes=2)

    max_itr = 2500
    max_episode_len = 500
    for i in range(max_itr):
        grad = fd.optimize(lin_approx, episode_len=max_episode_len)
        lin_approx.update(grad)
        run_test_episode(env, lin_approx, episode_len=max_episode_len)
