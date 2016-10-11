from __future__ import print_function

import sys, time
import numpy as np
import tensorflow as tf
import gym
from rlcore.policies.f_approx.linear_tf import LinearApproximator
from rlcore.algos.grad.finite_diff_tf import FiniteDifference


def run_test_episode(env, lin_approx, episode_len=np.inf):
    episode_itr = 0
    total_reward = 0.0
    done = False
    obs = env.reset()
    while not done and episode_itr < episode_len:
        env.render()
        action = lin_approx.predict(obs)
        step = env.step(action)
        done = step.done
        obs = step.observation
        total_reward += step.reward
        episode_itr += 1
    print ("Reward: " + str(total_reward) + ", on iteration " + str(i))


if __name__ == "__main__":
    env = gym.make("CartPole-v0") #normalize(CartpoleEnv())

    max_itr = 2500
    max_episode_len = 100

    with tf.Session() as sess:
        lin_approx = LinearApproximator(env.observation_space.flat_dim, env.action_dim, lr=0.0001)
        fd = FiniteDifference(env, lin_approx, num_passes=2)
        for i in range(max_itr):
            grad = fd.optimize(episode_len=max_episode_len)
            lin_approx.update(grad)
            run_test_episode(env, lin_approx, episode_len=max_episode_len)
