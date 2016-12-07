from __future__ import print_function

import gym
from rlcore.policies.f_approx.linear_tf import LinearApproximator
from rlcore.algos.td import QLearning
from rlcore.core import rl_utils


if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    policy = LinearApproximator(env.observation_space.shape[0], 1, prediction_postprocessors=[rl_utils.sign])
    ql = QLearning(env, policy, episode_len=100)
    ql.train(max_iterations=10000, max_episode_length=100)

    total_reward = rl_utils.run_test_episode(env, policy, render=True)
    print ("Total reward: ", total_reward)
