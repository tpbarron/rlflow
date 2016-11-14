from __future__ import print_function

import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from rlcore.policies.tab import TabularActionValue
from rlcore.algos.td import QLearning
from rlcore.core import rl_utils

if __name__ == "__main__":
    env = FrozenLakeEnv(is_slippery=False)

    policy = TabularActionValue(env.observation_space, env.action_space, epsilon=0.5)
    ql = QLearning(env, policy, episode_len=25)
    ql.train(max_iterations=10000, max_episode_length=25)
    policy.prettyprint()

    total_reward = rl_utils.run_test_episode(env, policy, render=True)
    print ("Total reward: ", total_reward)
