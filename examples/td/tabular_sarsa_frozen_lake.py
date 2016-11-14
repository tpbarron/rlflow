from __future__ import print_function

import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from rlcore.policies.tab import TabularActionValue
from rlcore.algos.td import SARSA
from rlcore.core import rl_utils

if __name__ == "__main__":
    env = FrozenLakeEnv(is_slippery=False)

    policy = TabularActionValue(env.observation_space, env.action_space, epsilon=0.5)
    sarsa = SARSA(env, policy, episode_len=25)
    sarsa.train(max_iterations=1000, max_episode_length=25)
    #policy.prettyprint()

    total_reward = rl_utils.run_test_episode(env, policy, render=True)
    print ("Reward = ", total_reward)
