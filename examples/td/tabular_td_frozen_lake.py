from __future__ import print_function

import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from rlcore.policies.tab import TabularPolicy
from rlcore.algos.td.tdlmda import TDLambda
from rlcore.core import rl_utils

if __name__ == "__main__":
    env = FrozenLakeEnv(is_slippery=True)

    tdlmda = TDLambda(env, evaluation_episodes=100)
    tdlmda.train(max_iterations=1000)

    policy = tdlmda.get_policy()
    # policy.prettyprint()

    total_reward = rl_utils.run_test_episode(env, policy)
    print ("Total reward: ", total_reward)
