from __future__ import print_function

import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from rlcore.policies.tab import TabularPolicy, TabularValue
from rlcore.algos.dp import PolicyIteration
from rlcore.core import rl_utils


if __name__ == "__main__":

    env = FrozenLakeEnv(map_name="4x4", is_slippery=False)
    policy = TabularPolicy(env.observation_space, env.action_space, prediction_postprocessors=[rl_utils.cast_int])
    pitr = PolicyIteration(env, policy)

    stable = pitr.iterate()
    print ("Policy stable: ", stable)
    policy.prettyprint()

    total_reward = rl_utils.run_test_episode(env, policy)
    print ("Total reward: ", total_reward)
