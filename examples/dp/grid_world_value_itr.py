"""
This example works under python2.7 but not 3.5. I haven't had the time to find
the cause yet.
"""

from __future__ import print_function

import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from rlflow.policies.tab import TabularPolicy, TabularValue
from rlflow.algos.dp import ValueIteration
from rlflow.core import rl_utils

if __name__ == "__main__":
    env = FrozenLakeEnv(map_name="8x8", is_slippery=False)
    policy = TabularValue(env.observation_space, env.action_space)
    vitr = ValueIteration(env, policy)

    # get back a deterministic policy from the value iteration
    deterministic_policy = vitr.iterate()
    deterministic_policy.prettyprint()
    deterministic_policy.prediction_postprocessors = [rl_utils.cast_int]

    total_reward = rl_utils.run_test_episode(env, deterministic_policy)
    print ("Total reward: ", total_reward)
