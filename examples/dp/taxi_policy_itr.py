from __future__ import print_function

import numpy as np
import gym
from markov.policies.tab import TabularPolicy
from markov.algos.dp import PolicyIteration
from markov.core import rl_utils

if __name__ == "__main__":
    """
    This is currently broken since my policy iter assumed grid world action
    """
    env = gym.make('Taxi-v1')

    print (env.action_space)
    policy = TabularPolicy(env.observation_space, env.action_space, prediction_postprocessors=[rl_utils.cast_int])
    pitr = PolicyIteration(env, policy)
    stable = pitr.iterate()

    print ("Policy stable: ", stable)
    # policy.prettyprint()

    total_reward = rl_utils.run_test_episode(env, policy)
    print ("Total reward: ", total_reward)
