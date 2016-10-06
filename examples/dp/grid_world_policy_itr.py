from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from rlcore.policies.tab import TabularPolicy, TabularValue
from rlcore.algos.dp.pitr import PolicyIteration


def run():
    env = FrozenLakeEnv(map_name="4x4", is_slippery=False)
    policy = TabularPolicy(env.observation_space.n)
    pitr = PolicyIteration(env, policy)

    stable = pitr.iterate()
    print ("Policy stable: ", stable)
    policy.prettyprint()

    done = False
    total_reward = 0.0
    state = env.reset()
    while not done:
        action = policy.get_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

    print ("Total reward: ", total_reward)


if __name__ == "__main__":
    run()
