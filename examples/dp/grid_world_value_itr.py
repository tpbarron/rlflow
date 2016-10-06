from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from rlcore.policies.tab import TabularPolicy, TabularValue
from rlcore.algos.dp.vitr import ValueIteration

def run():
    env = FrozenLakeEnv(map_name="8x8", is_slippery=False)
    policy = TabularValue(env.observation_space.n)
    vitr = ValueIteration(env, policy)
    det_pol = vitr.iterate()
    det_pol.prettyprint()

    done = False
    total_reward = 0.0
    state = env.reset()
    while not done:
        action = det_pol.get_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        # print ("State: ", state, ", Action: ", action, ", Reward = ", total_reward)

    print ("Total reward: ", total_reward)


if __name__ == "__main__":
    run()
