from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import gym
from rlcore.policies.tab import TabularPolicy
from rlcore.algos.td.tdlmda import TDLambda

def run():
    env = gym.make('Taxi-v1')
    print ("Observation space:", env.observation_space.n)

    tdlmda = TDLambda(env, 0)
    tdlmda.iterate(1, episode_len=np.inf)

    policy = TabularPolicy(env.observation_space.n)

    # print ("Policy stable: ", stable)
    # policy.prettyprint()

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
