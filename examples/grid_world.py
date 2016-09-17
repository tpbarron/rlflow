from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from rlcore.envs.grid_world_env import GridWorldEnv


if __name__ == "__main__":
    env = GridWorldEnv()
    
    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        step = env.step(action)
        reward = step.reward
        done = step.done

        print (env.desc)
        print (env.state)

        total_reward += step.reward
