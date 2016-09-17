from __future__ import print_function

import sys
import numpy as np
from rlcore.envs.normalized_env import normalize
from rlcore.envs.box2d.cartpole_env import CartpoleEnv
from rlcore.policies.f_approx.linear import LinearFunctionApproximator
from rlcore.algos.grad.finite_diff import FiniteDifference

if __name__ == "__main__":
    env = normalize(CartpoleEnv())

    fd = FiniteDifference(num_passes=2, lr=0.01)
    lin_approx = LinearFunctionApproximator(env.action_space.n)

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
