from __future__ import print_function

import sys
import numpy as np

from rlcore.envs.normalized_env import normalize
from rlcore.envs.box2d.cartpole_env import CartpoleEnv
from rlcore.policies.f_approx.linear import LinearApproximator
from rlcore.policies.f_approx.linear_gaussian import LinearGaussianApproximator
from rlcore.algos.grad.finite_diff import FiniteDifference


def run_test_episode(env, lin_approx, episode_len=np.inf):
    episode_itr = 0
    total_reward = 0.0
    done = False
    obs = env.reset()
    while not done and episode_itr < episode_len:
        env.render()
        action = lin_approx.predict(obs)
        step = env.step(action)
        done = step.done
        obs = step.observation
        total_reward += step.reward
        episode_itr += 1
    print ("Reward: " + str(total_reward) + ", on iteration " + str(i))


if __name__ == "__main__":
    env = normalize(CartpoleEnv())

    lin_approx = LinearGaussianApproximator(env.observation_space.flat_dim,
                                            num_gaussians=20,
                                            discretization=100)
    lin_approx = LinearApproximator(env.observation_space.flat_dim)
    fd = FiniteDifference(env, num_passes=10)

    # TODO: something goes wrong when I set the episode_len
    # rewards are no longer correct
    max_itr = 1000
    for i in range(max_itr):
        grad = fd.optimize(lin_approx, episode_len=100) #lin_approx.discretization)
        lin_approx.update(grad)
        run_test_episode(env, lin_approx, episode_len=100)# lin_approx.discretization)
