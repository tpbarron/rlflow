from __future__ import print_function

import sys
import numpy as np

from rlcore.envs.normalized_env import normalize
from rlcore.envs.baxter.baxter_reacher_env import BaxterReacherEnv
from rlcore.policies.f_approx.linear import LinearApproximator
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
    env = BaxterReacherEnv(control=BaxterReacherEnv.TORQUE, limbs=BaxterReacherEnv.LEFT_LIMB) #normalize(BaxterReacherEnv())
    print ("Goal = ", env.goal)
    lin_approx = LinearApproximator(env.observation_space.flat_dim, env.action_dim, lr=0.0001)
    fd = FiniteDifference(env, num_passes=2)

    max_itr = 2500
    max_episode_len = 100
    for i in range(max_itr):
        print ("Optimization iter = ", i)
        grad = fd.optimize(lin_approx, episode_len=max_episode_len)
        #print ("Grad = ", grad)
        lin_approx.update(grad)
        run_test_episode(env, lin_approx, episode_len=max_episode_len)
