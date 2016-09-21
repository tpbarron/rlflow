from __future__ import print_function

import sys, os, datetime
import numpy as np
import argparse

from iprims.promp import ProbabilisticMovementPrimitive

from rlcore.logger.snapshotter import Snapshotter
from rlcore.envs.normalized_env import normalize
from rlcore.envs.baxter.baxter_utils import BaxterUtils
from rlcore.envs.baxter.baxter_reacher_env import BaxterReacherEnv
from rlcore.policies.f_approx.movement_primitive import MovementPrimitivesApproximator
from rlcore.algos.grad.finite_diff import FiniteDifference


def run_test_episode(env, approx, episode_len=np.inf):
    episode_itr = 0
    total_reward = 0.0
    done = False
    obs = env.reset()
    while not done and episode_itr < episode_len:
        env.render()
        action = approx.predict(obs)
        step = env.step(action, t=episode_itr)
        done = step.done
        obs = step.observation
        total_reward += step.reward
        episode_itr += 1
    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baxter Pro MP')
    parser.add_argument('-d', '--dir', dest='dir', required=True)
    parser.add_argument('-m', '--mode', dest='mode', default='p',
        required=False, help='p for position, v for velocity')
    parser.add_argument('-t', '--time', dest='time', default=3,
        required=False, help='the number of seconds to run')
    args = parser.parse_args()

    mode = BaxterUtils.POSITION if args.mode == 'p' else BaxterUtils.VELOCITY
    secs = int(args.time)

    bc = BaxterUtils()
    bc.load_trajectories(args.dir)
    bax_promp = ProbabilisticMovementPrimitive(num_bases=20,
                                  num_dof=(bc.get_num_dof()-1)/2,
                                  timesteps=100)
    bax_promp.set_training_trajectories(bc.get_trajectories_without_time(limb=BaxterUtils.LEFT_LIMB))
    bax_promp.compute_basis_functions()
    bax_promp.compute_promp_prior()

    env = BaxterReacherEnv(bax_promp.timesteps, control=BaxterReacherEnv.POSITION, limbs=BaxterReacherEnv.LEFT_LIMB)
    #goal_both = np.array([1.1753262657425387, 0.11223764298019563-.5, 0.026154249917259995, 0.8521478534645788, -0.29585696905772585, 0.08423954120828164])
    goal_left = np.array([1.1753262657425387, 0.11223764298019563-.5, 0.026154249917259995])
    env.goal = goal_left

    logdir = os.path.join("../logs/", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logdir)

    # load = False
    # if (load):
    #     promp_approx = Snapshotter.load("../logs/promp_approx_0_1.0.pklz")
    # else:
    promp_approx = MovementPrimitivesApproximator(bax_promp, lr=0.001)
    fd = FiniteDifference(env)

    max_itr = 2500
    max_episode_len = bax_promp.timesteps
    for i in range(max_itr):
        print ("Optimization iter = ", i)
        grad = fd.optimize(promp_approx, num_variations=50, episode_len=max_episode_len)
        promp_approx.update(grad)
        reward = run_test_episode(env, promp_approx, episode_len=max_episode_len)

        Snapshotter.snapshot(os.path.join(logdir, "promp_approx_"+str(i)+"_"+str(reward)+".pklz"), promp_approx)
        print ("Reward: " + str(reward) + ", on iteration " + str(i))
