from __future__ import print_function

import sys
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
        step = env.step(action)
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
                                  num_dof=bc.get_num_dof(),
                                  timesteps=1000)
    bax_promp.set_training_trajectories(bc.get_trajectories())
    bax_promp.compute_basis_functions()
    bax_promp.compute_promp_prior()

    prior = bax_promp.get_weights_from_dist(bax_promp.Uw)
    prior_traj = bax_promp.get_trajectory_from_weights(prior)

    env = BaxterReacherEnv(control=BaxterReacherEnv.POSITION, limbs=BaxterReacherEnv.BOTH_LIMBS) #normalize(BaxterReacherEnv())
    env.goal = np.array([1.227658244962476, 0.4768209163944621, 0.23905696693000875, 0.8169629678058284, -0.6792894187957359, 0.08773132644854637])
    #env.goal = np.array([1.0915797781830172, 0.4509717292472076, -0.14485613121910212])

    promp_approx = MovementPrimitivesApproximator(bax_promp, lr=0.0001)
    fd = FiniteDifference(env)

    max_itr = 2500
    max_episode_len = bax_promp.timesteps
    for i in range(max_itr):
        print ("Optimization iter = ", i)
        grad = fd.optimize(promp_approx, num_variations=100, episode_len=max_episode_len)
        promp_approx.update(grad)
        reward = run_test_episode(env, promp_approx, episode_len=max_episode_len)

        Snapshotter.snapshot("../logs/promp_approx_"+str(i)+"_"+str(reward)+".pklz", promp_approx)
        print ("Reward: " + str(reward) + ", on iteration " + str(i))
