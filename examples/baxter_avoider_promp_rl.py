from __future__ import print_function

import sys, os, datetime
import numpy as np
import argparse

from iprims.promp import ProbabilisticMovementPrimitive

from rlcore.logger.snapshotter import Snapshotter
from rlcore.envs.normalized_env import normalize
from rlcore.envs.baxter.baxter_utils import BaxterUtils
from rlcore.envs.baxter.baxter_avoider_env import BaxterAvoiderEnv
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
    parser.add_argument('-p', '--policy', dest='policy', default=None,
        required=False, help='File to load policy from')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    mode = BaxterAvoiderEnv.POSITION if args.mode == 'p' else BaxterAvoiderEnv.VELOCITY

    bc = BaxterUtils()
    bc.load_trajectories(args.dir)
    bax_promp = ProbabilisticMovementPrimitive(num_bases=2,
                                  num_dof=BaxterUtils.DOF_NO_TIME_NO_GRIPPER_LEFT,
                                  timesteps=250)
    bax_promp.set_training_trajectories(bc.get_trajectories_without_time_and_gripper(limb=BaxterUtils.LEFT_LIMB))
    bax_promp.compute_basis_functions()
    bax_promp.compute_promp_prior()
             #(x, y, z, r)
    sphere = (0.588178141493, 0.374959360077, 1.33473496229, .05)
    env = BaxterAvoiderEnv(bax_promp.timesteps, sphere, control=mode, limbs=BaxterAvoiderEnv.LEFT_LIMB)
    goal_left_2b = np.array([1.0827930984783736, 0.04922204914382992, 0.06132891610869118])
    goal_left_10b = np.array([1.1724322298736176, 0.17367959828418025, 0.00945407307459449])
    goal_left_20b = np.array([1.1753262657425387, 0.11223764298019563, 0.026154249917259995])
    env.goal = goal_left_2b

    logdir = os.path.join("../logs/", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logdir)

    # TODO: check this
    if (args.policy != None):
        #promp_approx = Snapshotter.load('/home/tpbarron/Documents/dev/ml/rl/rlcore/logs/2016-09-21_17-05-11/promp_approx_0_0.533274865676.pklz')
        promp_approx = Snapshotter.load(args.policy)
    else:
        promp_approx = MovementPrimitivesApproximator(bax_promp, lr=0.01)
    fd = FiniteDifference(env)

    max_itr = 2500
    max_episode_len = bax_promp.timesteps

    if (args.test):
        for i in range(5):
            reward = run_test_episode(env, promp_approx, episode_len=max_episode_len)
            print ("Reward: " + str(reward) + ", on test " + str(i))
    else:
        reward = run_test_episode(env, promp_approx, episode_len=max_episode_len)
        Snapshotter.snapshot(os.path.join(logdir, "promp_approx_"+str(reward)+".pklz"), promp_approx)
        print ("Reward: " + str(reward) + ", at start")

        for i in range(max_itr):
            print ("Optimization iter = ", i)
            grad = fd.optimize(promp_approx, episode_len=max_episode_len)
            promp_approx.update(grad)
            reward = run_test_episode(env, promp_approx, episode_len=max_episode_len)

            Snapshotter.snapshot(os.path.join(logdir, "promp_approx_"+str(i)+"_"+str(reward)+".pklz"), promp_approx)
            print ("Reward: " + str(reward) + ", on iteration " + str(i))
