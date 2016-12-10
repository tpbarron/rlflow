from __future__ import print_function

import sys, os, datetime
import numpy as np
import argparse

from iprims.promp import ProbabilisticMovementPrimitive

from markov.logger.snapshotter import Snapshotter
from markov.envs.normalized_env import normalize
from markov.envs.baxter.baxter_utils import BaxterUtils
from markov.envs.baxter.baxter_reacher_env import BaxterReacherEnv
from markov.policies.f_approx.movement_primitive import MovementPrimitivesApproximator
from markov.algos.prob.promp_itr import ProMPIteration


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
    args = parser.parse_args()

    mode = BaxterReacherEnv.POSITION if args.mode == 'p' else BaxterReacherEnv.VELOCITY

    bc = BaxterUtils()
    bc.load_trajectories(args.dir)
    bax_promp = ProbabilisticMovementPrimitive(num_bases=2,
                                  num_dof=BaxterUtils.DOF_NO_TIME_NO_GRIPPER_LEFT,
                                  timesteps=250)
    bax_promp.set_training_trajectories(bc.get_trajectories_without_time_and_gripper(limb=BaxterUtils.LEFT_LIMB))
    bax_promp.compute_basis_functions()
    bax_promp.compute_promp_prior()

    env = BaxterReacherEnv(bax_promp.timesteps, control=mode, limbs=BaxterReacherEnv.LEFT_LIMB)
    #goal_both = np.array([1.1753262657425387, 0.11223764298019563-.5, 0.026154249917259995, 0.8521478534645788, -0.29585696905772585, 0.08423954120828164])
    goal_left_2b = np.array([1.0827930984783736, 0.04922204914382992+.5, 0.06132891610869118])
    goal_left_10b = np.array([1.1724322298736176, 0.17367959828418025-.5, 0.00945407307459449])
    goal_left_20b = np.array([1.1753262657425387, 0.11223764298019563-.5, 0.026154249917259995])
    env.goal = goal_left_2b

    logdir = os.path.join("../logs/", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logdir)

    # TODO: check this
    if (args.policy != None):
        promp_approx = Snapshotter.load(args.policy)
    else:
        promp_approx = MovementPrimitivesApproximator(bax_promp)
    promp_itr = ProMPIteration(env)

    max_itr = 250
    max_episode_len = bax_promp.timesteps
    reward = run_test_episode(env, promp_approx, episode_len=max_episode_len)
    Snapshotter.snapshot(os.path.join(logdir, "promp_approx_"+str(reward)+".pklz"), promp_approx)
    print ("Reward: " + str(reward) + ", at start")

    for i in range(max_itr):
        print ("Optimization iter = ", i)
        Uw, Ew = promp_itr.optimize(promp_approx, episode_len=max_episode_len)
        promp_approx.update(Uw, Ew)
        reward = run_test_episode(env, promp_approx, episode_len=max_episode_len)

        Snapshotter.snapshot(os.path.join(logdir, "promp_approx_"+str(i)+"_"+str(reward)+".pklz"), promp_approx)
        print ("Reward: " + str(reward) + ", on iteration " + str(i))
