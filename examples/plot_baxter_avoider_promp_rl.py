from __future__ import print_function

import argparse
from rlcore.logger.snapshotter import Snapshotter
from rlcore.viz.plotter import Plotter
from os import listdir
from os.path import isfile, join


def parse_rewards(path):
    itrs = []
    rewards = []
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('pklz')]
    for f in onlyfiles:
        post = f[13:-5]
        itr, reward = post.split("_")
        itrs.append(int(itr)+1)
        rewards.append(float(reward))

    ordered_rewards = [r for (i, r) in sorted(zip(itrs, rewards))]
    return ordered_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot baxter avoider promp rewards')
    parser.add_argument('-d', '--dir', dest='dir', required=True)
    parser.add_argument('--traj1', dest='t1', default=None, required=False)
    parser.add_argument('--traj2', dest='t2', default=None, required=False)
    args = parser.parse_args()

    ordered_rewards = parse_rewards(args.dir)
    Plotter.plot_values(ordered_rewards)

    if (args.traj1 != None and args.traj2 != None):
        traj1 = Snapshotter.load(args.traj1)
        traj2 = Snapshotter.load(args.traj2)
        Plotter.plot_trajectories(traj1, traj2)
