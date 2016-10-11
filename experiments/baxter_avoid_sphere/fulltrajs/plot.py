import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import gzip
import cPickle as pickle

def load(fname):
    #restore the object
    f = gzip.open(fname, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def average(list, window=10):
    # print (list.shape)
    list = np.copy(list)
    for i in range(window, list.shape[0]):
        # find average of index i through i - 5
        # replace i with average
        list[i] = np.average(list[i-5+1:i+1])
    return list



def plot_trajectories(trajs, labels):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # goal = np.array([1.0827930984783736, 0.04922204914382992, 0.06132891610869118])
    # goal = trajs[0][-1]
    # ax.scatter(goal[0], goal[1], goal[2], c='r', marker='o')

    # plot the sphere
    x0, y0, z0, r = (0.588178141493, 0.374959360077, 1.33473496229, .05)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_aspect("equal")
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=r*np.cos(u)*np.sin(v) + x0
    y=r*np.sin(u)*np.sin(v) + y0
    z=r*np.cos(v) + z0
    ax.plot_wireframe(x, y, z, color="r")

    #     b: blue.
    # g: green.
    # r: red.
    # c: cyan.
    # m: magenta.
    # y: yellow.
    # k: black.
    # w: white.
    colors = [
        'r', 'b', 'g', 'c', 'm', 'y', 'b'
    ]
    # plot the trajectories
    for j in range(7):
        # j = 4
        for i in range(len(trajs)):
            c = colors[i]
            # for each joint
            t = trajs[i]
            # print (j)
            t[:,j,0] = average(t[:,j,0])
            t[:,j,1] = average(t[:,j,1])
            t[:,j,2] = average(t[:,j,2])

            # ax.plot(t[:,j,0], t[:,j,1], t[:,j,2], label=labels[i], c=c)# + ", joint: " + str(j))

            if (j == 0):
                ax.plot(t[:,j,0], t[:,j,1], t[:,j,2], label=labels[i], c=c)# + ", joint: " + str(j))
            else:
                ax.plot(t[:,j,0], t[:,j,1], t[:,j,2], c=c)#, label=labels[i] + ", joint: " + str(j))
    #     'Initial Trajectory')
    # ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], label='Post Training Trajectory')
    ax.legend()

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def plot_sphere():
    #draw sphere
    x0, y0, z0, r = (0.588178141493, 0.374959360077, 1.33473496229, .05)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=r*np.cos(u)*np.sin(v) + x0
    y=r*np.sin(u)*np.sin(v) + y0
    z=r*np.cos(v) + z0
    ax.plot_wireframe(x, y, z, color="r")
    plt.show()

import sys

if __name__ == "__main__":
    files = [("traj_init.pklz", "Initial Trajectory"),
            # ("traj_25.pklz", "Iteration 25"),
            # ("traj_50.pklz", "Iteration 50"),
            # ("traj_75.pklz", "Iteration 75"),
            # ("traj_100.pklz", "Iteration 100"),
            # ("traj_125.pklz", "Iteration 125"),
            # ("traj_150.pklz", "Iteration 150"),
            # ("traj_175.pklz", "Iteration 175"),
            # ("traj_200.pklz", "Iteration 200"),
            # ("traj_225.pklz", "Iteration 225"),
            # ("traj_250.pklz", "Iteration 250"),
            # ("traj_275.pklz", "Iteration 275"),
            # ("traj_300.pklz", "Iteration 300"),
            # ("traj_325.pklz", "Iteration 325"),
            # ("traj_350.pklz", "Iteration 350"),
            # ("traj_375.pklz", "Iteration 375"),
            # ("traj_400.pklz", "Iteration 400"),
            # ("traj_425.pklz", "Iteration 425"),
            # ("traj_450.pklz", "Iteration 450"),
            # ("traj_475.pklz", "Iteration 475"),
            ("traj_500.pklz", "Final Trajectory"),
    ]
    trajs = []
    labels = []

    for pair in files:
        trajs.append(load(pair[0]))
        print (trajs[-1].shape)
        labels.append(pair[1])

    plot_trajectories(trajs, labels)
    # plot_sphere()
    # t1, t2 = load(fname1), load(fname2)
    # plot_trajectories(t1, t2)
