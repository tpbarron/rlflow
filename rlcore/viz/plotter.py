
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

class Plotter(object):


    @staticmethod
    def plot_values(values):
        plt.plot(values)
        plt.title('Rewards per iteration')
        plt.ylabel('Reward')
        plt.xlabel('Iteration')
        plt.show()
        

    @staticmethod
    def plot_trajectories(traj1, traj2):
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], label='Initial Trajectory')
        ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], label='Post Training Trajectory')
        ax.legend()
        plt.show()
