
import matplotlib.pyplot as plt

class Plotter(object):


    @staticmethod
    def plot_values(values):
        plt.plot(values)
        plt.title('Rewards per iteration')
        plt.ylabel('Reward')
        plt.xlabel('Iteration')
        plt.show()
