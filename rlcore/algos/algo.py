import gym
import numpy as np
from rlcore.core import rl_utils

class RLAlgorithm(object):

    def __init__(self):
        pass


    def train(self,
              max_iterations=10000,
              max_episode_length=np.inf,
              gym_record=False,
              gym_record_dir='/tmp/rlcore/experiment/'):

        if (gym_record):
            self.env.monitor.start(gym_record_dir)

        for i in range(max_iterations):
            self.optimize()

            # TODO: improve data output
            if (i % 100 == 0):
                reward = rl_utils.run_test_episode(self.env, self.policy, episode_len=max_episode_length)
                print ("Reward: " + str(reward) + ", on iteration " + str(i))


        if (gym_record):
            self.env.monitor.close()
