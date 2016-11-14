import gym
import tensorflow as tf
import numpy as np
from rlcore.core import rl_utils
from rlcore.core.tf.tensorflow_utils import wrap_session

class RLAlgorithm(object):

    def __init__(self):
        pass


    def optimize(self):
        raise NotImplementedError


    def train(self,
              max_iterations=10000,
              max_episode_length=np.inf,
              gym_record=False,
              gym_record_dir='/tmp/rlcore/gym/',
              tensorboard_log=False,
              tensorboard_log_dir='/tmp/rlcore/tensorboard/'):

        if (gym_record):
            self.env.monitor.start(gym_record_dir)

        # if (tensorboard_log):
        #     summary = tf.Summary()
        #     summary_value = summary.value.add()
        #     summary_value.simple_value = value.item()
        #     summary_value.tag = name
        #     self.writer.add_summary(summary, epoch)

        for i in range(max_iterations):
            self.optimize()

            # TODO: improve data output
            if (i % 10 == 0):
                # TODO: log rewards to tensorboard
                reward = rl_utils.run_test_episode(self.env, self.policy, episode_len=max_episode_length)
                print ("Reward: " + str(reward) + ", on iteration " + str(i))

        if (gym_record):
            self.env.monitor.close()
