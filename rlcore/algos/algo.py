"""
This module constains a base class for all implemented algorithms
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from rlcore.core import rl_utils

class RLAlgorithm(object):
    """
    Base class for all reinforment learning algorithms. Implements a train
    function that calls the subclass optimize method.
    """

    def __init__(self, env, policy, session, episode_len, discount, standardize):
        """
        RLAlgorithm constructor
        """
        self.env = env
        self.policy = policy
        self.sess = session
        self.episode_len = episode_len
        self.discount = discount
        self.standardize = standardize


    def optimize(self):
        """
        Optimize method that subclasses implement
        """
        raise NotImplementedError


    def train(self,
              max_iterations=10000,
              gym_record=False,
              gym_record_dir='/tmp/rlcore/gym/',
              tensorboard_log=False,
              tensorboard_log_dir='/tmp/rlcore/tensorboard/'):
        """
        This method is an abstraction to the training process.

        It allows for recording the run to upload to OpenAI gym
        """
        if gym_record:
            self.env.monitor.start(gym_record_dir)

        # summary = tf.Summary()
        # summary_value = summary.value.add()
        # summary_value.simple_value = value.item()
        # summary_value.tag = name
        # self.writer.add_summary(summary, epoch)

        for i in range(max_iterations):
            self.optimize()

            if i % 10 == 0:
                # TODO: log rewards to tensorboard
                reward = rl_utils.run_test_episode(self.env,
                                                   self.policy,
                                                   episode_len=self.episode_len)
                print ("Reward: " + str(reward) + ", on iteration " + str(i))

        if gym_record:
            self.env.monitor.close()
