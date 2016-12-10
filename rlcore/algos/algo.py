"""
This module constains a base class for all implemented algorithms
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from rlcore.core import rl_utils
from rlcore.core.input_stream_processor import InputStreamProcessor


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

        self.input_stream_processor = InputStreamProcessor()


    def optimize(self):
        """
        Optimize method that subclasses implement
        """
        raise NotImplementedError


    def apply_prediction_preprocessors(self, obs):
        # print (policy.prediction_preprocessors)
        if self.policy.prediction_preprocessors is not None:
            for preprocessor in self.policy.prediction_preprocessors:
                obs = preprocessor(obs)
        return obs


    def apply_prediction_postprocessors(self, action):
        # print (policy.prediction_postprocessors)
        if self.policy.prediction_postprocessors is not None:
            for postprocessor in self.policy.prediction_postprocessors:
                action = postprocessor(action)
        return action


    def run_episode(self, render=True, verbose=False):
        """
        Runs environment to completion and returns reward under given policy
        Returns the sequence of rewards, states, raw actions (direct from the policy),
            and processed actions (actions sent to the env)

        TODO: Make this method more modular so can abstract a `step` method
        that subclasses can override to make more custom
        """
        ep_rewards = []
        ep_states = []
        ep_raw_actions = []
        ep_processed_actions = []

        done = False
        obs = self.env.reset()

        episode_itr = 0
        while not done and episode_itr < self.episode_len:
            if render:
                self.env.render()

            obs = self.input_stream_processor.process_observation(obs)
            obs = self.apply_prediction_preprocessors(obs)

            # store observation used to predict
            ep_states.append(obs)

            action = self.policy.predict(obs)

            # store raw action
            ep_raw_actions.append(action)

            action = self.apply_prediction_postprocessors(action)

            # store action sent to environment
            ep_processed_actions.append(action)

            obs, reward, done, _ = self.env.step(action)

            # store reward from environment
            ep_rewards.append(reward)

            episode_itr += 1

        if verbose:
            print ('Game finished, reward: %f' % (sum(ep_rewards)))

        return ep_states, ep_raw_actions, ep_processed_actions, ep_rewards


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
                # print (i)
                # TODO: log rewards to tensorboard
                reward = rl_utils.run_test_episode(self.env,
                                                   self.policy,
                                                   episode_len=self.episode_len)
                print ("Reward: " + str(reward) + ", on iteration " + str(i))

        if gym_record:
            self.env.monitor.close()
