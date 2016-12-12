"""
This module constains a base class for all implemented algorithms
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from markov.core import rl_utils
from markov.core.input.input_stream_processor import InputStreamProcessor


class RLAlgorithm(object):
    """
    Base class for all reinforment learning algorithms. Implements a train
    function that calls the subclass optimize method.
    """

    def __init__(self,
                 env,
                 policy,
                 session,
                 episode_len,
                 discount,
                 standardize,
                 input_processor):
        """
        RLAlgorithm constructor
        """
        self.env = env
        self.policy = policy
        self.sess = session
        self.episode_len = episode_len
        self.discount = discount
        self.standardize = standardize
        self.input_stream_processor = input_processor
        if self.input_stream_processor is None:
            # just create a default
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


    def reset(self):
        """
        Reset the environment to the start state. Can override this method
        to do custom reset processing.
        """
        self.input_stream_processor.reset()
        return self.env.reset()


    def act(self, obs):
        """
        Take in the current observation and return the action to take.
        Override for custom action selection, eg. epsilon greedy, gradient
        updates during trajectories (DQN), etc.
        """
        obs = self.input_stream_processor.process_observation(obs)
        if obs is None:
            return self.env.action_space.sample()

        # obs = self.apply_prediction_preprocessors(obs)
        action = self.policy.predict(obs)
        # action = self.apply_prediction_postprocessors(action)
        return action


    def step_callback(self, state, action, reward, done, info):
        """
        Callback method that subclasses can implement to receive info about
        last step
        """
        return


    def run_episode(self, render=True, verbose=False):
        """
        Runs environment to completion and returns reward under given policy
        Returns the sequence of rewards, states, raw actions (direct from the policy),
            and processed actions (actions sent to the env)
        """
        ep_rewards = []
        ep_states = []
        ep_actions = []
        ep_infos = []

        done = False
        obs = self.reset()

        episode_itr = 0
        while not done and episode_itr < self.episode_len:
            if render:
                self.env.render()

            action = self.act(obs)

            # store observation used to predict and raw action
            ep_states.append(obs)
            ep_actions.append(action)

            next_obs, reward, done, info = self.env.step(action)

            # call the callback method for subclasses
            self.step_callback(obs, action, reward, done, info)

            # store reward from environment and any meta data returned
            ep_rewards.append(reward)
            ep_infos.append(info)

            obs = next_obs

            episode_itr += 1

        if verbose:
            print ('Game finished, reward: %f' % (sum(ep_rewards)))

        return ep_states, ep_actions, ep_rewards, ep_infos


    def train(self,
              max_iterations=10000,
              gym_record=False,
              gym_record_dir='/tmp/markov/gym/',
              tensorboard_log=False,
              tensorboard_log_dir='/tmp/markov/tensorboard/'):
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

            # if i % 10 == 0:
            #     # TODO: log rewards to tensorboard
            #     reward = rl_utils.run_test_episode(self.env,
            #                                        self.policy,
            #                                        episode_len=self.episode_len)
            #     print ("Reward: " + str(reward) + ", on iteration " + str(i))

        if gym_record:
            self.env.monitor.close()
