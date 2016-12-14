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

    TRAIN, TEST = range(2)

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
        if obs is None:
            return self.env.action_space.sample()

        action = self.policy.predict(obs)
        return action


    def on_step_completion(self, state, action, reward, done, info):
        """
        Callback method that subclasses can implement to receive info about
        last step, this is called during training but not testing.
        """
        return


    def run_episode(self, render=True, verbose=False, mode=TRAIN):
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

            obs = self.input_stream_processor.process_observation(obs)
            action = self.act(obs)

            # store observation used to predict and raw action
            ep_states.append(obs)
            ep_actions.append(action)

            next_obs, reward, done, info = self.env.step(action)

            # call the callback method for subclasses
            self.on_step_completion(obs, action, reward, done, info, mode)

            # store reward from environment and any meta data returned
            ep_rewards.append(reward)
            ep_infos.append(info)

            obs = next_obs

            episode_itr += 1

        if verbose:
            print ('Game finished, reward: %f' % (sum(ep_rewards)))

        return ep_states, ep_actions, ep_rewards, ep_infos


    def optimize(self):
        """
        Optimize method that subclasses implement
        """
        raise NotImplementedError


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
            if i % 10 == 0:
                _, _, ep_rewards, _ = self.run_episode(mode=RLAlgorithm.TEST)
                reward = sum(ep_rewards)
                print ("Reward: " + str(reward) + ", on iteration " + str(i))

        if gym_record:
            self.env.monitor.close()


    def test(self,
             episodes=10):
        """
        """
        total_reward = 0.0
        for i in range(episodes):
            _, _, ep_rewards, _ = self.run_episode(mode=RLAlgorithm.TEST)
            reward = sum(ep_rewards)
            print ("Reward: " + str(reward) + ", on iteration " + str(i))
            total_reward += reward

        return float(total_reward) / episodes


    def checkpoint(self):
        """
        Can save the model itself by using tensorflow saver

        But would also like to save any other relevent data, memories,
        other parameters, etc.
        """
        pass


    def restore(self, ckpt_file):
        """
        Restore state from a file
        """
        pass


    def summarize(self):
        """

        """
        pass
