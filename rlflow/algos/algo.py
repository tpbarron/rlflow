"""
This module constains a base class for all implemented algorithms
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn

from rlflow.core import tf_utils
from rlflow.core.input.input_stream_processor import InputStreamProcessor

# # Fix for TF 0.12
# try:
#     writer_summary = tf.summary.FileWriter
#     merge_all_summaries = tf.summary.merge_all
#     histogram_summary = tf.summary.histogram
#     scalar_summary = tf.summary.scalar
# except Exception:
#     writer_summary = tf.train.SummaryWriter
#     merge_all_summaries = tf.merge_all_summaries
#     histogram_summary = tf.histogram_summary
#     scalar_summary = tf.scalar_summary


class RLAlgorithm(object):
    """
    Base class for all reinforment learning algorithms. Implements a train
    function that calls the subclass optimize method.
    """

    TRAIN, TEST = range(2)

    # SUMMARY_COLLECTION_NAME = "summary_collection"

    def __init__(self,
                 env,
                 policy,
                 episode_len,
                 discount,
                 standardize,
                 input_processor,
                 optimizer,
                 clip_gradients):
        """
        RLAlgorithm constructor
        """
        self.env = env
        self.policy = policy
        self.sess = tf_utils.get_tf_session()
        self.episode_len = episode_len
        self.discount = discount
        self.standardize = standardize
        self.input_stream_processor = input_processor
        if self.input_stream_processor is None:
            # just create a default
            self.input_stream_processor = InputStreamProcessor()


        self.saver = tf.train.Saver()
        # self.summary_op = None
        # self.build_summary_ops()
        # self.summary_writer = tf.train.SummaryWriter('/tmp/rlflow/log',
        #                                              self.sess.graph)

        self.opt = optimizer
        self.clip_gradients = clip_gradients


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


    def on_step_completion(self, state, action, reward, done, info, mode):
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

        In order for reward summarization to work properly this method should return
        the total episode reward.
        """
        _, _, ep_rewards, _ = self.run_episode()
        # print ("Rewards: ", ep_rewards)
        return sum(ep_rewards)


    def on_train_start(self):
        """
        Callback for subclasses to implement if they wish to do any processing right
        before training commences.
        """
        return


    def train(self,
              max_episodes=10000,
              save_frequency=100,
              gym_record=False,
              gym_record_dir='/tmp/rlflow/gym/',
              tensorboard_log=False,
              tensorboard_log_dir='/tmp/rlflow/tensorboard/'):
        """
        This method is an abstraction to the training process.

        It allows for recording the run to upload to OpenAI gym
        """

        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()

        self.on_train_start()

        if gym_record:
            self.env.monitor.start(gym_record_dir)

        for i in range(max_episodes):
            ep_reward = self.optimize()
            print ("Episode reward: ", ep_reward)
            # self.summarize(i, ep_reward)

            if i % save_frequency == 0:
                self.checkpoint(step=i)

        if gym_record:
            self.env.monitor.close()


    def test(self, episodes=10):
        """
        Run the given number of episodes and return a list of the episode rewards
        """
        rewards = []
        for i in range(episodes):
            _, _, ep_rewards, _ = self.run_episode(mode=RLAlgorithm.TEST)
            reward = sum(ep_rewards)
            rewards.append(reward)

        return rewards


    def checkpoint(self, step=None, ckpt_file='/tmp/rlflow/model.ckpt'):
        """
        Can save the model itself by using tensorflow saver

        But would also like to save any other relevent data, memories,
        other parameters, etc.
        """
        save_path = self.saver.save(self.sess, ckpt_file, global_step=step)
        print("Session saved in file: %s" % save_path)


    def restore(self, ckpt_file='/tmp/rlflow/model.ckpt'):
        """
        Restore state from a file
        """
        self.saver.restore(self.sess, ckpt_file)
        # if '-' in ckpt_file[ckpt_file.rfind('.ckpt'):]:
        #     last_step = int(ckpt_file[ckpt_file.find('-')+1:])
        #     self.step = last_step
        print("Session restored from file: %s" % ckpt_file)


    # def build_summary_ops(self, verbose=3):
    #     """
    #     Build summary ops for activations, gradients, reward, q values,
    #     values estimates, etc
    #     Create summaries with `verbose` level
    #     """
    #     if verbose >= 3:
    #         # Summarize activations
    #         activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
    #         tflearn.summarize_activations(activations, RLAlgorithm.SUMMARY_COLLECTION_NAME)
    #     if verbose >= 2:
    #         # Summarize variable weights
    #         tflearn.summarize_variables(tf.trainable_variables(), RLAlgorithm.SUMMARY_COLLECTION_NAME)
    #     if verbose >= 1:
    #         # summarize reward
    #         episode_reward = tf.Variable(0., trainable=False)
    #         self.episode_reward_summary = scalar_summary("Reward", episode_reward, collections=RLAlgorithm.SUMMARY_COLLECTION_NAME)
    #         self.episode_reward_placeholder = tf.placeholder("float")
    #         self.episode_reward_op = episode_reward.assign(self.episode_reward_placeholder)
    #         tf.add_to_collection(RLAlgorithm.SUMMARY_COLLECTION_NAME, self.episode_reward_summary)
    #
    #         # Summarize gradients
    #         # tflearn.summarize_gradients(self.grads_and_vars, summ_collection)
    #
    #     if len(tf.get_collection(RLAlgorithm.SUMMARY_COLLECTION_NAME)) != 0:
    #         self.summary_op = merge_all_summaries(key=RLAlgorithm.SUMMARY_COLLECTION_NAME)


    def summarize(self, step, reward):
        """
        Run all of the previously defined summary ops
        """
        pass
        # self.sess.run(self.episode_reward_op, feed_dict={self.episode_reward_placeholder: reward})
        # summary = self.sess.run(self.summary_op)
        # self.summary_writer.add_summary(summary, step)
