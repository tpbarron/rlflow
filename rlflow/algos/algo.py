"""
This module constains a base class for all implemented algorithms
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle
import os

from rlflow.core import tf_utils
from rlflow.core import io_utils
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


        try:
            self.saver = tf.train.Saver()
        except:
            self.saver = None
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


    def act(self, obs, mode):
        """
        Take in the current observation and return the action to take.
        Override for custom action selection, eg. epsilon greedy, gradient
        updates during trajectories (DQN), etc.
        """
        if obs is None:
            return self.env.action_space.sample()

        action = self.policy.predict(obs)
        return action


    def run_episode(self, render=False, verbose=False, mode=TRAIN):
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
            action = self.act(obs, mode)

            # store observation used to predict and raw action
            ep_states.append(obs)
            ep_actions.append(action)

            self.on_step_start()
            next_obs, reward, done, info = self.env.step(action)

            # call the callback method for subclasses
            self.on_step_finish(obs, action, reward, done, info, mode)

            # store reward from environment and any meta data returned
            ep_rewards.append(reward)
            ep_infos.append(info)

            obs = next_obs
            episode_itr += 1

        if verbose:
            print ('Game finished, reward: %f' % (sum(ep_rewards)))

        # print ("rew list: ", ep_rewards, sum(ep_rewards))
        return ep_states, ep_actions, ep_rewards, ep_infos


    def optimize(self):
        """
        Optimize method that subclasses implement

        TODO: remove this and simply use run_episode
        """
        ep_states, ep_actions, ep_rewards, ep_infos = self.run_episode()
        return ep_states, ep_actions, ep_rewards, ep_infos


    def train(self,
              max_episodes=10000,
              test_frequency=100,
              save_frequency=100,
              render_train=False,
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

        if gym_record:
            self.env.monitor.start(gym_record_dir)

        self.on_train_start()
        for i in range(max_episodes):
            self.on_episode_start()
            _, _, ep_rewards, _ = self.optimize()
            self.on_episode_finish()
            ep_reward = sum(ep_rewards)
            # print ("Episode reward: ", ep_reward)
            # self.summarize(i, ep_reward)

            if i % test_frequency == 0:
                _, _, test_ep_rewards, _ = self.run_episode(render=True, mode=RLAlgorithm.TEST)
                test_reward = sum(test_ep_rewards)
                print ("Test episode (itr " + str(i) + "), reward: ", test_reward)

            if i % save_frequency == 0:
                self.checkpoint(step=i)

        self.on_train_finish()

        if gym_record:
            self.env.monitor.close()


    # Training callbacks
    def on_train_start(self):
        """
        Callback for subclasses to implement if they wish to do any processing right
        before training commences.
        """
        return


    def on_train_finish(self):
        """
        Callback for subclasses to implement if they wish to do any processing right
        after training completes.
        """
        return


    def on_episode_start(self):
        return


    def on_episode_finish(self):
        pass


    def on_step_start(self):
        """
        Callback method that subclasses can implement to receive info about
        last step, this is called during training but not testing.
        """
        return


    def on_step_finish(self, state, action, reward, done, info, mode):
        """
        """
        return


    def test(self,
             episodes=10,
             record_experience=False,
             record_experience_path='/tmp/rlflow/data/'):
        """
        Run the given number of episodes and return a list of the episode rewards
        """
        io_utils.create_dir_if_not_exists(record_experience_path)
        rewards = []
        for i in range(episodes):
            ep_states, ep_actions, ep_rewards, _ = self.run_episode(render=False, mode=RLAlgorithm.TEST)
            if record_experience:
                ep_states = np.array(ep_states)
                ep_actions = np.array(ep_actions)
                ep_rewards = np.array(ep_rewards)
                io_utils.save_h5py(os.path.join(record_experience_path, "ep_"+str(i)+".h5"), ep_states, ep_actions, ep_rewards)
                # io_utils.save_pickle(os.path.join(record_experience_path, "ep_states_"+str(i)+".pkl"), ep_states)
                # io_utils.save_pickle(os.path.join(record_experience_path, "ep_actions_"+str(i)+".pkl"), ep_actions)
                # io_utils.save_pickle(os.path.join(record_experience_path, "ep_rewards_"+str(i)+".pkl"), ep_rewards)

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
