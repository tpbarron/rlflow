from __future__ import print_function

import gym
import tensorflow as tf
import tflearn

from rlcore.core import rl_utils
from rlcore.core import tf_utils
from rlcore.policies.f_approx import Network
from rlcore.algos.grad.dqn import DQN
from rlcore.memories.experience_replay import ExperienceReplay
from rlcore.exploration.egreedy import EpsilonGreedy

if __name__ == "__main__":
    env = gym.make("Pong-v0")

    with tf.Session() as sess:
        input_tensor = tf_utils.get_input_tensor_shape(env)
        net = tflearn.conv_2d(input_tensor, 16, 8, 4, activation='relu')
        net = tflearn.conv_2d(net, 32, 4, 2, activation='relu')
        net = tflearn.flatten(net)
        net = tflearn.fully_connected(net, 1024, activation='relu')
        net = tflearn.fully_connected(net, env.action_space.n, activation='linear')

        egreedy = EpsilonGreedy(0.9, 0.1, 1000)
        # TODO: I think the exploration should go in the algorithm,
        # then can call object with the current step
        # and decide whether to call predict or not
        network = Network(input_tensor,
                          net,
                          sess)

        memory = ExperienceReplay(size=10000)
        dqn = DQN(env,
                  network,
                  sess,
                  memory,
                  episode_len=100,
                  discount=0.9,
                  optimizer='adam')

        dqn.train(max_iterations=1000, gym_record=False)

        average_reward = rl_utils.average_test_episodes(env,
                                                        network,
                                                        10,
                                                        episode_len=dqn.episode_len)
        print ("Average: ", average_reward)
