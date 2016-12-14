from __future__ import print_function

import gym
import tensorflow as tf
import tflearn

from markov.core import rl_utils
from markov.core import tf_utils
from markov.policies.f_approx import Network
from markov.algos.grad.dqn import DQN
from markov.memories.experience_replay import ExperienceReplay
from markov.exploration.egreedy import EpsilonGreedy

from markov.core.input.input_stream_downsampler_processor import InputStreamDownsamplerProcessor
from markov.core.input.input_stream_sequential_processor import InputStreamSequentialProcessor
from markov.core.input.input_stream_processor import InputStreamProcessor

if __name__ == "__main__":
    env = gym.make("Pong-v0")

    with tf.Session() as sess:
        input_tensor = tflearn.input_data(shape=(None, 84, 84, 4)) #tf_utils.get_input_tensor_shape(env))
        net = tflearn.conv_2d(input_tensor, 16, 8, 4, activation='relu')
        net = tflearn.conv_2d(net, 32, 4, 2, activation='relu')
        net = tflearn.flatten(net)
        net = tflearn.fully_connected(net, 1024, activation='relu')
        net = tflearn.fully_connected(net, env.action_space.n, activation='linear')

        network = Network(input_tensor,
                          net,
                          sess,
                          Network.TYPE_DQN,
                          use_clone_net=False)

        memory = ExperienceReplay(max_size=1000000)
        egreedy = EpsilonGreedy(0.9, 0.1, 1000000)

        downsampler = InputStreamDownsamplerProcessor((84, 84), gray=True)
        sequential = InputStreamSequentialProcessor(frames=4)
        input_processor = InputStreamProcessor(processor_list=[downsampler, sequential])

        dqn = DQN(env,
                  network,
                  sess,
                  memory,
                  egreedy,
                  input_processor=input_processor,
                  episode_len=100,
                  discount=0.9,
                  optimizer='adam',
                  memory_init_size=1000,
                  clip_gradients=(-10,10))

        dqn.train(max_iterations=100000000, gym_record=False)

        # average_reward = rl_utils.average_test_episodes(env,
        #                                                 network,
        #                                                 10,
        #                                                 episode_len=dqn.episode_len)
        print ("Average: ", average_reward)
