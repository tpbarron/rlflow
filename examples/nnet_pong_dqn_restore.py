from __future__ import print_function

import gym
import tensorflow as tf
import tflearn

from rlflow.policies.f_approx import Network
from rlflow.algos.td import DQN
from rlflow.memories import ExperienceReplay
from rlflow.exploration import EpsilonGreedy
from rlflow.core.input import InputStreamDownsamplerProcessor, InputStreamSequentialProcessor, InputStreamProcessor


if __name__ == "__main__":
    env = gym.make("Pong-v0")

    with tf.Session() as sess:
        input_tensor = tflearn.input_data(shape=(None, 84, 84, 4)) #tf_utils.get_input_tensor_shape(env))
        net = tflearn.conv_2d(input_tensor, 16, 8, 4, activation='relu')
        net = tflearn.conv_2d(net, 32, 4, 2, activation='relu')
        net = tflearn.flatten(net)
        net = tflearn.fully_connected(net, 1024, activation='relu')
        net = tflearn.fully_connected(net, env.action_space.n, activation='linear')

        network = Network(net,
                          sess,
                          Network.TYPE_DQN,
                          use_clone_net=True)

        memory = ExperienceReplay(max_size=1000000)
        egreedy = EpsilonGreedy(0.9, 0.1, 100000)

        downsampler = InputStreamDownsamplerProcessor((84, 84), gray=True)
        sequential = InputStreamSequentialProcessor(observations=4)
        input_processor = InputStreamProcessor(processor_list=[downsampler, sequential])

        dqn = DQN(env,
                  network,
                  sess,
                  memory,
                  egreedy,
                  input_processor=input_processor,
                  discount=0.99,
                  learning_rate=0.001,
                  optimizer='adagrad',
                  memory_init_size=5000,
                  clip_gradients=(-10.0, 10.0),
                  clone_frequency=10000)

        dqn.restore('/tmp/rlflow/model.ckpt-0')
        dqn.train(max_episodes=100, save_frequency=10)
        rewards = dqn.test(episodes=10)
        print ("Rewards on test: ", rewards)
