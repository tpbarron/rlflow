"""
Record a series of (random) transitions for model building
"""
import gym
from gym.envs.atari import AtariEnv
import tensorflow as tf
import numpy as np

from rlflow.policies import RandomPolicy
from rlflow.algos.algo import RLAlgorithm
from rlflow.core.input import InputStreamDownsamplerProcessor, InputStreamSequentialProcessor, InputStreamProcessor
import math

if __name__ == "__main__":
    # env = gym.make('Pendulum-v0')
    # env = gym.make('CartPole-v0')
    # env.theta_threshold_radians = math.pi / 2.
    # env = gym.make('MountainCarContinuous-v0')
    env = gym.make('DoomDefendLine-v0')
    # env = gym.make('Breakout-v0')
    # env = AtariEnv(game='mspacman', obs_type='image', frameskip=(1, 2))

    random_pol = RandomPolicy(env.action_space)

    downsampler = InputStreamDownsamplerProcessor((32, 32), gray=True, scale=False)
    input_processor = InputStreamProcessor(processor_list=[downsampler])

    algo = RLAlgorithm(env,
                       random_pol,
                       np.inf, # episode len
                       1.0, # discount
                       False, # standardize
                       None, #input_processor, # input_processor
                       None, # optimizer
                       None) # clip_gradients

    algo.test(episodes=1000,
              record_experience=True,
              record_experience_path='/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/data/minecraft_default_random_states/',
              save_images=False)
