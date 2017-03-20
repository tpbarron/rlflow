"""
Record a series of (random) transitions for model building
"""
from gym.envs.atari import AtariEnv
import tensorflow as tf
import numpy as np

from rlflow.policies import RandomPolicy
from rlflow.algos.algo import RLAlgorithm
from rlflow.core.input import InputStreamDownsamplerProcessor, InputStreamSequentialProcessor, InputStreamProcessor


if __name__ == "__main__":
    env = AtariEnv(game='pong', obs_type='image', frameskip=(1, 2))

    random_pol = RandomPolicy(env.action_space)

    downsampler = InputStreamDownsamplerProcessor((28, 28), gray=True)
    input_processor = InputStreamProcessor(processor_list=[downsampler])

    algo = RLAlgorithm(env,
                       random_pol,
                       np.inf, # episode len
                       1.0, # discount
                       False, # standardize
                       input_processor, # input_processor
                       None, # optimizer
                       None) # clip_gradients

    algo.test(episodes=10,
              record_experience=True,
              record_experience_path='/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/data/pong_random_small/')
