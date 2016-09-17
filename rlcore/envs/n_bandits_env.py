import numpy as np
from .base import Env
from rlcore.envs.base import Step
from rlcore.core.serializable import Serializable
from rlcore.spaces import Discrete


class NBanditsEnv(Env, Serializable):

    def __init__(self, nbandits, episode_len, random_walk=False):
        Serializable.quick_init(self, locals())
        self.nbandits = nbandits
        self.episode_len = episode_len
        self.random_walk = random_walk
        self.reset()


    def reset(self):
        self.itr = 0
        self.bandits = np.random.normal(0.0, 1.0, self.nbandits)
        observation = self.bandits
        return observation


    def get_observation(self):
        if (self.random_walk):
            # just deviate from the current values slightly
            self.bandits += np.random.normal(0.0, 0.001, self.nbandits)
        # generate noise
        noise = np.random.normal(0.0, 1.0, self.nbandits)
        obs = self.bandits + noise
        return obs


    def step(self, action):
        done=False
        self.itr += 1
        if (self.itr >= self.episode_len):
            done=True

        obs = self.get_observation()
        reward = obs[action]
        return Step(observation=obs, reward=reward, done=done)


    @property
    def action_space(self):
        return Discrete(self.nbandits)


    @property
    def observation_space(self):
        return Discrete(self.nbandits)
