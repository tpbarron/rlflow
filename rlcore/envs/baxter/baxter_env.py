from __future__ import print_function

from rlcore.envs.base import Env
from rlcore.core.serializable import Serializable


class BaxterEnv(Env, Serializable):
    """
    The connection with the Baxter or Gazebo must be initialized before using
    this environment
    """


    def __init__(self):
        Serializable.quick_init(self, locals())
        # TODO: Initialize baxter connection and set up state vars for joints


    def reset(self):
        raise NotImplementedError


    def step(self, action):
        raise NotImplementedError


    @property
    def action_space(self):
        raise NotImplementedError


    @property
    def observation_space(self):
        raise NotImplementedError
