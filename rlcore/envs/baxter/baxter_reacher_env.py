import numpy as np
from .baxter_env import BaxterEnv
from rlcore.spaces import Box
from rlcore.envs.base import Step
from rlcore.core.serializable import Serializable

class BaxterReacherEnv(BaxterEnv, Serializable):
    """
    An environment to test training the Baxter to reach a given location
    """

    def __init__(self, **kwargs):
        super(BaxterReacherEnv, self).__init__(**kwargs)


    def reset(self):
        raise NotImplementedError


    def step(self, action):
        raise NotImplementedError


    @property
    def action_space(self):
        # this will be the joint space
        raise NotImplementedError


    @property
    def observation_space(self):
        # 
        raise NotImplementedError
