import numpy as np
from .baxter_env import BaxterEnv
from rlcore.spaces import Box
from rlcore.envs.base import Step
from rlcore.core.serializable import Serializable

class BaxterReacherEnv(BaxterEnv, Serializable):
    """
    An environment to test training the Baxter to reach a given location
    """

    def __init__(self, control=BaxterEnv.VELOCITY):
        super(BaxterReacherEnv, self).__init__(control=control)


    #def reset(self):
    #   in parent class


    def step(self, action):
        if (self.control == VELOCITY):
            self.llimb.set_joint_velocities(action)
            self.rlimb.set_joint_velocities(action)
        elif (self.control == TORQUE):
            self.llimb.set_joint_torques(action)
            self.rlimb.set_joint_torques(action)

        self.state = self.get_joint_angles()

        # reward = -(distance from goal)
        # done = within threshold of goal
        return Step(observation=self.state, reward=reward, done=done)


    @property
    def action_space(self):
        # this will be the joint space
        return Box((-np.inf, np.inf), (self.joint_space))


    @property
    def observation_space(self):
        # [baxter joint angles; goal pos]
        return Box((-np.inf, np.inf), (self.joint_space))
