from __future__ import print_function
import sys
import numpy as np
from .baxter_env import BaxterEnv
from rlcore.spaces import Box
from rlcore.envs.base import Step
from rlcore.core.serializable import Serializable
from rlcore.misc.overrides import overrides

class BaxterReacherEnv(BaxterEnv, Serializable):
    """
    An environment to test training the Baxter to reach a given location
    """

    def __init__(self, control=BaxterEnv.VELOCITY):
        super(BaxterReacherEnv, self).__init__(control=control)

        self.goal = np.random.rand(6)
        self.threshold = .1


    #def reset(self):
    #   in parent class

    @overrides
    def step(self, action):
        # print ("Taking step: ", action)
        # if (np.isnan(np.sum(action))):
        #     print ("Action has nan")
        #     sys.exit(1)
        # TODO: this needs to be split up for each arm
        laction = self.get_joint_action_dict(action, BaxterEnv.LEFT_LIMB)
        raction = self.get_joint_action_dict(action, BaxterEnv.RIGHT_LIMB)
        if (self.control == BaxterEnv.VELOCITY):
            self.llimb.set_joint_velocities(laction)
            self.rlimb.set_joint_velocities(raction)
        elif (self.control == BaxterEnv.TORQUE):
            self.llimb.set_joint_torques(laction)
            self.rlimb.set_joint_torques(raction)

        self.state = self.get_joint_angles()

        # reward = -(distance from goal)
        dist = np.linalg.norm(self.goal-self.get_endeff_position())
        reward = -dist
        # done = within threshold of goal
        done = True if dist < self.threshold else False
        return Step(observation=self.state, reward=reward, done=done)


    @property
    @overrides
    def action_space(self):
        # this will be the joint space
        return Box(-np.inf, np.inf, (self.joint_space,))


    @property
    @overrides
    def observation_space(self):
        # [baxter joint angles; goal pos]
        return Box(-np.inf, np.inf, (self.joint_space,))
