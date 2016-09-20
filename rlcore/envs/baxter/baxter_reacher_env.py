from __future__ import print_function
import sys, time
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

    def __init__(self, control=BaxterEnv.POSITION, limbs=BaxterEnv.BOTH_LIMBS):
        super(BaxterReacherEnv, self).__init__(control=control, limbs=limbs)

        # left(x, y, z), right(x, y, z)
        # x is forward, y is left, z is up
        if (self.limbs == BaxterEnv.BOTH_LIMBS):
            self.goal = np.array([np.random.random(), -np.random.random(), 1.0,
                                  np.random.random(), np.random.random(), 1.0])
        elif (self.limbs == BaxterEnv.LEFT_LIMB):
            self.goal = np.array([np.random.random(), -np.random.random(), 1.0])
        elif (self.limbs == BaxterEnv.RIGHT_LIMB):
            self.goal = np.array([np.random.random(), np.random.random(), 1.0])

        self.threshold = .5


    @overrides
    def reset(self):
        if (self.limbs == BaxterEnv.BOTH_LIMBS):
            self.llimb.move_to_neutral()
            self.rlimb.move_to_neutral()
            self.state = self.get_joint_angles()
            return np.hstack((self.state, self.goal))
        elif (self.limbs == BaxterEnv.LEFT_LIMB):
            self.llimb.move_to_neutral()
            self.state = self.get_joint_angles()
            return np.hstack((self.state, self.goal))
        if (self.limbs == BaxterEnv.RIGHT_LIMB):
            self.rlimb.move_to_neutral()
            self.state = self.get_joint_angles()
            return np.hstack((self.state, self.goal))


    @overrides
    def step(self, action):
        if (self.limbs == BaxterEnv.BOTH_LIMBS):
            laction, raction = self.get_joint_action_dict(action)
            if (self.control == BaxterEnv.VELOCITY):
                self.llimb.set_joint_velocities(laction)
                self.rlimb.set_joint_velocities(raction)
            elif (self.control == BaxterEnv.TORQUE):
                self.llimb.set_joint_torques(laction)
                self.rlimb.set_joint_torques(raction)
            elif (self.control == BaxterEnv.POSITION):
                self.llimb.set_joint_positions(laction)
                self.rlimb.set_joint_positions(raction)
        elif (self.limbs == BaxterEnv.LEFT_LIMB):
            laction = self.get_joint_action_dict(action)
            if (self.control == BaxterEnv.VELOCITY):
                self.llimb.set_joint_velocities(laction)
            elif (self.control == BaxterEnv.TORQUE):
                self.llimb.set_joint_torques(laction)
            elif (self.control == BaxterEnv.POSITION):
                self.llimb.set_joint_positions(laction)
        elif (self.limbs == BaxterEnv.RIGHT_LIMB):
            raction = self.get_joint_action_dict(action)
            if (self.control == BaxterEnv.VELOCITY):
                self.rlimb.set_joint_velocities(raction)
            elif (self.control == BaxterEnv.TORQUE):
                self.rlimb.set_joint_torques(raction)
            elif (self.control == BaxterEnv.POSITION):
                self.rlimb.set_joint_positions(raction)

        self.control_rate.sleep()
        self.state = np.hstack((self.get_joint_angles(), self.goal))

        # reward = -dist from goal
        # reward = 1 if dist < threshold, else 0
        dist = np.linalg.norm(self.goal-self.get_endeff_position())
        reward = 1 if dist < self.threshold else 0

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
        return Box(-np.inf, np.inf, (self.joint_space + self.goal.size,))
