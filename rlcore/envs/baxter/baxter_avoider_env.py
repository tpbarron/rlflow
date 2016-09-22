from __future__ import print_function
import sys, time
import numpy as np
from .baxter_env import BaxterEnv
from .baxter_utils import BaxterUtils
from rlcore.spaces import Box
from rlcore.envs.base import Step
from rlcore.core.serializable import Serializable
from rlcore.misc.overrides import overrides

class BaxterAvoiderEnv(BaxterEnv, Serializable):
    """
    An environment to test training the Baxter to reach a given location
    """

    def __init__(self, timesteps, sphere, control=BaxterEnv.POSITION, limbs=BaxterEnv.BOTH_LIMBS,):
        super(BaxterAvoiderEnv, self).__init__(timesteps, sphere, control=control, limbs=limbs)

        # left(x, y, z), right(x, y, z)
        # x is forward, y is left, z is up
        if (self.limbs == BaxterEnv.BOTH_LIMBS):
            self.goal = np.array([np.random.random(), -np.random.random(), 1.0,
                                  np.random.random(), np.random.random(), 1.0])
        elif (self.limbs == BaxterEnv.LEFT_LIMB):
            self.goal = np.array([np.random.random(), -np.random.random(), 1.0])
        elif (self.limbs == BaxterEnv.RIGHT_LIMB):
            self.goal = np.array([np.random.random(), np.random.random(), 1.0])

        self.sphere = sphere
        self.step_in_sphere = 0
        self.bax_utils = BaxterUtils()


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

        self.step_in_sphere = 0


    @overrides
    def step(self, action, **kwargs):
        if (self.limbs == BaxterEnv.BOTH_LIMBS):
            laction, raction = self.get_joint_action_dict(action)
            assert(len(laction)) == len(self.llimb.joint_angles())
            assert(len(raction)) == len(self.rlimb.joint_angles())
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
            assert(len(laction)) == len(self.llimb.joint_angles())
            if (self.control == BaxterEnv.VELOCITY):
                self.llimb.set_joint_velocities(laction)
            elif (self.control == BaxterEnv.TORQUE):
                self.llimb.set_joint_torques(laction)
            elif (self.control == BaxterEnv.POSITION):
                self.llimb.set_joint_positions(laction)
        elif (self.limbs == BaxterEnv.RIGHT_LIMB):
            raction = self.get_joint_action_dict(action)
            assert(len(raction)) == len(self.rlimb.joint_angles())
            if (self.control == BaxterEnv.VELOCITY):
                self.rlimb.set_joint_velocities(raction)
            elif (self.control == BaxterEnv.TORQUE):
                self.rlimb.set_joint_torques(raction)
            elif (self.control == BaxterEnv.POSITION):
                self.rlimb.set_joint_positions(raction)

        self.control_rate.sleep()
        self.state = np.hstack((self.get_joint_angles(), self.goal))

        # check if in area
        if (self.limbs == BaxterEnv.BOTH_LIMBS):
            if (self.bax_utils.limb_in_sphere(self.llimb, self.sphere) or
                    self.bax_utils.limb_in_sphere(self.rlimb, self.sphere)):
                self.step_in_sphere += 1
        elif (self.limbs == BaxterEnv.LEFT_LIMB):
            if (self.bax_utils.limb_in_sphere(self.llimb, self.sphere)):
                self.step_in_sphere += 1
        elif (self.limbs == BaxterEnv.RIGHT_LIMB):
            if (self.bax_utils.limb_in_sphere(self.rlimb, self.sphere)):
                self.step_in_sphere += 1


        # only consider reward at end of task
        # if we are finished, the reward is weighted combination of the distance
        # from the goal and the number of steps in the space
        done = True if kwargs['t'] == self.timesteps-1 else False
        if done:
            dist = np.linalg.norm(self.goal-self.get_endeff_position())
            dist_reward = np.exp(-dist)
            space_reward = np.exp(-float(self.step_in_sphere) / self.timesteps)
            print ("num steps in sphere: ", self.step_in_sphere, dist_reward, space_reward)
            reward = .5*dist_reward+.5*space_reward # if at goal reward = 1, else asymptopes to 0
            # TODO: fix this
            self.step_in_sphere = 0
        else:
            reward = 0.0

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
