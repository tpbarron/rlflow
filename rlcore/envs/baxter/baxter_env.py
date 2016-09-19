from __future__ import print_function

from rlcore.envs.base import Env
from rlcore.core.serializable import Serializable

import rospy
from dynamic_reconfigure.server import Server
from std_msgs.msg import Empty
import baxter_interface
from baxter_examples.cfg import JointSpringsExampleConfig
from baxter_interface import CHECK_VERSION


class BaxterEnv(Env, Serializable):
    """
    The connection with the Baxter or Gazebo must be initialized before using
    this environment
    """

    RIGHT_LIMB = "right"
    LEFT_LIMB = "left"

    TORQUE = "torque"
    VELOCITY = "velocity"

    def __init__(self, control=None):
        Serializable.quick_init(self, locals())

        print("Initializing node... ")
        rospy.init_node("rlcore_baxter_env")

        if control == None:
            control = BaxterEnv.VELOCITY
        self.control = control

        # create our limb instance
        self.llimb = baxter_interface.Limb(LEFT_LIMB)
        self.rlimb = baxter_interface.Limb(RIGHT_LIMB)

        # robot state
        self.joint_space = 2*len(self.llimb.joint_angles())
        self.state = np.zeros(self.joint_space)

        # verify robot is enabled
        print("Getting Baxter state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling Baxter... ")
        self._rs.enable()

        rospy.on_shutdown(self.clean_shutdown)


    def clean_shutdown(self):
        """
       Switches out of joint torque mode to exit cleanly
       """
        print("Exiting Baxter env...")
        self.llimb.exit_control_mode()
        self.rlimb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()


    def get_joint_angles(self):
        return np.array(self.llimb.joint_angles() + self.rlimb.joint_angles())


    def reset(self):
        self.llimb.move_to_neutral()
        self.rlimb.move_to_neutral()
        self.state = self.get_joint_angles()
        return self.state


    def step(self, action):
        raise NotImplementedError


    @property
    def action_space(self):
        raise NotImplementedError


    @property
    def observation_space(self):
        raise NotImplementedError
