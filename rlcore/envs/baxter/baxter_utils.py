from __future__ import print_function
import argparse, csv, os, sys
import numpy as np
import matplotlib.pyplot as plt
import rospy
import baxter_interface
from baxter_interface import CHECK_VERSION
from baxter_trajectory import Trajectory

class BaxterUtils:

    POSITION, VELOCITY = range(2)
    LEFT_LIMB, RIGHT_LIMB, BOTH_LIMBS = range(3)

    DOF = 17
    DOF_NO_TIME = 16
    DOF_NO_TIME_NO_GRIPPER = 14
    DOF_NO_TIME_LEFT = 8
    DOF_NO_TIME_RIGHT = 8
    DOF_NO_TIME_NO_GRIPPER_LEFT = 7
    DOF_NO_TIME_NO_GRIPPER_RIGHT = 7

    def __init__(self):
        # print ("Initializing node...")
        # rospy.init_node("baxter_control")
        # print ("Getting robot state...")
        # self.rs = baxter_interface.RobotEnable(CHECK_VERSION)
        # self.init_state = self.rs.state().enabled
        # rospy.on_shutdown(self.clean_shutdown)
        # print ("Enabling robot...")
        # self.rs.enable()
        #
        # self.left = baxter_interface.Limb('left')
        # self.right = baxter_interface.Limb('right')
        # self.grip_left = baxter_interface.Gripper('left', CHECK_VERSION)
        # self.grip_right = baxter_interface.Gripper('right', CHECK_VERSION)
        # self.rate = rospy.Rate(1000)
        # self.check_init()
        self.trajs = []
        self.keys = None


    def check_init(self):
        if self.grip_left.error():
            self.grip_left.reset()
        if self.grip_right.error():
            self.grip_right.reset()
        if (not self.grip_left.calibrated() and self.grip_left.type() != 'custom'):
            self.grip_left.calibrate()
        if (not self.grip_right.calibrated() and self.grip_right.type() != 'custom'):
            self.grip_right.calibrate()


    def clean_shutdown(self):
        print ("Exiting")
        if not self.init_state:
            print ("Disabling robot")
            self.rs.disable()


    def limb_in_sphere(self, limb, sphere):
        """
        sphere - a sphere defined by (x, y, z, r)
        limb - a baxter limb object
        Returns True if any segment of limb is in space
        """
        states = limb.get_link_states()
        links = limb._link_names[limb.name]
        for i in range(len(links)-1):
            l1 = links[i]
            l2 = links[i+1]
            p1 = states[l1].position
            # print ("p1 = ", p1)
            p2 = states[l2].position
            # print ("p2 = ", p2)
            p1 = (p1.x, p1.y, p1.z)
            p2 = (p2.x, p2.y, p2.z)
            if (self.check_segment(sphere, (p1, p2))):
                return True
        return False


    def check_segment(self, sphere, segment):
        """
        segment = ((x1,y1,z1), (x2,y2,z2))
        returns True if intersection
        """
        p1, p2 = segment
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3, r = sphere
        a = (x2-x1)**2.0 + (y2-y1)**2.0 + (z2-z1)**2.0
        b = 2*( (x2-x1)*(x1-x3) + (y2-y1)*(y1-y3) + (z2-z1)*(z1-z3) )
        c = x3**2.0 + y3**2.0 + z3**2.0 + x1**2.0 + y1**2.0 + z1**2.0 - 2*(x3*x1 + y3*y1 + z3*z1) - r**2.0

        u1 = (-b + np.sqrt(b*b-4*a*c)) / 2*a
        u2 = (-b - np.sqrt(b*b-4*a*c)) / 2*a
        if ((u1 < 0 and u2 < 0) or (u1 > 1 and u2 > 1)):
            # line seg outside sphere
            return False
        if ((u1 < 0 and u2 > 1) or (u1 > 1 and u2 < 0)):
            # line segment inside sphere
            return True
        if ((abs(u1) < 1 and (u2 > 1 or u2 < 0)) or (abs(u2) < 1 and (u1 > 1 or u1 < 0))):
            # line segment intersects at one point
            return True
        if (abs(u1) < 1 and abs(u2) < 1):
            # intersects at two points
            return True
        if (abs(u1) < 1 and abs(u2) < 1 and u1 - u2 < .001):
            # tangential
            return True
        return False



    def load_trajectories(self, dir):
        """
        Load all trajectories from dir into a list of numpy arrays
        """
        self.trajs = []
        self.keys = None
        files = os.listdir(dir)
        for f in files:
            self.keys, traj = self.load_trajectory(os.path.join(dir, f))
            self.trajs.append(traj)


    def load_trajectory(self, fname):
        """
        Load the trajectory in the file fname to a numpy array
        """
        with open(fname, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            keys = reader.next()
            data = []
            for row in reader:
                row = [float(f) for f in row]
                data.append(row)
            data = np.array(data)
            return keys, data


    def get_trajectories(self):
        return self.trajs


    def get_trajectories_without_time(self, limb=BOTH_LIMBS):
        if (limb == BaxterUtils.BOTH_LIMBS):
            trajs = [t[:,1:] for t in self.trajs]
            return trajs
        elif (limb == BaxterUtils.LEFT_LIMB):
            # remove time
            trajs = [t[:,1:] for t in self.trajs]
            trajs = [t[:,:8] for t in trajs]
            return trajs
        elif (limb == BaxterUtils.RIGHT_LIMB):
            # remove time
            trajs = [t[:,1:] for t in self.trajs]
            trajs = [t[:,8:] for t in trajs]
            return trajs


    def get_trajectories_without_time_and_gripper(self, limb=BOTH_LIMBS):
        if (limb == BaxterUtils.BOTH_LIMBS):
            trajs = [np.hstack((t[:,1:8],t[:,9:16])) for t in self.trajs]
            return trajs
        elif (limb == BaxterUtils.LEFT_LIMB):
            trajs = [t[:,1:8] for t in self.trajs]
            return trajs
        elif (limb == BaxterUtils.RIGHT_LIMB):
            trajs = [t[:,9:16] for t in self.trajs]
            return trajs


    def interpolate_time(self, secs, timesteps):
        time = np.linspace(0, secs, timesteps)
        return time


    def run_loaded_trajectory(self, t=0, mode=POSITION):
        if mode == BaxterUtils.POSITION:
            self.run_position_trajectory(self.trajs[t])
        elif mode == BaxterUtils.VELOCITY:
            self.run_velocity_trajectory(self.trajs[t])


    def run_trajectory(self, traj, mode=POSITION):
        if (mode == BaxterUtils.POSITION):
            self.run_position_trajectory(traj)
        elif (mode == BaxterUtils.VELOCITY):
            self.run_velocity_trajectory(traj)


    def run_velocity_trajectory(self, traj_data):
        traj = Trajectory()
        traj.parse_traj(self.keys, traj_data)
        #for safe interrupt handling
        rospy.on_shutdown(traj.stop)
        result = True
        print ("Starting trajectory")
        traj.start()
        result = traj.wait()
        print("Result: " + str(result) + ", Playback Complete")


    def run_position_trajectory(self, traj):
        # traj is a numpy array where each column is a DOF
        # and each row is a timestep
        time, lstart, rstart = self.get_cmds_from_row(traj[0])
        self.left.move_to_joint_positions(lstart)
        self.right.move_to_joint_positions(rstart)

        start_time = rospy.get_time()
        for t in range(traj.shape[0]): # for each row
            sys.stdout.write("\r Record %d of %d" %
                             (t, traj.shape[0]))
            sys.stdout.flush()
            time, lcmd, rcmd = self.get_cmds_from_row(traj[t])
            # send these commands until the next frame
            while (rospy.get_time() - start_time) < time:
                if rospy.is_shutdown():
                    print ("ROS shutdown, aborting")
                    return
                self.left.set_joint_positions(lcmd)
                self.right.set_joint_positions(rcmd)
                # TODO: future gripper handling?
                self.rate.sleep()


    def get_cmds_from_row(self, row):
        limb_dof = 7
        row_list = row.tolist()
        row_dict = dict(zip(self.keys, row_list))

        # create a dictionary of joint to value for left limb
        # skip element 0 because it is the time param
        # skip elements 8, 16 because they are gripper params

        ldict = {k: row_dict[k] for k in self.keys[1:limb_dof+1]}
        rdict = {k: row_dict[k] for k in self.keys[limb_dof+2:-1]}

        return row[0], ldict, rdict


    def get_num_dof(self):
        return self.trajs[0].shape[1]


    def get_limb_coordinate(self):
        return self.left.endpoint_pose(), self.right.endpoint_pose()


    def plot_trajectory(self, t=0):
        """
        Plot the given trajectory for each dof
        """
        traj = self.trajs[t]
        for i in range(traj.shape[1]): # num dofs
            plt.subplot(3,6,i+1)
            plt.plot(traj[:,i], lw=2)
            plt.title("Traj for DOF="+str(i))
            plt.xlabel('timesteps')
            plt.ylabel('joint position')
        plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baxter controller')
    parser.add_argument('-d', '--dir', dest='dir', required=True)
    args = parser.parse_args()

    bc = BaxterUtils()
    bc.load_trajectories(args.dir)
    bc.run_trajectory()
    #bc.plot_trajectory()
