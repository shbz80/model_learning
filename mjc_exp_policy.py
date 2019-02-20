import numpy as np
import sys
import copy
from utilities import *
import math
# import PyKDL as kdl
import pykdl_utils
import hrl_geom.transformations as trans
# from hrl_geom.pose_converter import PoseConv
from urdf_parser_py.urdf import Robot
from pykdl_utils.kdl_kinematics import *
sys.path.append('/home/shahbaz/Research/Software/Spyder_ws/gps/python')
from gps.agent.agent_utils import generate_noise

# euler_from_matrix = pydart.utils.transformations.euler_from_matrix
euler_from_matrix = trans.euler_from_matrix
J_G_to_A = jacobian_geometric_to_analytic
# IK = closed_loop_IK

#pykdl stuff
f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')
robot = Robot.from_xml_string(f.read())
base_link = robot.get_root()
end_link = 'left_tool0'
# end_link = 'left_contact_point'
kdl_kin = KDLKinematics(robot, base_link, end_link)



class Policy(object):
    def __init__(self, agent_params, exp_params, skel=None, gripper=None):
        self.agent_params = agent_params
        self.exp_params = exp_params
        self.skel = skel
        self.gripper = gripper
        self.noise_gain = exp_params['noise_gain']
        # self.noise_gain = 0

        self.ref_x_traj = []
        self.curr_x_traj = []
        self.ref_q_traj = []

        dP = self.exp_params['dP']
        dV = self.exp_params['dV']
        dU = self.exp_params['dU']
        self.dt = exp_params['dt']
        self.T = exp_params['T']
        self.Tc = self.T/exp_params['t_contact_factor'] # time till contact
        self.Kp = exp_params['Kp']
        self.Kpx = exp_params['Kpx']

        # init q, q_dot
        self.init_q = self.agent_params['x0'][:dP]
        self.init_q_dot = self.agent_params['x0'][dP:]
        self.targ_x_delta = self.exp_params['target_x_delta']

        self.curr_q = self.init_q
        self.curr_q_dot = self.init_q_dot

        # # pydart FK init_q
        # skel.set_positions(self.init_q)
        # skel.set_velocities(self.init_q_dot)
        # init_erot = gripper.T[:3,:3]
        # init_erot = euler_from_matrix(init_erot)
        # init_epos = gripper.T[:3,3]
        # self.init_x = np.append(init_epos,init_erot)

        # pykdl FK init_q
        gripper_T = kdl_kin.forward(self.init_q, end_link=end_link, base_link=base_link)
        epos = np.array(gripper_T[:3,3])
        epos = epos.reshape(-1)
        erot = np.array(gripper_T[:3,:3])
        erot = euler_from_matrix(erot)
        self.init_x = np.append(epos,erot)

        # pykdl FK for target_q
        # gripper_T = kdl_kin.forward(self.target_q, end_link=end_link, base_link=base_link)
        # epos = np.array(gripper_T[:3,3])
        # epos = epos.reshape(-1)
        # erot = np.array(gripper_T[:3,:3])
        # erot = euler_from_matrix(erot)
        # self.target_x = np.append(epos,erot)
        # self.target_x = self.init_x + np.array([0, 0, -0.110, 0, 0, 0]) # no contact
        self.target_x = self.init_x + self.targ_x_delta

        self.ref_x_dot_d = (self.target_x - self.init_x)/float(self.Tc)/self.dt
        self.ref_x = self.init_x

        self.ref_q = self.init_q
        self.t = 0
        # self.expl_noise_gain = np.array([1., 1., .8, .5, .1, .1, .1])
        self.ref_x_dot_noise = generate_noise(self.T,6,self.agent_params)
        self.ref_x_dot_noise_mask = np.array([1., 1., 0, 0, 0, 0])
        self.ref_x_dot = np.zeros(6)
        # raw_input()

    def act(self, x, obs, t, noise=None):
        dP = self.exp_params['dP']
        # Kp = np.array([.15, .15, .12, .075, .05, .05, .05])
        Kp = self.Kp
        Kd = Kp*10.
        # Kpx = np.array([.5, .5, .5, .5, .5, .5])
        Kpx = self.Kpx
        Kdx = Kpx*10


        self.curr_q = x[:dP]
        self.curr_q_dot = x[dP:]

        # pykdl FK curr_x
        gripper_T = kdl_kin.forward(self.curr_q, end_link=end_link, base_link=base_link)
        epos = np.array(gripper_T[:3,3])
        epos = epos.reshape(-1)
        erot = np.array(gripper_T[:3,:3])
        erot = euler_from_matrix(erot)
        self.curr_x = np.append(epos,erot)
        self.curr_x_traj.append(copy.deepcopy(self.curr_x))

        mask = self.ref_x_dot_noise_mask
        # print mask
        common_gain = self.noise_gain
        noise = self.ref_x_dot_noise[t,:]
        # print noise
        # print mask*noise*common_gain
        # print self.ref_x_dot
        if self.t < self.Tc:
            self.ref_x_dot = self.ref_x_dot_d + mask*noise*common_gain
        else:
            self.ref_x_dot = mask*noise*common_gain
        # print self.ref_x_dot
        # raw_input()
        self.ref_x += self.ref_x_dot*self.dt
        self.error_x = self.ref_x - self.curr_x
        self.ref_x_traj.append(copy.deepcopy(self.ref_x))

        J_G = np.array(kdl_kin.jacobian(self.curr_q))
        J_G = J_G.reshape((6,7))
        J_A = J_G_to_A(J_G, self.curr_x[3:])
        J_A_inv = np.linalg.pinv(J_A)
        ref_q_dot = J_A_inv.dot(self.ref_x_dot + np.diag(Kpx).dot(self.error_x))
        self.ref_q = self.ref_q + ref_q_dot*self.dt
        self.ref_q_traj.append(copy.deepcopy(self.ref_q))

        err = (self.ref_q - self.curr_q)/self.dt
        err_dot = ref_q_dot - self.curr_q_dot
        u = np.diag(Kp).dot(err) + np.diag(Kd).dot(err_dot)

        # this is not working
        # error_x_dot = self.ref_x_dot - J_A.dot(self.curr_q_dot)
        # f = np.diag(Kpx).dot(self.error_x) + np.diag(Kdx).dot(error_x_dot)
        # u = J_A.T.dot(f)
        # noise = np.diag(self.expl_noise_gain).dot(self.noise[t,:])*self.noise_gain
        self.t += 1
        # raw_input()
        # return u + noise
        return u

    def get_traj_data(self):
        return self.ref_x_traj, self.curr_x_traj, self.ref_q_traj


    def predict(self, X, t, return_std=True):
        dP = self.exp_params['dP']
        dU = self.exp_params['dU']
        # Kp = np.array([.15, .15, .12, .075, .05, .05, .05])
        Kp = self.Kp
        Kd = Kp*10.0*0.5
        # Kpx = np.array([.5, .5, .5, .5, .5, .5])
        Kpx = self.Kpx
        Kdx = Kpx*5.

        U = np.zeros((X.shape[0],dU))
        U_noise = np.zeros((X.shape[0], dU))
        for i in range(X.shape[0]):
            x = X[i]
            self.curr_q = x[:dP]
            self.curr_q_dot = x[dP:]

            # pykdl FK curr_x
            gripper_T = kdl_kin.forward(self.curr_q, end_link=end_link, base_link=base_link)
            epos = np.array(gripper_T[:3,3])
            epos = epos.reshape(-1)
            erot = np.array(gripper_T[:3,:3])
            erot = euler_from_matrix(erot)
            self.curr_x = np.append(epos,erot)

            mask = self.ref_x_dot_noise_mask
            common_gain = self.noise_gain
            noise = self.ref_x_dot_noise[t,:]
            if t < self.Tc:
                self.ref_x_dot = self.ref_x_dot_d + mask*noise*common_gain
            else:
                self.ref_x_dot = mask*noise*common_gain
            self.ref_x[:3] = self.curr_x[:3] + self.ref_x_dot[:3]*self.dt
            self.ref_x[3:] = self.ref_x[3:] + self.ref_x_dot[3:]*self.dt
            self.error_x = self.ref_x - self.curr_x

            J_G = np.array(kdl_kin.jacobian(self.curr_q))
            J_G = J_G.reshape((6,7))
            J_A = J_G_to_A(J_G, self.curr_x[3:])
            J_A_inv = np.linalg.pinv(J_A)
            ref_q_dot = J_A_inv.dot(self.ref_x_dot + np.diag(Kpx).dot(self.error_x))
            self.ref_q = self.ref_q + ref_q_dot*self.dt

            err = self.ref_q - self.curr_q
            err_dot = ref_q_dot - self.curr_q_dot
            u = np.diag(Kp).dot(err) + np.diag(Kd).dot(err_dot)
            U[i] = u
        return U, U_noise