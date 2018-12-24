'''
mujo agent for model learnig research
'''
import numpy as np
import matplotlib.pyplot as plt
import logging
import imp
import os
import os.path
import sys
import copy
import pprint
import pickle
import pydart2 as pydart
from utilities import *
import math

import PyKDL as kdl
import pykdl_utils
import hrl_geom.transformations as trans
from hrl_geom.pose_converter import PoseConv
from urdf_parser_py.urdf import Robot
from pykdl_utils.kdl_kinematics import *


# Add gps/python to path so that imports work.
sys.path.append('/home/shahbaz/Research/Software/Spyder_ws/gps/python')
# from gps.gui.gps_training_gui import GPSTrainingGUI
# from gps.utility.data_logger import DataLogger
from gps.utility.data_logger import DataLogger
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.sample.sample_list import SampleList
from gps.agent.agent_utils import generate_noise
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 7,
}

common = {
    'conditions': 1,
}

# skel_path = "/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf"
# skel_path = "/home/shahbaz/Research/Software/Spyder_ws/gps/mjc_models/yumi_mjcf_l_peg_model_learning.urdf"
f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')


agent_hyperparams = {
    'type': AgentMuJoCo,
    # 'filename': '/home/shahbaz/Research/Software/Spyder_ws/gps/mjc_models/yumi_mjcf_l_peg_model_learning.xml',
    'filename': '/home/shahbaz/Research/Software/Spyder_ws/gps/mjc_models/yumi_model_learning_blocks_1.xml',
    # 'filename': '/home/shahbaz/Research/Software/Spyder_ws/gps/mjc_models/yumi_mjcf_rl_peg.xml',
    # 'filename': '/home/shahbaz/Research/Software/model_learning/model_learning_1d.xml',
    # 'x0': np.concatenate([np.array([0.4, -2.2, -0.7, 0.35, 0.7, 0., -1.]),
    #                       np.zeros(7)]),
    'x0': np.concatenate([np.array([-1.366, -1.094, 1.085, 0.901, 1.999, 1.636, -2.912]),
                          np.zeros(7)]),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [[np.array([0, 0, 0])]],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [],
    'camera_pos': np.array([5.0, 0.5, 3., 0., 0., 0.]),
    'Kp': np.array([1.0, 1.0, 1., 1., 1., 1.]),
    'Kd': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    'smooth_noise': True,
    'smooth_noise_var': 1.,
    'smooth_noise_renormalize': True,
}

exp_params = {
            'dt': agent_hyperparams['dt'],
            'T': agent_hyperparams['T'],
            'num_samples': 40, # only even number, to be slit into 2 sets
            'dP': SENSOR_DIMS[JOINT_ANGLES],
            'dV': SENSOR_DIMS[JOINT_VELOCITIES],
            'dU': SENSOR_DIMS[ACTION],
            'mean_action': 0.,
            'target_end_effector': np.array([0.4 ,-0.45, 0.25, 0.4, -0.45, 0.05]),
            'target_joint_pos': np.array([-1.594, -1.319, 1.597, 0.425, 2.467, 1.312, -2.771]),
            'noise_gain': 0.075,
            # 'noise_gain': 0.1,
            # 'target_joint_pos': np.array([-1.4, -1.319, 1.597, 0.425, 2.467, 1.312, -2.771]),
            # 'init_end_effector': np.array([0.4 ,-0.45, 0.25, 0.4, -0.45, 0.05]),
}

num_samples = exp_params['num_samples']
cond = 0
dU = SENSOR_DIMS[ACTION]
noise_gain = exp_params['noise_gain']
noise_mean = np.zeros(dU)
noise_std = np.ones(dU)*noise_gain
mean_action = exp_params['mean_action']
T = exp_params['T']
dt = exp_params['dt']
dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']

euler_from_matrix = pydart.utils.transformations.euler_from_matrix
J_G_to_A = jacobian_geometric_to_analytic
# IK = closed_loop_IK


class Policy(object):
    def __init__(self, agent_params, exp_params, skel=None, gripper=None):
        self.agent_params = agent_params
        self.exp_params = exp_params
        self.skel = skel
        self.gripper = gripper
        self.noise_gain = exp_params['noise_gain']
        # self.noise_gain = 0

        dP = self.exp_params['dP']
        dV = self.exp_params['dV']
        dU = self.exp_params['dU']
        self.dt = exp_params['dt']
        self.T = exp_params['T']
        self.Tc = self.T/3 # time till contact

        # init q, q_dot
        self.init_q = self.agent_params['x0'][:dP]
        self.init_q_dot = self.agent_params['x0'][dP:]

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
        self.target_x = self.init_x + np.array([0, 0, -0.122, 0, 0, 0])

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

        # Kp = np.array([.5, .5, .4, .25, .05, .05, .05]) # good tracking
        # Kp = np.array([.5, .5, .4, .25, .05, .05, .05])*0.3 # compliant
        Kp = np.array([.15, .15, .12, .075, .05, .05, .05])
        Kd = Kp*10.
        # Kpx = np.array([.5, .5, .5, .5, .5, .5]) # good tracking for IK
        Kpx = np.array([.5, .5, .5, .5, .5, .5])
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
        curr_x_traj.append(copy.deepcopy(self.curr_x))

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
        ref_x_traj.append(copy.deepcopy(self.ref_x))

        J_G = np.array(kdl_kin.jacobian(self.curr_q))
        J_G = J_G.reshape((6,7))
        J_A = J_G_to_A(J_G, self.curr_x[3:])
        J_A_inv = np.linalg.pinv(J_A)
        ref_q_dot = J_A_inv.dot(self.ref_x_dot + np.diag(Kpx).dot(self.error_x))
        self.ref_q = self.ref_q + ref_q_dot*self.dt
        ref_q_traj.append(copy.deepcopy(self.ref_q))

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

# pydart stuff
# pydart.init()
# print("pydart init OK")
# world = pydart.World(0.0002)
# print("World init OK")
# world.g = [0.0, 0.0, 0] # gravity is set to zero
# print("gravity = %s" % str(world.g))
# skel = world.add_skeleton(skel_path)
# # skel.set_root_joint_to_trans_and_euler()
# # skel.set_root_joint_to_weld()
# print("Skeleton add OK")
# print skel.bodynodes
# gripper = skel.bodynodes[-1]

#pykdl stuff
robot = Robot.from_xml_string(f.read())
base_link = robot.get_root()
end_link = 'left_tool0'
# end_link = 'left_contact_point'
kdl_kin = KDLKinematics(robot, base_link, end_link)


mjc_agent = AgentMuJoCo(agent_hyperparams)

for i in range(num_samples):
    ref_x_traj =[]
    curr_x_traj = []
    ref_q_traj = []

    pol = Policy(agent_hyperparams, exp_params)
    mjc_agent.sample(
                    pol, cond,
                    verbose=True,
                    noisy=True
                )
    del pol

SampleList = mjc_agent.get_samples(cond, -num_samples)
mjc_agent.clear_samples()
ref_x_traj = np.array(ref_x_traj)
curr_x_traj = np.array(curr_x_traj)
ref_q_traj = np.array(ref_q_traj)

Xs = SampleList.get_X()
Us = SampleList.get_U()

# plot samples for sample 0
sample = SampleList[-1]
Js = sample.get(JOINT_ANGLES)
J_dots = sample.get(JOINT_VELOCITIES)
# Es = sample.get(END_EFFECTOR_POINTS)
# E_dots = sample.get(END_EFFECTOR_POINT_VELOCITIES)
Tqs = sample.get(ACTION)

plt.figure()
tm = np.linspace(0,T*dt,T)
#jPos
for j in range(7):
    plt.subplot(5,7,1+j)
    plt.title('j%dPos' %(j+1))
    plt.plot(tm,Js[:,j],color='g')
    plt.plot(tm,ref_q_traj[:,j])


#jVel
for j in range(7):
    plt.subplot(5,7,8+j)
    plt.title('j%dVel' %(j+1))
    plt.plot(tm,J_dots[:,j],color='b')

#jTrq
for j in range(7):
    plt.subplot(5,7,15+j)
    plt.title('j%dTrq' %(j+1))
    plt.plot(tm,Tqs[:,j],color='r')
# EE
for j in range(6):
    plt.subplot(5,7,22+j)
    plt.title('EE%dPos' %(j+1))
    plt.plot(tm,ref_x_traj[:,j])
    plt.plot(tm,curr_x_traj[:,j])

# plt.figure()
# for j in range(3):
#     plt.subplot(2,7,1+j)
#     plt.title('EE%dPos' %(j+1))
#     plt.plot(tm,Es[:,j],color='m')
#     plt.plot(tm,ref_x_traj[:,j])
#
# for j in range(3):
#     plt.subplot(2,7,8+j)
#     plt.title('EE%dVel' %(j+1))
#     plt.plot(tm,E_dots[:,j],color='c')

plt.show()

sample_data = {}
sample_data['exp_params'] = exp_params
# sample_data['agent_params'] = agent_hyperparams
sample_data['X'] = Xs
sample_data['U'] = Us

# raw_input()
pickle.dump( sample_data, open( "mjc_blocks_raw_10_10.p", "wb" ) )
