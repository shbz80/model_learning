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
import pickle
from mjc_exp_policy import Policy
from model_leraning_utils import YumiKinematics


# Add gps/python to path so that imports work.
sys.path.append('/home/shahbaz/Research/Software/Spyder_ws/gps/python')
# from gps.gui.gps_training_gui import GPSTrainingGUI
# from gps.utility.data_logger import DataLogger
from gps.utility.data_logger import DataLogger
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.sample.sample_list import SampleList
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

# use this for yumi_gps_generated.urdf
# f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_gps_generated.urdf', 'r')
# base_link = 'yumi_base_link'
# end_link = 'gripper_l_base'
# yumiKin = YumiKinematics(f, base_link, end_link, euler_string='szyx', reverse_angles=True)

# use this for yumi_ABB_left.urdf
f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')
base_link = 'world'
end_link = 'left_gripper_base'
yumiKin = YumiKinematics(f, base_link, end_link, euler_string='szyx', reverse_angles=True)

agent_hyperparams = {
    'type': AgentMuJoCo,
    # 'filename': '/home/shahbaz/Research/Software/Spyder_ws/gps/mjc_models/yumi_mjcf_l_peg_model_learning.xml',
    'filename': '/home/shahbaz/Research/Software/Spyder_ws/gps/mjc_models/yumi_model_learning_blocks_2.xml',
    # 'x0': np.concatenate([np.array([0.4, -2.2, -0.7, 0.35, 0.7, 0., -1.]),
    #                       np.zeros(7)]),
    'x0': np.concatenate([np.array([-1.3033, -1.3531, 0.9471, 0.3177, 2.0745, 1.4900, -2.1547]),
                          np.zeros(7)]),
    'x0var': np.concatenate((np.full(7, 0.001), np.full(7, 0.001))),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [[np.array([0, 0, 0])]],
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [],
    'camera_pos': np.array([5.0, 0.5, 3., 0., 0., 0.]),
    'smooth_noise': False,
    'smooth_noise_var': 1.,
    'smooth_noise_renormalize': False,
}

exp_params_yumi = {
            'dt': agent_hyperparams['dt'],
            'T': agent_hyperparams['T'],
            'num_samples': 10, # only even number, to be slit into 2 sets
            'dP': SENSOR_DIMS[JOINT_ANGLES],
            'dV': SENSOR_DIMS[JOINT_VELOCITIES],
            'dU': SENSOR_DIMS[ACTION],
            'mean_action': 0.,
            'x0': np.concatenate([np.array([-1.3033, -1.3531, 0.9471, 0.3177, 2.0745, 1.4900, -2.1547]),
                          np.zeros(7)]),
            'target_x': np.array([ 0.39067804, 0.14011851, -0.06375249, 0.31984032, 1.55309358, 1.93199837]),
            'target_x_delta': np.array([-0.1, -0.1, -0.1, 0.0, 0.0, 0.0]),
            'Kp': np.array([0.22, 0.22, 0.18, 0.15, 0.05, 0.05, 0.025])*100.0*0.5,
            'Kd': np.array([0.07, 0.07, 0.06, 0.05, 0.015, 0.015, 0.01])*10.0*0.5,
            'Kpx': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*0.7,
            'noise_gain': 0.01,
            't_contact_factor': 3,
            'joint_space_noise': None,
}

exp_params_mjc = {
            'dt': agent_hyperparams['dt'],
            'T': agent_hyperparams['T'],
            'num_samples': 15,
            'dP': SENSOR_DIMS[JOINT_ANGLES],
            'dV': SENSOR_DIMS[JOINT_VELOCITIES],
            'dU': SENSOR_DIMS[ACTION],
            'mean_action': 0.,
            'x0': np.concatenate([np.array([-1.3033, -1.3531, 0.9471, 0.3177, 2.0745, 1.4900, -2.1547]),
                          np.zeros(7)]),
            'target_x': np.array([ 0.39067804, 0.14011851, -0.06375249, 0.31984032, 1.55309358, 1.93199837]),
            # 'target_x_delta': np.array([-0.1, -0.1, -0.1, 0.0, 0.0, 0.0]),
            'target_x_delta': np.array([0.0, -0.09, -0.09, 0.0, 0.0, 0.0]),
            'Kp': np.array([.15, .15, .12, .075, .05, .05, .05]),
            'Kd': np.array([.15, .15, .12, .075, .05, .05, .05])*10.0,
            'Kpx': np.array([.5, .5, .5, .5, .5, .5]),
            'noise_gain': 0.005*0.,
            't_contact_factor': 2,
            'joint_space_noise': 0.25,
}


# exp_params = exp_params_yumi
exp_params = exp_params_mjc

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

mjc_agent = AgentMuJoCo(agent_hyperparams)

for i in range(num_samples):
    ref_x_traj = []
    curr_x_traj = []
    ref_q_traj = []
    pol = Policy(agent_hyperparams, exp_params)
    mjc_agent.sample(
                    pol, cond,
                    verbose=True,
                    noisy=True
                )
    ref_x_traj, curr_x_traj, ref_q_traj = pol.get_traj_data()
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

plt.show()

Xs = np.array(Xs)
Us = np.array(Us)

dP = 7
dV = 7
dU = 7
N, T, dX = Xs.shape

assert(Xs.shape[2]==(dP+dV))
assert(dP==dV)

Qts = Xs[:, :, :dP]
Qts_d = Xs[:, :, dP:dP+dV]
Uts = Us
Ets = np.zeros((N, T, 6))
Ets_d = np.zeros((N, T, 6))
Fts = np.zeros((N, T, 6))
for n in range(N):
    for i in range(T):
        Ets[n, i] = yumiKin.fwd_pose(Qts[n,i])
        J_A = yumiKin.get_analytical_jacobian(Qts[n,i])
        Ets_d[n,i] = J_A.dot(Qts_d[n,i])
        Fts[n,i] = np.linalg.pinv(J_A.T).dot(Uts[n,i])

EXts = np.concatenate((Ets,Ets_d),axis=2)


exp_params = {}
exp_params['dP'] = 7
exp_params['dV'] = 7
exp_params['dU'] = 7
exp_params['dt'] = 0.05
exp_params['T'] = T
exp_params['num_samples'] = N



exp_data={}
exp_data['exp_params'] = exp_params
exp_data['X'] = Xs
exp_data['U'] = Us
exp_data['EX'] = EXts
exp_data['F'] = Fts


# raw_input()
pickle.dump( exp_data, open( "./Results/mjc_exp_2_sec_raw.p", "wb" ) )
