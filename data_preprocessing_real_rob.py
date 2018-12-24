import numpy as np
import matplotlib.pyplot as plt
# import logging
import imp
# import os
# import os.path
import sys
# import copy
# import pprint
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

sys.path.insert(0, '/home/shahbaz/Research/Software/Spyder_ws/gps/python')

from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS
from gps.proto.gps_pb2 import END_EFFECTOR_POINT_VELOCITIES
from gps.proto.gps_pb2 import JOINT_ANGLES
from gps.proto.gps_pb2 import JOINT_VELOCITIES
from gps.proto.gps_pb2 import ACTION

data_logger = DataLogger()

gps_dir = '/home/shahbaz/Research/Software/Spyder_ws/gps/'
exp_name = 'yumi_model_learning_1'
exp_dir = gps_dir + 'experiments/' + exp_name + '/'
hyperparams_file = exp_dir + 'hyperparams.py'
hyperparams = imp.load_source('hyperparams', hyperparams_file)
config = hyperparams.config
_data_files_dir = config['common']['data_files_dir']

f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')
euler_from_matrix = pydart.utils.transformations.euler_from_matrix
J_G_to_A = jacobian_geometric_to_analytic

#pykdl stuff
robot = Robot.from_xml_string(f.read())
base_link = robot.get_root()
# end_link = 'left_tool0'
end_link = 'left_contact_point'
kdl_kin = KDLKinematics(robot, base_link, end_link)

itr = 0
traj_sample_lists = data_logger.unpickle(_data_files_dir +
    ('traj_sample_itr_%02d.pkl' % itr))
traj_sample_list = traj_sample_lists[0] # cond 0
# sample_data = pickle.load( open( "mjc_blocks_raw_10_10.p", "rb" ) )
num_samples = len(traj_sample_list)
Ts = traj_sample_list[0].T
Xs = []
Us = []
for sample in traj_sample_list:
    jp_pt = sample.get(JOINT_ANGLES)
    jv_pt = sample.get(JOINT_VELOCITIES)
    Xs.append(np.concatenate((jp_pt,jv_pt),axis=1))
    jt_pt = sample.get(ACTION)
    Us.append(jt_pt)

Xs = np.array(Xs)
Us = np.array(Us)

dP = 7
dV = 7
dU = 7
N, T, dX = Xs.shape
assert(T==Ts)
train_data_id = num_samples/2
assert(Xs.shape[2]==(dP+dV))
assert(dP==dV)

XU = np.zeros((N,T,dX+dU))
for n in range(N):
    XU[n] = np.concatenate((Xs[n,:,:],Us[n,:,:]),axis=1)
XU_t = XU[:,:-1,:]
X_t1 = XU[:,1:,:dX]
X_t = XU[:,:-1,:dX]
delX = X_t1 - X_t
dynamics_data = np.concatenate((XU_t,X_t1),axis=2)
train_data = dynamics_data[0:train_data_id,:,:]
test_data = dynamics_data[train_data_id:,:,:]

train_data_flattened = train_data.reshape((-1,train_data.shape[-1]))
# X_train = train_data_flattened[:,0:dP+dV]
data_train = train_data_flattened

Qt = data_train[:,0:dP].reshape((-1,dP))
Qt_d = data_train[:,dP:dP+dV].reshape((-1,dV))
Ut = data_train[:,dP+dV:dP+dV+dU].reshape((-1,dU))
Qt_1 = data_train[:,dP+dV+dU:dP+dV+dU+dP].reshape((-1,dP))
Qt_1_d = data_train[:,dP+dV+dU+dP:dP+dV+dU+dP+dV].reshape((-1,dV))
Xt = np.concatenate((Qt,Qt_d),axis=1)
Xt_1 = np.concatenate((Qt_1,Qt_1_d),axis=1)
Et = np.zeros((Qt.shape[0],6))
Et_d = np.zeros((Qt.shape[0],6))
Et_1 = np.zeros((Qt.shape[0],6))
Et_1_d = np.zeros((Qt.shape[0],6))
Ft = np.zeros((Qt.shape[0],6))

for i in range(Qt.shape[0]):
    Tr = kdl_kin.forward(Qt[i], end_link=end_link, base_link=base_link)
    epos = np.array(Tr[:3,3])
    epos = epos.reshape(-1)
    erot = np.array(Tr[:3,:3])
    erot = euler_from_matrix(erot)
    Et[i] = np.append(epos,erot)
    Tr = kdl_kin.forward(Qt_1[i], end_link=end_link, base_link=base_link)
    epos = np.array(Tr[:3,3])
    epos = epos.reshape(-1)
    erot = np.array(Tr[:3,:3])
    erot = euler_from_matrix(erot)
    Et_1[i] = np.append(epos,erot)

    J_G = np.array(kdl_kin.jacobian(Qt[i]))
    J_G = J_G.reshape((6,7))
    J_A = J_G_to_A(J_G, Et[i][3:])
    Et_d[i] = J_A.dot(Qt_d[i])
    J_G = np.array(kdl_kin.jacobian(Qt_1[i]))
    J_G = J_G.reshape((6,7))
    J_A = J_G_to_A(J_G, Et_1[i][3:])
    Et_1_d[i] = J_A.dot(Qt_1_d[i])
    Ft[i] = np.linalg.pinv(J_A.T).dot(Ut[i])

EXt = np.concatenate((Et,Et_d),axis=1)
EXt_1 = np.concatenate((Et_1,Et_1_d),axis=1)

exp_params = {}
exp_params['dP'] = 7
exp_params['dV'] = 7
exp_params['dU'] = 7
exp_params['dt'] = 0.05
exp_params['T'] = T


sample_data={}
sample_data['exp_params'] = exp_params
sample_data['train_data'] = train_data
sample_data['test_data'] = test_data
sample_data['EXt'] = EXt
sample_data['EXt_1'] = EXt_1
sample_data['Ft'] = Ft
sample_data['Xt'] = Xt
sample_data['Xt_1'] = Xt_1
sample_data['Ut'] = Ut
pickle.dump( sample_data, open( "yumi_blocks_l1_processed_1.p", "wb" ) )
