import numpy as np
from numpy import random
from mixture_model_gibbs_sampling import *
import matplotlib.pyplot as plt
import copy
import pickle
# from utilities import *
# import PyKDL as kdl
# import pykdl_utils
# import hrl_geom.transformations as trans
# from hrl_geom.pose_converter import PoseConv
# from urdf_parser_py.urdf import Robot
# from pykdl_utils.kdl_kinematics import *
# import pydart2 as pydart

# np.random.seed(12345) # terminates with 3 clusters when starting with 1 and alpha 1
f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')
# euler_from_matrix = pydart.utils.transformations.euler_from_matrix
# J_G_to_A = jacobian_geometric_to_analytic

exp_params = {
                'T': 5., # length of time series in seconds, not used
                'max_itr': 1000,
                'init_count': 5,
                'DP_flag': True, # if False: do not initialize assignment with -1
                'rand_clust_size': False,
                'seq_init': False, # only for DP
                }
init_state = {
            'cluster_ids_': None, # from 0 to K-1
            'num_clusters_': 1, # K, use K=1 for sequencial initial assignment for DP
            # 'cluster_variance_': .01, # fixed variance of each Gaussian
            'hyperparameters_': {
                                'alpha_': .1, # DP or Dirichlet parameter
                                'k0': 0.01
                                # NIW hyperparams set according to KPM book, p133
                                # assuming the same prior for all clusters
                                },
            'sampling_scheme': 'crp',
            # 'sampling_scheme': 'k-means',
}

sample_data = pickle.load( open( "mjc_floor_contact_raw.p", "rb" ) )
# X = sample_data['X'] # N X T X dX
# U = sample_data['U'] # N X T X dU
E_train = sample_data['E_train']

mjc_exp_params = sample_data['exp_params']
# dP = mjc_exp_params['dP']
# dV = mjc_exp_params['dV']
# dU = mjc_exp_params['dU']
# down samples in line train_data = train_data[::2]
# num_samples = mjc_exp_params['num_samples']

params = {}
params['exp_params'] = exp_params
params['init_state'] = init_state
params['mjc_exp_params'] = mjc_exp_params

# N, T, dX = X.shape
# train_data_id = num_samples/2
# assert(X.shape[2]==(dP+dV))
# assert(dP==dV)
#
# # P = X[:,:,0:dP].reshape((N,T,dP))
# # V = X[:,:,dP:dP+dV].reshape((N,T,dV))
# #
# XU = np.zeros((N,T,dX+dU))
# for n in range(N):
#     XU[n] = np.concatenate((X[n,:,:],U[n,:,:]),axis=1)
# XU_t = XU[:,:-1,:]
# X_t1 = XU[:,1:,:dX]
# X_t = XU[:,:-1,:dX]
# delX = X_t1 - X_t
# dynamics_data = np.concatenate((XU_t,X_t1),axis=2)
# train_data = dynamics_data[0:train_data_id,:,:]
# test_data = dynamics_data[train_data_id:,:,:]
#
# robot = Robot.from_xml_string(f.read())
# base_link = robot.get_root()
# end_link = 'left_tool0'
# kdl_kin = KDLKinematics(robot, base_link, end_link)
#
# train_data_flattened = train_data.reshape((-1,train_data.shape[-1]))
# X_train = train_data_flattened[:,0:dP+dV]
#
# Qt = X_train[:,0:dP].reshape((-1,dP))
# Qt_d = X_train[:,dP:dP+dV].reshape((-1,dV))
# Xt = np.concatenate((Qt,Qt_d),axis=1)
# Et = np.zeros((Qt.shape[0],6))
# Et_d = np.zeros((Qt.shape[0],6))
# for i in range(Qt.shape[0]):
#     Tr = kdl_kin.forward(Qt[i], end_link=end_link, base_link=base_link)
#     epos = np.array(Tr[:3,3])
#     epos = epos.reshape(-1)
#     erot = np.array(Tr[:3,:3])
#     erot = euler_from_matrix(erot)
#     Et[i] = np.append(epos,erot)
#
#     J_G = np.array(kdl_kin.jacobian(Qt[i]))
#     J_G = J_G.reshape((6,7))
#     J_A = J_G_to_A(J_G, Et[i][3:])
#     Et_d[i] = J_A.dot(Qt_d[i])

# data = X_t # state-space clustering
# data = np.concatenate((XU_t,delX),axis=2)
# train_data = data[0:train_data_id,:,:] # half-half training and test data, move this to exp_params
# train_data = train_data.reshape((-1,train_data.shape[-1]))
# train_data = train_data[::4]
# test_data = data[train_data_id:,:,:]
# test_data = test_data.reshape((-1,test_data.shape[-1]))
# test_data = test_data[::4]

# E_train = np.concatenate((Et,Et_d),axis=1)
# E_train = E_train[::2]
max_itr = exp_params["max_itr"]
DP_flag = exp_params["DP_flag"]
rand_clust_size = exp_params["rand_clust_size"]
seq_init = exp_params["seq_init"]

init_count = exp_params['init_count']
exp_stats = []
# run init_count number of Markov chains
for chain_num in range(init_count):
    print 'Chain', chain_num, 'starting...'
    chain_stat = run_Markov_chain(init_state, E_train, DP_flag,rand_clust_size,seq_init,max_itr)
    print 'Chain',chain_num, 'terminated...'
    exp_stats.append(chain_stat)

# pickle exp_stats
exp_result = {}
exp_result['exp_stats'] = exp_stats
# exp_result['train_data'] = train_data
# exp_result['test_data'] = test_data
# exp_result['X_train'] = X_train
# exp_result['E_train'] = E_train
exp_result['params'] = params
# pickle.dump( exp_result, open( "mjc_1d_finite_gauss_clustered.p", "wb" ) )
pickle.dump( exp_result, open( "floor_contact_DPM_clustered.p", "wb" ) )
