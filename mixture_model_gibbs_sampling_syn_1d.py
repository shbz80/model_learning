import numpy as np
from numpy import random
from mixture_model_gibbs_sampling import *
import matplotlib.pyplot as plt
from discont_function import DiscontinuousFunction
import copy
import pickle

# np.random.seed(12345) # terminates with 3 clusters when starting with 1 and alpha 1
# np.random.seed(5) # works for 3 and 5 cluster data sets
np.random.seed(6) #

# discon_params = {
#             'dt': 0.05,
#             'Nsam': 5,
#             'Nsec': 4,
#             'noise_gain': 3.,
#             # 'noise_gain': 0,
#             'disc_flag': True,
#             'lin_m': -5.,
#             'lin_o': 10.,
#             'quad_o': -10.,
#             # 'quad_o': -15.,
#             # 'quad_a': 10.,
#             'quad_a': 15.,
#             'sin_o': -10.,
#             'sin_a': 5.,
#             }
discon_params = {
            # 'dt': 0.00625, # for 2 clusters
            # 'dt': 0.015625, # for 5 clusters
            'dt': 0.025, # for 8 clusters
            'Nsam': 1,
            'Nsec': 3, # number of clusters in the dataset
            'noise_gain': 2.,
            # 'noise_gain': 0,
            'disc_flag': True,
            'lin_m': -7.,
            'lin_o': 10.,
            'quad_o': -10.,
            # 'quad_o': -15.,
            # 'quad_a': 10.,
            'quad_a': 10.,
            'sin_o': -10.,
            'sin_a': 3.,
            'offset': 10.,
            }
exp_params = {
                'T': 5., # length of time series in seconds, not used anymore
                'max_itr': 1000,
                'init_count': 1,
                'DP_flag': False, # if False: do not initialize assignment with -1
                'rand_clust_size': False,
                'seq_init': False, # only for DP
                }
init_state = {
            'cluster_ids_': None, # from 0 to K-1
            'num_clusters_': 3, # K, use K=1 for sequencial initial assignment for DP
            # 'cluster_variance_': .01, # fixed variance of each Gaussian
            'hyperparameters_': {
                                'alpha_': 1., # DP or Dirichlet parameter
                                'k0': 0.01,
                                # NIW hyperparams set according to KPM book, p133
                                # assuming the same prior for all clusters
                                },
            'sampling_scheme': 'crp',
            # 'sampling_scheme': 'k-means',
}

params = {}
params['exp_params'] = exp_params
params['discon_params'] = discon_params
params['init_state'] = init_state

dt = discon_params['dt']
disc_flag = discon_params['disc_flag']
Nsam = discon_params['Nsam']
Nsec = discon_params['Nsec']
# T = exp_params['T']
T = Nsec * 1. # assumimg each segment is 1 sec
params['exp_params']['T'] = T
# N = int(T/dt) # number of time steps
discFunc = DiscontinuousFunction(discon_params)
xr,yr,X,Y = discFunc.genNsamplesNew(plot=True)
plt.show()
N = X.shape[0]
data = np.c_[X,Y]
# data = np.reshape(data,[N*Nsam,data.shape[2]])

max_itr = exp_params["max_itr"]
DP_flag = exp_params["DP_flag"]
rand_clust_size = exp_params["rand_clust_size"]
seq_init = exp_params["seq_init"]

init_count = exp_params['init_count']
exp_stats = []
# run init_count number of Markov chains
for chain_num in range(init_count):
    print 'Chain', chain_num, 'starting...'
    chain_stat = run_Markov_chain(init_state, data, DP_flag,rand_clust_size,seq_init,max_itr)
    print 'Chain',chain_num, 'terminated...'
    exp_stats.append(chain_stat)

# pickle exp_stats
exp_result = {}
exp_result['exp_stats'] = exp_stats
exp_result['data'] = data
exp_result['xr'] = xr
exp_result['yr'] = yr
exp_result['params'] = params
pickle.dump( exp_result, open( "syn_1d_finite_gauss_clustered.p", "wb" ) )
# pickle.dump( exp_result, open( "syn_1d_infinite_gauss_clustered.p", "wb" ) )
