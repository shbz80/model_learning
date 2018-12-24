import numpy as np
from numpy import random
from mixture_model_gibbs_sampling import *
import matplotlib.pyplot as plt
import copy
import pickle

# np.random.seed(12345) # terminates with 3 clusters when starting with 1 and alpha 1
f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')

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

sample_data = pickle.load( open( "mjc_floor_contact_processed.p", "rb" ) )
clust_data = sample_data['clust_data']

mjc_exp_params = sample_data['exp_params']

params = {}
params['exp_params'] = exp_params
params['init_state'] = init_state
params['mjc_exp_params'] = mjc_exp_params

max_itr = exp_params["max_itr"]
DP_flag = exp_params["DP_flag"]
rand_clust_size = exp_params["rand_clust_size"]
seq_init = exp_params["seq_init"]

init_count = exp_params['init_count']
exp_stats = []
# run init_count number of Markov chains
for chain_num in range(init_count):
    print 'Chain', chain_num, 'starting...'
    chain_stat = run_Markov_chain(init_state, clust_data, DP_flag,rand_clust_size,seq_init,max_itr)
    print 'Chain',chain_num, 'terminated...'
    exp_stats.append(chain_stat)

# pickle exp_stats
exp_result = {}
exp_result['exp_stats'] = exp_stats
exp_result['params'] = params
exp_result['train_data'] = sample_data['train_data']
exp_result['test_data'] = sample_data['test_data']
exp_result['clust_data'] = sample_data['clust_data']
# pickle.dump( exp_result, open( "mjc_1d_finite_gauss_clustered.p", "wb" ) )
pickle.dump( exp_result, open( "floor_contact_DPM_clustered.p", "wb" ) )
