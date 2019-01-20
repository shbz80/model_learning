import numpy as np
from numpy import random
from mixture_model_gibbs_sampling import *
import matplotlib.pyplot as plt
import copy
import pickle

# np.random.seed(12345) # terminates with 3 clusters when starting with 1 and alpha 1

exp_params = {
                'T': 5., # length of time series in seconds, not used
                'max_itr': 10000,
                'init_count': 3,
                'DP_flag': True, # if False: do not initialize assignment with -1
                'rand_clust_size': False,
                'seq_init': True, # only for DP
                }
init_state = {
            'cluster_ids_': None, # from 0 to K-1
            'num_clusters_': 1, # K, use K=1 for sequencial initial assignment for DP
            # 'cluster_variance_': .01, # fixed variance of each Gaussian
            'hyperparameters_': {
                                'alpha_': 4., # DP or Dirichlet parameter
                                'k0': 0.01
                                # NIW hyperparams set according to KPM book, p133
                                # assuming the same prior for all clusters
                                },
            'sampling_scheme': 'crp',
            # 'sampling_scheme': 'k-means',
}

sample_data = pickle.load( open( "mjc_1d_raw_5.p", "rb" ) )
X = sample_data['X'] # N X T X dX
U = sample_data['U'] # N X T X dU

mjc_exp_params = sample_data['exp_params']
dP = mjc_exp_params['dP']
dV = mjc_exp_params['dV']
dU = mjc_exp_params['dU']
mjc_exp_params['dt'] = mjc_exp_params['dt'] # down samples in line train_data = train_data[::2]
num_samples = mjc_exp_params['num_samples']

params = {}
params['exp_params'] = exp_params
params['init_state'] = init_state
params['mjc_exp_params'] = mjc_exp_params

N, T, dX = X.shape
train_data_id = num_samples/2
assert(X.shape[2]==(dP+dV))
assert(dP==dV)

P = X[:,:,0:dP].reshape((N,T,dP))
V = X[:,:,dP:dP+dV].reshape((N,T,dV))

XU = np.zeros((N,T,dX+dU))
for n in range(N):
    XU[n] = np.concatenate((X[n,:,:],U[n,:,:]),axis=1)
XU_t = XU[:,:-1,:]
X_t1 = XU[:,1:,:dX]
X_t = XU[:,:-1,:dX]
delX = X_t1 - X_t

data = np.concatenate((XU_t,X_t1),axis=2)
# data = np.concatenate((XU_t,delX),axis=2)
train_data = data[0:train_data_id,:,:] # half-half training and test data,
train_data = train_data.reshape((-1,train_data.shape[-1]))
train_data = train_data[::4]
test_data = data[train_data_id:,:,:]
test_data = test_data.reshape((-1,test_data.shape[-1]))
test_data = test_data[::4]

# prepare reduced data set for clustering
p_r = train_data[:,0].reshape(-1,1)
x1_r = train_data[:,4:]
train_data_r = np.concatenate((p_r,x1_r),axis=1)

max_itr = exp_params["max_itr"]
DP_flag = exp_params["DP_flag"]
rand_clust_size = exp_params["rand_clust_size"]
seq_init = exp_params["seq_init"]

init_count = exp_params['init_count']
exp_stats = []
# run init_count number of Markov chains
for chain_num in range(init_count):
    print 'Chain', chain_num, 'starting...'
    chain_stat = run_Markov_chain(init_state, train_data_r, DP_flag,rand_clust_size,seq_init,max_itr)
    print 'Chain',chain_num, 'terminated...'
    exp_stats.append(chain_stat)

# pickle exp_stats
exp_result = {}
exp_result['exp_stats'] = exp_stats
exp_result['train_data'] = train_data
exp_result['test_data'] = test_data
exp_result['params'] = params
# pickle.dump( exp_result, open( "mjc_1d_finite_gauss_clustered.p", "wb" ) )
pickle.dump( exp_result, open( "mjc_1d_infinite_gauss_clustered.p", "wb" ) )
