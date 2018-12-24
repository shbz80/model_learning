import pandas as pd
import numpy as np
# from collections import namedtuple, Counter
# from scipy import stats
from numpy import random
from collapsed_Gibbs_sampler_1 import *
import matplotlib.pyplot as plt
from discont_function import DiscontinuousFunction

np.random.seed(12345) # terminates with 3 clusters when starting with 1 and alpha 1

discon_params = {
            'dt': 0.03,
            'Nsam': 5,
            'Nsec': 4,
            'noise_gain': 3.,
            # 'noise_gain': 0,
            'disc_flag': True,
            'lin_m': -5.,
            'lin_o': 10.,
            'quad_o': -10.,
            # 'quad_o': -15.,
            # 'quad_a': 10.,
            'quad_a': 15.,
            'sin_o': -10.,
            'sin_a': 5.,
            }
exp_params = {
                'T': 5. # length of time series in seconds
                }
init_state = {
            'cluster_ids_': None, # from 0 to K-1
            'num_clusters_': 1, # K
            # 'cluster_variance_': .01, # fixed variance of each Gaussian
            'hyperparameters_': {
                                'alpha_': 1, # DP or Dirichlet parameter
                                'k0': 0.01
                                # NIW hyperparams set according to KPM book, p133
                                # assuming the same prior for all clusters
            }
}
stop_cond = 10

'''
actual number of clusters 3
convergence criterion: number of cluster for 10 consequtive Gibbs step remains unchanged
with init cluter count 2 and alpha 1 it never really converged, but almost to 4
with alpha 0.1 it converged much earlier to 3 in 36 iterations
with init cluster count 10 and alpha 0.1 it converged to 4 in 67 iterations
'''
data = pd.Series.from_csv("clusters.csv")

dt = discon_params['dt']
disc_flag = discon_params['disc_flag']
Nsam = discon_params['Nsam']
T = exp_params['T']
N = int(T/dt) # number of time steps
discFunc = DiscontinuousFunction(discon_params)
X, Y = discFunc.genNsamplesFunc(T,disc_flag=disc_flag,plot=False)
# data = np.c_[X,Y]
# data = np.reshape(data,[N*Nsam,data.shape[2]])

state = initial_state(init_state, data)
# print state
# raw_input("Press Enter to continue ...")
plt.figure()
plot_clusters(state)
nc = state["num_clusters_"]
rcnt = 0
max_itr = 200
for i in range(max_itr):
    gibbs_step_dp(state)
    # gibbs_step(state)
    print 'Itr:',i,'Number of clusters:', state["num_clusters_"]
    pnc = nc
    nc = state["num_clusters_"]
    if nc == pnc:
        rcnt += 1
    else:
        rcnt = 0
    print 'Repetition counter:',rcnt
    if rcnt == stop_cond: break
print 'Itr:',i,'Converged cluster count:',state["num_clusters_"]
plt.figure()
plot_clusters(state)
plt.show()
