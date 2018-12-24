import pandas as pd
import numpy as np
# from collections import namedtuple, Counter
# from scipy import stats
from numpy import random
from collapsed_Gibbs_sampler_1 import *
import matplotlib.pyplot as plt
from discont_function import DiscontinuousFunction

from utilities import get_N_HexCol

np.random.seed(12345) # terminates with 3 clusters when starting with 1 and alpha 1

exp_params = {
                'T': 5., # length of time series in seconds
                'stop_cond': 10,
                'max_itr': 200,
                'min_itr': 50,
                'DP_flag': False,
                }
init_state = {
            'cluster_ids_': None, # from 0 to K-1
            'num_clusters_': 3, # K
            'hyperparameters_': {
                                'alpha_': 1, # DP or Dirichlet parameter
                                'k0': 0.01
                                # NIW hyperparams set according to KPM book, p133
                                # assuming the same prior for all clusters
            }
}


''' Notes
actual number of clusters 3
convergence criterion: number of cluster for 10 consequtive Gibbs step remains unchanged
with init cluter count 2 and alpha 1 it never really converged, but almost to 4
with alpha 0.1 it converged much earlier to 3 in 36 iterations
with init cluster count 10 and alpha 0.1 it converged to 4 in 67 iterations
'''
data = pd.Series.from_csv("clusters.csv")

state = initial_state(init_state, data)
plt.figure()
plot_clusters(state)
nc = state["num_clusters_"]
stop_cond = exp_params["stop_cond"]
max_itr = exp_params["max_itr"]
min_itr = exp_params["min_itr"]
DP_flag = exp_params["DP_flag"]
rcnt = 0
for i in range(max_itr):
    if DP_flag:
        gibbs_step_dp(state)
    else:
        gibbs_step(state)
    print 'Itr:',i,'Number of clusters:', state["num_clusters_"]
    pnc = nc
    nc = state["num_clusters_"]
    if nc == pnc:
        rcnt += 1
    else:
        rcnt = 0
    if ((rcnt >= stop_cond)and(i>min_itr)): break
print 'Itr:',i,'Converged cluster count:',state["num_clusters_"]

plt.figure()
plot_clusters(state)
plt.show()
