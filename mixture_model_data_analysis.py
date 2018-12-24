import numpy as np
import scipy.linalg
# from collections import namedtuple, Counter
# from scipy import stats
from numpy import random
from collapsed_Gibbs_sampler import *
import matplotlib.pyplot as plt
# from discont_function import DiscontinuousFunction
from utilities import plot_ellipse
from utilities import get_N_HexCol
from utilities import logsum
from utilities import get_N_HexCol
import copy
import pickle

def ACF(X,t):
    mu = np.mean(X)
    # mu = mu_theory
    N = X.shape[0]
    assert(N > t)
    num = 0
    den = 0
    for i in range(N-t):
        xd = X[i] - mu
        num += xd*(X[i+t] - mu)
        den += xd*xd
        if xd==0:
            acor = 0
        else:
            acor = num/den
    return acor

# # autocorrelation plot for cluster numbers
# exp_result = pickle.load( open( "1d_synthetic_clustering.p", "rb" ) )
# exp_stats = exp_result['exp_stats']
# for ic in range(len(exp_stats)):
#     kstat = np.array(exp_stats[ic]['k_stat'])
#     log_k = np.log(kstat)
#     # log_k = kstat
#     N = log_k.shape[0]
#     print N
#     for n in range(100,N+1,50):
#     # for n in range(N,N+1):
#         print n
#         X = log_k[:n]
#         auto_cor_plot = [np.array([t,ACF(X,t)]) for t in range(n/2)]
#         auto_cor_plot = np.array(auto_cor_plot)
#         plt.figure()
#         plt.plot(auto_cor_plot[:,0],auto_cor_plot[:,1])
#         plt.show()
#         # raw_input()

# # plot mujoco_1d_slide data
# sample_data = pickle.load( open( "mujoco_1d_block_1.p", "rb" ) )
# X = sample_data['X'] # N X T X dX
# U = sample_data['U'] # N X T X dU
#
# exp_params = sample_data['exp_params']
# dP = exp_params['dP']
# dV = exp_params['dV']
# dU = exp_params['dU']
# N, T, dX = X.shape
#
# assert(X.shape[2]==(dP+dV))
# assert(dP==dV)
#
# P = X[:,:,0:dP].reshape((N,T,dP))
# V = X[:,:,dP:dP+dV].reshape((N,T,dV))
#
# i = range(exp_params['T']-1)
# t = np.array(i)*exp_params['dt']
#
# pos_samples = P.reshape((N,T)).T
# vel_samples = V.reshape((N,T)).T
# act_samples = U.reshape((N,T)).T
#
# # plt.figure()
# # plt.plot(t,pos_samples)
# #
# # plt.figure()
# # plt.plot(t,vel_samples)
# #
# # plt.figure()
# # plt.plot(t,act_samples)
#
# XU = np.zeros((N,T,dX+dU))
# for n in range(N):
#     XU[n] = np.concatenate((X[n,:,:],U[n,:,:]),axis=1)
# XU_t = XU[:,:-1,:]
# X_t1 = XU[:,1:,:dX]
#
# XY = np.concatenate((XU_t,X_t1),axis=2)
# print XY.shape
# XY = XY.reshape((-1,XY.shape[-1]))
# print XY.shape
# T=T-1
# pos_samples = XY[0*T:1*T,0]
# vel_samples = XY[0*T:1*T,1]
# plt.figure()
# plt.plot(t,pos_samples)
# plt.plot(t,vel_samples)
# plt.show()

# plot mujoco_1d_block clustered data
exp_result = pickle.load( open( "1d_mujoco_sliding_clustering_1.p", "rb" ) )
XY = exp_result['data']
exp_stats = exp_result['exp_stats']
cids = exp_stats[-1]['cluster_ids']
assignment = exp_stats[-1]['assignment']
K = exp_stats[-1]['expert_count']

colors = get_N_HexCol(K)
colors = np.asarray(colors)
col = np.zeros([XY.shape[0],3])
idx = np.array(assignment)
i=0
for cid in cids:
    col[(idx==cid)] = colors[i]
    # print 'Cluster size:',cid,state['suffstats'][cid].N_k
    i += 1

pc = [exp_stats[-1]['theta'][cid].mean[0] for cid in cids]
pc = np.array(pc)

# XY = XY.reshape((5,-1,XY.shape[-1]))
p = XY[:,0]
v = XY[:,1]
u = XY[:,2]
delp = XY[:,3]
delv = XY[:,4]

p1 = delp
v1 = delv
# p1 = delp + p
# v1 = delv + v
plt.figure()
plt.subplot(231)
plt.xlabel('p(t)')
plt.ylabel('p(t+1)')
plt.scatter(p,p1,c=col/255.0)
plt.subplot(232)
plt.xlabel('v(t)')
plt.ylabel('p(t+1)')
plt.scatter(v,p1,c=col/255.0)
plt.subplot(233)
plt.xlabel('u(t)')
plt.ylabel('p(t+1)')
plt.scatter(u,p1,c=col/255.0)
plt.subplot(234)
plt.xlabel('p(t)')
plt.ylabel('v(t+1)')
plt.scatter(p,v1,c=col/255.0)
plt.subplot(235)
plt.xlabel('v(t)')
plt.ylabel('v(t+1)')
plt.scatter(v,v1,c=col/255.0)
plt.subplot(236)
plt.xlabel('u(t)')
plt.ylabel('v(t+1)')
plt.scatter(u,v1,c=col/255.0)
plt.show()
