import sys
sys.path.insert(0, '/home/shahbaz/Research/Software/Spyder_ws/gps/python')
import numpy as np
import matplotlib.pyplot as plt
from discont_function import DiscontinuousFunction
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.gaussian_process import GaussianProcessRegressor
from copy import deepcopy

np.random.seed(5)
# np.random.seed(100)

################################################################################
# works with the old function that could not generate random segments
# discon_params = {
#             'dt': 0.03,
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
# gp_params = {
#                 'alpha': 0., # alpha=0 when using white kernal
#                 'K_C': C(1.0, (1e-2, 1e2)),
#                 'K_RBF': RBF(1, (1e-3, 1e3)),
#                 # 'K_RBF': RBF(1, (1e-3, 1e0)),
#                 'K_W': W(noise_level=1., noise_level_bounds=(1e-1, 1e1)),
#                 # 'K_W': W(noise_level=1., noise_level_bounds=(1e0, 5e0)),
#                 'normalize_y': True,
#                 'restarts': 10,
#                 }
###############################################################################
# works with the new function that can generate random segments
discon_params = {
            'dt': 0.01,
            'Nsam': 1,
            'Nsec': 4, # works well with random seed 5 and Nsec 5
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
            'offset': 7.5,
            }
gp_params = {
                'alpha': 0., # alpha=0 when using white kernal
                'K_C': C(1.0, (1e-2, 1e2)),
                'K_RBF': RBF(1, (1e-3, 1e3)),
                # 'K_RBF': RBF(1, (1e-3, 1e0)),
                'K_W': W(noise_level=1., noise_level_bounds=(1e-1, 1e1)),
                # 'K_W': W(noise_level=1., noise_level_bounds=(1e0, 5e0)),
                'normalize_y': True,
                'restarts': 10,
                }
gmm_params = {
                'K': 4, # cluster size
                'restarts': 10, # number of restarts
                }
exp_params = {
                'T': 10. # length of time series in seconds
                }


dt = discon_params['dt']
Nsec = discon_params['Nsec']
T = Nsec * 1. # assumimg each segment is 1 sec
disc_flag = discon_params['disc_flag']
Nsam = discon_params['Nsam']
# T = exp_params['T']
# N = int(T/dt) # number of time steps
K = gmm_params['K']
gmm_restarts = gmm_params['restarts']

discFunc = DiscontinuousFunction(discon_params)

# xr, yr = discFunc.genRealFunc(T,disc_flag=disc_flag,plot=False)
#
# X, Y = discFunc.genNsamplesFunc(T,disc_flag=disc_flag,plot=True)
sec_list = ['flat','lin','quad','sin']
xr,yr,X,Y = discFunc.genNsamplesNew(sec_list=sec_list,plot=True)
_,_,X_test,Y_test = discFunc.genNsamplesNew(sec_list=sec_list,plot=False)
X_test = X_test.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)

N = X.shape[0]

w,wn,mu,sigma,idx,mass = discFunc.clusterGmmFunc(X,Y,xr,yr,T,K,gmm_restarts,plot=True)

x = np.reshape(X,(N*Nsam,1))
y = np.reshape(Y,(N*Nsam,1))
gp_theta, gp_score, gp_hyperparams = discFunc.gpFitFunc(x,y,xr,yr,gp_params)
global_gp_score = discFunc.gpScoreFunc(X_test,Y_test)
print 'Global GP score:', global_gp_score
y_mean, y_std = discFunc.gpPredictFunc(T,xr,yr,plot=True)

# mixture of GP experts - independent gating and expert learning
idx = np.argmax(w,axis=1)
MGP = []
for k in range(K):
    xk = x[(idx==k)]
    yk = y[(idx==k)]
    alpha = gp_params['alpha']
    K_C = gp_params['K_C']
    K_RBF = gp_params['K_RBF']
    K_W = gp_params['K_W']
    normalize_y = gp_params['normalize_y']
    restarts = gp_params['restarts']
    # kernel = K_C + K_RBF + K_W
    kernel = K_RBF + K_W
    # gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(xk, yk)
    MGP.append(deepcopy(gp))
    del gp


# xn,yn_xn,clsidx = discFunc.gmmPredictFunc(T,MGP) # prepare grid and predict clusters from gmm
xn,yn_xn,clsidx = discFunc.gmmPredictFunc(T) # prepare grid and predict clusters from gmm
xn_test,_,clsidx_test = discFunc.gmmPredictFunc(T,x_test=X_test) # prepare grid and predict clusters from gmm
# print xn.shape, yn_xn.shape,clsidx.shape

# clsidx_gp = np.zeros(N)
# yn_mean_gp = np.zeros(N)
# yn_std_gp = np.zeros(N)
# for n in range(N):
#     x = xn[n]
#     ys = []
#     log_probs=[]
#     for k in range(K):
#         gp = MGP[k]
#         y_m,y_cov  = gp.predict(x.reshape(1,1),return_cov=True)
#         ys.append([y_m,np.sqrt(y_cov)])
#         log_prob = stats.norm.logpdf(y_m, y_m, np.sqrt(y_cov))+np.log(mass[k])
#         log_probs.append(log_prob)
#     log_probs = np.array(log_probs)
#     i = int(np.argmax(log_probs))
#     clsidx_gp[n] = i
#     yn_mean_gp[n] = ys[i][0]
#     yn_std_gp[n] = ys[i][1]
#     del log_probs, ys
#
# fig, ax = plt.subplots()
# xt = xn.reshape(N)
# plt.plot(xt,yn_mean_gp,color='b')        # predictions from mixture model
# plt.fill_between(xt, yn_mean_gp - yn_std_gp, yn_mean_gp + yn_std_gp, alpha=0.2, color='b')
# # plt.plot(xn,y_mean)       # predictions from GP
# plt.plot(xr,yr,color='r')   # Ground truth function

yn_mean = np.zeros(N)
y_pred = np.zeros(N)
yn_cov = np.zeros(N)
for n in range(N):
    z = clsidx[n]
    z_test = clsidx_test[n]
    yn_mean[n], yn_cov[n] = MGP[z].predict(xn[n].reshape(1,1),return_cov=True)
    y_pred[n] = MGP[z_test].predict(xn_test[n].reshape(1,1))
assert(yn_cov.shape==(N,)) # make sure it is single point gp prediction
yn_std = np.sqrt(yn_cov)
moe_gp_score = np.linalg.norm(y_pred - Y_test)
print 'MOE score:', moe_gp_score
# fig, ax = plt.subplots()
# # plt.plot(xn,yn_mean)            # predictions from mixture model
# # plt.plot(xn,y_mean)             # predictions from GP
# plt.plot(xn,yn_xn.reshape((N,1))) # conditional mean of GMM
# plt.plot(xr,yr,color='r')         # Ground truth function
# plt.scatter(mu[:,0],mu[:,1])      # Cluster means
# plt.title('Best conditional mean')
# colors = get_N_HexCol(K)
# colors = np.asarray(colors)
# for k in range(K):
#     plot_ellipse(ax, mu[k], sigma[k], color=colors[k]/255.0) # cluster ellipses

fig, ax = plt.subplots()
xt = xn.reshape(N)
plt.plot(xt,yn_mean,color='b')        # predictions from mixture model
plt.fill_between(xt, yn_mean - yn_std, yn_mean + yn_std, alpha=0.2, color='b')
# plt.plot(xn,y_mean)       # predictions from GP
plt.plot(xr,yr,color='r')   # Ground truth function
# colors = get_N_HexCol(K)
# colors = np.asarray(colors)
# for k in range(K):
#     plot_ellipse(ax, mu[k], sigma[k], color=colors[k]/255.0)

plt.show()
plt.close('all')
