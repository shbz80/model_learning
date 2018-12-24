# import sys
# sys.path.insert(0, '/home/shahbaz/Research/Software/Spyder_ws/gps/python')
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scalar_dynamics_sys import sim_1d
from scalar_dynamics_sys import MomentMatching
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from copy import deepcopy
from utilities import get_N_HexCol
# from utilities import plot_ellipse
# from utilities import logsum
from sklearn import mixture
from collections import Counter
import operator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time


# np.random.seed(1) # 2 extra clusters
np.random.seed(3) # no extra clusters
# np.random.seed(4) # 1 extra clusters

# sim_1d_params = {
#     'x0': 0.,
#     'xT': 5.,
#     'a1': 0.95,
#     'a2': 0.5,
#     'b1': 0.5,
#     'b2': 1.,
#     'L1': -.2,
#     'L2': -.5,
#     'L3': -.1,
#     'dt': 0.05,
#     'T': 1.,
#     'xt1': 2.5,
#     'xt2': 10.,
#     'xt3': 5.,
#     'w_sigma_1': 0.1, # std dev
#     'w_sigma_2': 1., # std dev
#     'w_sigma_3': .5, # std dev
#     'init_x_var': 0.1,
#     'num_episodes': 20,
#     'type': 'disc',
#     # 'type': 'cont',
# }
sim_1d_params = {
    'x0': 0.,
    'xT': 5.,
    'a1': 0.95,
    'a2': 0.5,
    'b1': 0.5,
    'b2': 1.,
    'L1': -.1,
    'L2': -.1,
    'L3': -.1,
    'dt': 0.05,
    'T': 1.,
    'xt1': 2.5,
    'xt2': 7.5,
    'xt3': 5.,
    'w_sigma_1': 0.1, # std dev
    'w_sigma_2': .5, # std dev
    'w_sigma_3': 1., # std dev
    'init_x_var': 0.0,
    'num_episodes': 20,
    'type': 'disc',
    # 'type': 'cont',
}

cluster = True

fit_moe = True

# generate 1D continuous data
sim_1d_sys = sim_1d(sim_1d_params)
traj_gt = sim_1d_sys.sim_episode(noise=False)
num_episodes = sim_1d_params['num_episodes']
traj_list = sim_1d_sys.sim_episodes(num_episodes)
traj_data = np.array(traj_list)
# plot 1D continuous data
plt.figure()
plt.title('1D continuous system data')
for i in range(num_episodes):
    plt.subplot(311)
    plt.plot(traj_list[i][:,0],traj_list[i][:,1]) # x
    plt.subplot(312)
    plt.plot(traj_list[i][:,0],traj_list[i][:,2]) # un
    plt.subplot(313)
    plt.plot(traj_list[i][:,0],traj_list[i][:,3]) # u
plt.subplot(311)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.plot(traj_gt[:,0],traj_gt[:,1],color='k')
plt.subplot(312)
plt.xlabel('t')
plt.ylabel('un(t)')
plt.plot(traj_gt[:,0],traj_gt[:,3],color='k')
plt.subplot(313)
plt.xlabel('t')
plt.ylabel('u(t)')
plt.plot(traj_gt[:,0],traj_gt[:,3],color='k')
# plt.show()

# prepare data for GP training
dX = dU = 1
X = traj_data[:,:,1]
U = traj_data[:,:,2:4]
Wu = traj_data[:,:,4]
N, T = X.shape
X = X.reshape(N,T,dX)
U = U.reshape(N,T,dU+1)
XU = np.zeros((N,T,dX+dU+1))
XU = np.concatenate((X,U),axis=2)
XU_t = XU[:,:-1,:]
X_t1 = XU[:,1:,:dX]
X_t = XU[:,:-1,:dX]
delX = X_t1 - X_t

data = np.concatenate((XU_t,X_t1),axis=2)
# data = np.concatenate((XU_t,delX),axis=2)

Nsam = sim_1d_params['num_episodes']
N_train = 2 * Nsam//3

data_train_s = data[:N_train,:,:]
Xs_train = data_train_s[:,:,:dX]
X_train = Xs_train.reshape((-1,Xs_train.shape[-1]))
Uns_train = data_train_s[:,:,dX:dX+dU]
Us_train = data_train_s[:,:,dX+dU:dX+dU+1]
Ys_train = data_train_s[:,:,dX+dU+1:dX+dU+1+dX]
XUns_train = np.concatenate((Xs_train,Uns_train),axis=2)
XUs_train = np.concatenate((Xs_train,Us_train),axis=2)
Y_train = Ys_train.reshape((-1,Ys_train.shape[-1]))
XUn_train = XUns_train.reshape((-1,XUns_train.shape[-1]))
XUnY_train = np.concatenate((XUn_train, Y_train), axis=1)
# np.random.shuffle(XUnY_train)
XUn_train = XUnY_train[:,0:2]
XUn_train = XUn_train.reshape(XUn_train.shape[0],-1)
Y_train = XUnY_train[:,2]
Y_train = Y_train.reshape(Y_train.shape[0],1)

data_test_s = data[N_train:,:,:]
Xs_test = data_test_s[:,:,:dX]
Uns_test = data_test_s[:,:,dX:dX+dU]
Us_test = data_test_s[:,:,dX+dU:dX+dU+1]
Ys_test = data_test_s[:,:,dX+dU+1:dX+dU+1+dX]
XUns_test = np.concatenate((Xs_test,Uns_test),axis=2)
XUs_test = np.concatenate((Xs_test,Us_test),axis=2)
Y_test = Ys_test.reshape((-1,Ys_test.shape[-1]))
XUn_test = XUns_test.reshape((-1,XUns_test.shape[-1]))


# Train global GP
gpr_params = {
                'alpha': 0., # alpha=0 when using white kernal
                'kernel': C(1.0, (1e-2,1e2))*RBF(np.ones(dX+dU), (1e-3, 1e3))+W(noise_level=1., noise_level_bounds=(1e-2, 1e2)),
                'n_restarts_optimizer': 10,
                'normalize_y': False, # is not supported in the propogation function
                }
gp = GaussianProcessRegressor(**gpr_params)
start_time = time.time()
gp.fit(XUn_train, Y_train)
print 'Global GP fit time', time.time()-start_time

# plot 1D continuous dynamics
Xg = np.arange(-10, 10, 0.2)
Ug = np.arange(-10, 10, 0.2)
Xp, Up = np.meshgrid(Xg, Ug)
Yp = np.zeros((len(Ug),len(Xg)))
Yp_sigma_top = np.zeros((len(Ug),len(Xg)))
Yp_sigma_bottom = np.zeros((len(Ug),len(Xg)))
start_time = time.time()
for i in range(len(Xg)):
    XUp = np.concatenate((Xp[:,i].reshape(-1,1),Up[:,i].reshape(-1,1)),axis=1)
    Yt, Yt_sigma = gp.predict(XUp,return_std=True)
    Yp[:,i] = Yt.reshape(-1)
    Yp_sigma_top[:,i] = Yp[:,i] + Yt_sigma.reshape(-1)*1.96
    Yp_sigma_bottom[:,i] = Yp[:,i] - Yt_sigma.reshape(-1)*1.96
print 'Global GP prediction time', time.time()-start_time

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('1D continuous system dynamcis model')
# Plot the surface.
ax.plot_surface(Xp, Up, Yp, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.plot_wireframe(Xp, Up, Yp)
ax.plot_surface(Xp, Up, Yp_sigma_top, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.2)
ax.plot_surface(Xp, Up, Yp_sigma_bottom, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.2)
for i in range(XUns_train.shape[0]):
    ax.plot(XUns_train[i,:,0], XUns_train[i,:,1], Ys_train[i,:,0])
ax.set_xlabel('x(t)')
ax.set_ylabel('u(t)')
ax.set_zlabel('x(t+1)')
# plt.show()

# long-term prediction
XU_0 = XUs_train[0,:,0:2]
T = XU_0.shape[0]
# T = 10
X_pred = np.zeros(T)
x_0 = np.asscalar(XU_0[0,0])
u_0 = np.asscalar(XU_0[0,1])

# gp mean long-term prediction without uncertainty
# X_pred[0] = x_0
# xu_t = np.append(x_0,u_0)
# for t in range(1,T):
#     x_t1, _ = gp.predict(xu_t.reshape(1, -1), return_cov=True)
#     u_t1 = XU_0[t][1]
#     xu_t = np.append(np.asscalar(x_t1), u_t1)
#     X_pred[t] = np.asscalar(x_t1)
#
# dt = sim_1d_params['dt']
# time = np.array(range(T))*dt
#
# plt.figure()
# plt.plot(time, X_pred, label='x_pred')
# plt.plot(time, XU_0[:T,0], label='x_data')
# plt.plot(time, XU_0[:T,1], label='u_data')
# plt.legend()
# # plt.show()

gp_mm = MomentMatching(gp)

predition_params = {
                    'horizon': T,
                    'initial_state_var': 0.,
}

H = predition_params['horizon']
v0 = sim_1d_params['init_x_var']
# v0 = predition_params['initial_state_var']
w0 = np.asscalar(Wu[0,0])

# # State evolution (training data) with uncertainty propagation
# mu_X_pred = np.zeros(H)
# sigma_X_pred = np.zeros(H)
#
# XU_0 = XUs_train[0,:,0:2]
# x_0 = np.asscalar(XU_0[0,0])
# u_0 = np.asscalar(XU_0[0,1])
# xu_0 = np.append(x_0,u_0)
# mu_X_pred[0], sigma_X_pred[0] = x_0, v0
#
# mu_xu_t = np.append(x_0,u_0)
# sigma_xu_t = np.array([[v0, 0.],
#                         [0., w0]])
# start_time = time.time()
# for t in range(1,H):
#     mu_x_t1, sigma_x_t1 = gp_mm.predict_dynamics_1_step(mu_xu_t, sigma_xu_t)
#     u_t = np.asscalar(XU_0[t, 1])
#     mu_xu_t = np.append(mu_x_t1, u_t)
#     wu = np.asscalar(Wu[0,t])
#     sigma_xu_t = np.array([[sigma_x_t1, 0.],
#                           [0., wu]])
#     mu_X_pred[t] = mu_x_t1
#     sigma_X_pred[t] = sigma_x_t1
# print 'Prediction time for horizon', H,':', time.time()-start_time
#
# dt = sim_1d_params['dt']
# tm = np.array(range(H))*dt
#
# plt.figure()
# plt.title('State evolution (training data) with uncertainty propagation (moment matching)')
# plt.xlabel('t')
# plt.ylabel('x(t)')
# for XUn in XUns_train:
#     plt.plot(tm, XUn[:,0])
# plt.plot(tm, mu_X_pred, color='b', ls='-', marker='s', linewidth='2', label='learned model', markersize=7)
# plt.plot(tm, traj_gt[:-1,1], color='g', ls='-', marker='^', linewidth='2', label='real system', markersize=7)
# plt.fill_between(tm, mu_X_pred - np.sqrt(sigma_X_pred)*1.96, mu_X_pred + np.sqrt(sigma_X_pred)*1.96, alpha=0.2)
# plt.legend()
# # compute prediction score
# start_index = 0
# horizon = T # cannot be > T
# end_index = start_index + horizon
# weight = np.ones(horizon) # weight long term prediction mse error based on time
# score_cum = 0.
# for XUn in XUns_train:
#     x_m = XUn[start_index:end_index,0]
#     x_m.reshape(-1)
#     bias_term = (x_m - mu_X_pred[start_index:end_index])**2 # assumes mu_X_pred is computed for T
#     var_term = sigma_X_pred[start_index:end_index]
#     mse_ = bias_term + var_term
#     mse_w = mse_*weight
#     score_cum += np.sum(mse_w)
#
# print 'Continuous system prediction train data score:', score_cum/float(XUns_train.shape[0])

# State evolution (test data) with uncertainty propagation
mu_X_pred = np.zeros(H)
sigma_X_pred = np.zeros(H)

XU_0 = XUs_test[0,:,0:2]
x_0 = np.asscalar(XU_0[0,0])
u_0 = np.asscalar(XU_0[0,1])
xu_0 = np.append(x_0,u_0)
mu_X_pred[0], sigma_X_pred[0] = x_0, v0

mu_xu_t = np.append(x_0,u_0)
sigma_xu_t = np.array([[v0, 0.],
                        [0., w0]])

start_time = time.time()
for t in range(1,H):
    mu_x_t1, sigma_x_t1 = gp_mm.predict_dynamics_1_step(mu_xu_t, sigma_xu_t)
    u_t = np.asscalar(XU_0[t, 1])
    mu_xu_t = np.append(mu_x_t1, u_t)
    wu = np.asscalar(Wu[0, t])
    sigma_xu_t = np.array([[sigma_x_t1, 0.],
                          [0., wu]])
    mu_X_pred[t] = mu_x_t1
    sigma_X_pred[t] = sigma_x_t1
print 'Prediction time for horizon', H,':', time.time()-start_time
dt = sim_1d_params['dt']
tm = np.array(range(H))*dt

plt.figure()
plt.title('State evolution (testing data) with uncertainty propagation (moment matching)')
plt.xlabel('t')
plt.ylabel('x(t)')
for XUn in XUns_test:
    plt.plot(tm, XUn[:,0])
plt.plot(tm, mu_X_pred, color='b', ls='-', marker='s', linewidth='2', label='learned model', markersize=7)
plt.plot(tm, traj_gt[:-1,1], color='g', ls='-', marker='^', linewidth='2', label='real system', markersize=7)
plt.fill_between(tm, mu_X_pred - np.sqrt(sigma_X_pred)*1.96, mu_X_pred + np.sqrt(sigma_X_pred)*1.96, alpha=0.2)
plt.legend()
# compute prediction score
start_index = 0
horizon = T # cannot be > T
end_index = start_index + horizon
weight = np.ones(horizon) # weight long term prediction mse error based on time
score_cum = 0.
for XUn in XUns_test:
    x_m = XUn[start_index:end_index,0]
    x_m.reshape(-1)
    bias_term = (x_m - mu_X_pred[start_index:end_index])**2 # assumes mu_X_pred is computed for T
    var_term = sigma_X_pred[start_index:end_index]
    mse_ = bias_term + var_term
    mse_w = mse_*weight
    score_cum += np.sum(mse_w)
print 'Continuous system prediction test data score:', score_cum/float(XUns_test.shape[0])

if cluster:
    dpgmm_params = {
                    'K': 100, # cluster size
                    'restarts': 20, # number of restarts
                    # 'alpha': 1e-1, *
                    'alpha': 1e0,
                    'v0': 1+2,
                    'enable': True,
                    'dpgmm_inference_enable': True,
                    'gmm_inference_enable': False,
                    }
    dpgmm = mixture.BayesianGaussianMixture(n_components=dpgmm_params['K'],
                                                covariance_type='full',
                                                tol=1e-6,
                                                n_init=dpgmm_params['restarts'],
                                                max_iter=1000,
                                                weight_concentration_prior_type='dirichlet_process',
                                                weight_concentration_prior=dpgmm_params['alpha'],
                                                mean_precision_prior=None,
                                                mean_prior=None, # None = x_bar
                                                degrees_of_freedom_prior=dpgmm_params['v0'],
                                                covariance_prior=None,
                                                warm_start=False,
                                                init_params='random'
                                                )
    XY_train = np.concatenate((X_train, Y_train), axis=1)
    start_time = time.time()
    dpgmm.fit(XUnY_train)
    print 'DPGMM clustering time:', time.time() - start_time
    # dpgmm.fit(XY_train)
    print 'Converged DPGMM',dpgmm.converged_, 'on', dpgmm.n_iter_, 'iterations with lower bound',dpgmm.lower_bound_
    dpgmm_idx = dpgmm.predict(XUnY_train)
    # dpgmm_idx = dpgmm.predict(XY_train)
    K = dpgmm.weights_.shape[0]

    # plot cluster weights
    # get color array for data
    # colors = get_N_HexCol(K)
    # colors = np.asarray(colors)/255.
    # plt.figure()
    # plt.bar(range(K),dpgmm.weights_,color=colors)
    # plt.title('DPGMM cluster weights')

    # plot cluster distribution
    # get labels and counts
    labels, counts = zip(*sorted(Counter(dpgmm_idx).items(),key=operator.itemgetter(0)))
    K = len(labels)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    # colors_in_labels = colors[list(labels)]
    marker_set = ['.','o','*','+','^','x','o','D','s']
    marker_set_size = len(marker_set)
    if K < marker_set_size:
        markers = marker_set[:K]
    else:
        markers = ['o']*K
    plt.figure()
    plt.bar(labels,counts,color=colors)
    plt.title('DPGMM cluster dist')

    # plot clustered data
    col = np.zeros([XUnY_train.shape[0],3])
    mark = np.array(['None']*XUnY_train.shape[0])
    i=0
    for label in labels:
        col[(dpgmm_idx==label)] = colors[i]
        mark[(dpgmm_idx == label)] = markers[i]
        i+=1

    col = col.reshape(N_train,-1,3)
    mark = mark.reshape(N_train,-1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(XUns_train.shape[0]):
        for j in range(XUns_train.shape[1]):
            ax.scatter(XUns_train[i,j,0], XUns_train[i,j,1], Ys_train[i,j], c=col[i,j], marker=mark[i,j])
    ax.set_xlabel('x(t)')
    ax.set_ylabel('u(t)')
    ax.set_zlabel('x(t+1)')
    plt.title('DPGMM clustering')

    # plot clustered trajectory
    plt.figure()
    for i in range(XUns_train.shape[0]):
        for j in range(XUns_train.shape[1]):
            plt.scatter(tm[j],XUns_train[i,j,0], c=col[i,j], marker=mark[i,j])
    plt.title('Clustered trajectories')

if fit_moe and cluster:
    start_time = time.time()
    MoE = {}
    for label in labels:
        x_train = XUn_train[(dpgmm_idx == label)]
        y_train = Y_train[(dpgmm_idx == label)]
        gp_ = GaussianProcessRegressor(**gpr_params)
        gp_.fit(x_train, y_train)
        gp_expert = MomentMatching(gp_)
        MoE[label] = deepcopy(gp_expert)
        del gp_expert, gp_
    print 'MoE training time:', time.time() - start_time

    # gating network training
    svm_grid_params = {
                        'param_grid': {"C": np.logspace(-10, 10, endpoint=True, num=21, base=2.),
                                       "gamma": np.logspace(-10, 10, endpoint=True, num=21, base=2.)},
                        'scoring': 'accuracy',
                        'cv': 5,
                        'n_jobs':-1,
                        'iid': False,
    }
    svm_params = {

        'kernel': 'rbf',
        'decision_function_shape': 'ovr',
        'tol': 1e-06,
    }

    scaler = StandardScaler().fit(XUn_train)
    XUn_train_std = scaler.transform(XUn_train)
    start_time = time.time()
    clf = GridSearchCV(SVC(**svm_params), **svm_grid_params)
    clf.fit(XUn_train_std, dpgmm_idx)
    print 'SVM training time:', time.time()-start_time
    print 'Best SVM params:', clf.best_params_

    XUn_test_std = scaler.transform(XUn_test)
    # svm_test_idx = clf.predict(XUn_test_std)
    svm_test_idx = clf.predict(XUn_train_std)
    print svm_test_idx
    print dpgmm_idx
    XUnY_test = np.concatenate((XUn_test, Y_test), axis=1)
    dpgmm_test_idx = dpgmm.predict(XUnY_test)
    # total_correct = np.float(np.sum(dpgmm_test_idx == svm_test_idx))
    total_correct = np.float(np.sum(dpgmm_idx == svm_test_idx))
    # total = np.float(len(dpgmm_test_idx))
    total = np.float(len(dpgmm_idx))
    print 'Gating score: ', total_correct / total * 100.0

    # # long term prediction with MoE
    # mu_X_pred = np.zeros(H)
    # sigma_X_pred = np.zeros(H)
    #
    # XU_0 = XUs_test[0, :, 0:2]
    # x_0 = np.asscalar(XU_0[0, 0])
    # u_0 = np.asscalar(XU_0[0, 1])
    # xu_0 = np.append(x_0, u_0)
    # mu_X_pred[0], sigma_X_pred[0] = x_0, v0
    #
    # mu_xu_t = np.append(x_0, u_0)
    # sigma_xu_t = np.array([[v0, 0.],
    #                        [0., w0]])
    #
    #
    # mode = dpgmm.predict( XUnY_test[0].reshape(1,XUnY_test.shape[-1]) )
    # mode = int(np.asscalar(mode))
    #
    # start_time = time.time()
    # for t in range(1, H):
    #     gp_expert = MoE[mode]
    #     mu_x_t1, sigma_x_t1 = gp_expert.predict_dynamics_1_step(mu_xu_t, sigma_xu_t)
    #     u_t = np.asscalar(XU_0[t, 1])
    #     mu_xu_t = np.append(mu_x_t1, u_t)
    #     wu = np.asscalar(Wu[0, t])
    #     sigma_xu_t = np.array([[sigma_x_t1, 0.],
    #                            [0., wu]])
    #     mu_X_pred[t] = mu_x_t1
    #     sigma_X_pred[t] = sigma_x_t1
    #     mode = dpgmm.predict(mu_x_t1.reshape(1,-1))
    #     mode = int(np.asscalar(mode))
    # print 'Prediction time for MoE(w/o gating) with horizon', H, ':', time.time() - start_time
    #
    # dt = sim_1d_params['dt']
    # tm = np.array(range(H)) * dt
    #
    # plt.figure()
    # plt.title('MoE state evolution (testing data) with uncertainty propagation (moment matching)')
    # plt.xlabel('t')
    # plt.ylabel('x(t)')
    # for XUn in XUns_test:
    #     plt.plot(tm, XUn[:, 0])
    # plt.plot(tm, mu_X_pred, color='b', ls='-', marker='s', linewidth='2', label='learned model', markersize=7)
    # plt.plot(tm, traj_gt[:-1, 1], color='g', ls='-', marker='^', linewidth='2', label='real system', markersize=7)
    # plt.fill_between(tm, mu_X_pred - np.sqrt(sigma_X_pred) * 1.96, mu_X_pred + np.sqrt(sigma_X_pred) * 1.96, alpha=0.2)
    # plt.legend()
    # # compute prediction score
    # start_index = 0
    # horizon = T  # cannot be > T
    # end_index = start_index + horizon
    # weight = np.ones(horizon)  # weight long term prediction mse error based on time
    # score_cum = 0.
    # for XUn in XUns_test:
    #     x_m = XUn[start_index:end_index, 0]
    #     x_m.reshape(-1)
    #     bias_term = (x_m - mu_X_pred[start_index:end_index]) ** 2  # assumes mu_X_pred is computed for T
    #     var_term = sigma_X_pred[start_index:end_index]
    #     mse_ = bias_term + var_term
    #     mse_w = mse_ * weight
    #     score_cum += np.sum(mse_w)
    # print 'MoE continuous system prediction test data score:', score_cum / float(XUns_test.shape[0])

plt.show()
