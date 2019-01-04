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
import itertools

# np.random.seed(1) # 2 extra clusters
# np.random.seed(3) # no extra clusters
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
    'dt': 0.04,
    'T': 1.,
    'num_episodes': 20,
    'mode_c': {'mc':  {
                        'range': (0., 10.),
                        'dynamics': (-2.,10.),
                        'L': -0.2,
                        'target': 10.,
                        'noise': 1.,                # std dev
                        # 'noise': 0.,                # std dev
                        # 'init_x_var': 0.1,          # var
                        'init_x_var': 0.01,
                      }
              },
    'mode_d': { 'm1': {
                        'range': (0., 2.),
                        'dynamics': (-0.5, 10.),
                        'L': -0.05,
                        'target': 10.,
                        'noise': .125,                # std dev
                        # 'noise': 0.,
                        'init_x_var': 0.01,          # var
                        # 'init_x_var': 0.0,
                      },
                'm2': {
                        'range': (4., 6.),
                        'dynamics': (-1., 5.),
                        'L': -0.05,
                        'target': 10.,
                        'noise': .25,                # std dev
                        # 'noise': 0.,
                        'init_x_var': 0.05,          # var
                        # 'init_x_var': 0.0,
                      },
                'm3': {
                        'range': (8., 10.),
                        'dynamics': (-2, 10.),
                        'L': -0.2,
                        'target': 10.,
                        'noise': .5,                # std dev
                        # 'noise': 0.,
                        'init_x_var': 0.1,          # var
                        # 'init_x_var': 0.0,
                      },
               },
}

# type = 'cont'
type = 'disc'
mode_num = 3
mode_seq = ['m1','m2','m3']
# mode_seq = ['m1','m2']
global_pred = True
global_gp = True
cluster = True
fit_moe = True

# generate 1D continuous data
sim_1d_sys = sim_1d(sim_1d_params, type=type, mode_seq=mode_seq, mode_num=mode_num)
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
N_test = Nsam - N_train

data_train_s = data[:N_train,:,:]
Xs_train = data_train_s[:,:,:dX]
X_train = Xs_train.reshape((-1,Xs_train.shape[-1]))
Uns_train = data_train_s[:,:,dX:dX+dU]
Un_train = Uns_train.reshape((-1,Uns_train.shape[-1]))
Us_train = data_train_s[:,:,dX+dU:dX+dU+1]
Ys_train = data_train_s[:,:,dX+dU+1:dX+dU+1+dX]
XUns_train = np.concatenate((Xs_train,Uns_train),axis=2)
XUs_train = np.concatenate((Xs_train,Us_train),axis=2)
Y_train = Ys_train.reshape((-1,Ys_train.shape[-1]))
XYs_train = np.concatenate((Xs_train, Ys_train), axis=1)
XY_train = np.concatenate((X_train, Y_train), axis=1)
XUn_train = XUns_train.reshape((-1,XUns_train.shape[-1]))
XUnY_train = np.concatenate((XUn_train, Y_train), axis=1)
# np.random.shuffle(XUnY_train)
XUn_train = XUnY_train[:,0:2]
XUn_train = XUn_train.reshape(XUn_train.shape[0],-1)
Y_train = XUnY_train[:,2]
Y_train = Y_train.reshape(Y_train.shape[0],1)


# # plot training data
# plt.figure()
# plt.scatter(X_train, Y_train)
# plt.xlabel('x(t)')
# plt.ylabel('x(t+1)')
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # colors = iter(cm.rainbow(np.linspace(0, 1, len(XUns_train))))
# # plt.scatter(x, y, color=next(colors))
# for i in range(XUns_train.shape[0]):
#     ax.scatter(XUns_train[i,:,0], XUns_train[i,:,1], Ys_train[i,:,0])
# ax.set_xlabel('x(t)')
# ax.set_ylabel('u(t)')
# ax.set_zlabel('x(t+1)')

# plt.show()

data_test_s = data[N_train:,:,:]
Xs_test = data_test_s[:,:,:dX]
X_test = Xs_test.reshape((-1,Xs_test.shape[-1]))
Uns_test = data_test_s[:,:,dX:dX+dU]
Us_test = data_test_s[:,:,dX+dU:dX+dU+1]
Ys_test = data_test_s[:,:,dX+dU+1:dX+dU+1+dX]
XUns_test = np.concatenate((Xs_test,Uns_test),axis=2)
XUs_test = np.concatenate((Xs_test,Us_test),axis=2)
Y_test = Ys_test.reshape((-1,Ys_test.shape[-1]))
XUn_test = XUns_test.reshape((-1,XUns_test.shape[-1]))
XUnY_test = np.concatenate((XUn_test, Y_test), axis=1)

# # plot test data
# plt.figure()
# plt.scatter(X_test, Y_test)
# plt.xlabel('x(t)')
# plt.ylabel('x(t+1)')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # colors = iter(cm.rainbow(np.linspace(0, 1, len(XUns_train))))
# # plt.scatter(x, y, color=next(colors))
# for i in range(XUns_test.shape[0]):
#     ax.scatter(XUns_test[i,:,0], XUns_test[i,:,1], Ys_test[i,:,0])
# ax.set_xlabel('x(t)')
# ax.set_ylabel('u(t)')
# ax.set_zlabel('x(t+1)')
# plt.show()

# Train global GP
gpr_params = {
                    'alpha': 0., # alpha=0 when using white kernal
                    'kernel': C(1.0, (1e-2,1e2))*RBF(np.ones(dX+dU), (1e-3, 1e3))+W(noise_level=1., noise_level_bounds=(1e-2, 1e2)),
                    'n_restarts_optimizer': 10,
                    'normalize_y': False, # is not supported in the propogation function
                    }
if global_gp:
    gp = GaussianProcessRegressor(**gpr_params)
    start_time = time.time()
    gp.fit(XUn_train, Y_train)
    print 'Global GP fit time', time.time()-start_time

    # plot 1D continuous dynamics
    Xg = np.arange(-5, 15, 0.2)
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

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_title('1D continuous system dynamcis model')
    # # Plot the surface.
    # ax.plot_surface(Xp, Up, Yp, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # # ax.plot_wireframe(Xp, Up, Yp)
    # ax.plot_surface(Xp, Up, Yp_sigma_top, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.2)
    # ax.plot_surface(Xp, Up, Yp_sigma_bottom, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.2)
    # ax.plot(XUns_train[0,:,0], XUns_train[0,:,1], Ys_train[0,:,0])
    # ax.set_xlabel('x(t)')
    # ax.set_ylabel('u(t)')
    # ax.set_zlabel('x(t+1)')
    # plt.show()

# long-term prediction
predition_params = {
    'horizon': T-1,
    'initial_state_var': 0.,
}

H = predition_params['horizon']
v0 = sim_1d_params['mode_c']['mc']['init_x_var']
# v0 = predition_params['initial_state_var']
w0 = np.asscalar(Wu[0, 0])
dt = sim_1d_params['dt']
tm = np.array(range(H))*dt
if global_pred:
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
    # plt.figure()
    # plt.plot(tm, X_pred, label='x_pred')
    # plt.plot(tm, XU_0[:T,0], label='x_data')
    # plt.plot(tm, XU_0[:T,1], label='u_data')
    # plt.legend()
    # # plt.show()

    gp_mm = MomentMatching(gp)

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
    # plt.figure()
    # plt.title('GP State evolution (training data)')
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

    plt.figure()
    plt.title('GP state evolution (testing data)')
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
    # cluster_train_data = XUnY_train
    cluster_train_data = X_train
    cluster_train_op_data = Y_train
    cluster_test_data = X_test
    dpgmm_params = {
                    'K': 100, # cluster size
                    'restarts': 10, # number of restarts
                    # 'alpha': 1e-1, *
                    'alpha': 1e0,
                    'v0': 1+2,
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
    start_time = time.time()
    # dpgmm.fit(cluster_train_data)
    # dpgmm_train_idx = dpgmm.predict(cluster_train_data)
    dpgmm.fit(cluster_train_data)
    dpgmm_train_idx = dpgmm.predict(cluster_train_data)
    dpgmm_train_y_idx = dpgmm.predict(cluster_train_op_data)  # only work is the dpgmm was trained on only x
    # dpgmm.fit(XY_train)
    # dpgmm_train_idx = dpgmm.predict(XY_train)
    dpgmm_test_idx = dpgmm.predict(cluster_test_data)
    print 'DPGMM clustering time:', time.time() - start_time
    print 'Converged DPGMM',dpgmm.converged_, 'on', dpgmm.n_iter_, 'iterations with lower bound',dpgmm.lower_bound_

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
    labels, counts = zip(*sorted(Counter(dpgmm_train_idx).items(),key=operator.itemgetter(0)))
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

    # # plot cluster components
    # plt.figure()
    # plt.bar(labels,counts,color=colors)
    # plt.title('DPGMM cluster dist')

    # # plot clustered train data
    # col = np.zeros([cluster_train_data.shape[0],3])
    # mark = np.array(['None']*cluster_train_data.shape[0])
    # i=0
    # for label in labels:
    #     col[(dpgmm_train_idx==label)] = colors[i]
    #     mark[(dpgmm_train_idx == label)] = markers[i]
    #     i+=1
    #
    # col = col.reshape(N_train,-1,3)
    # mark = mark.reshape(N_train,-1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(XUns_train.shape[0]):
    #     for j in range(XUns_train.shape[1]):
    #         ax.scatter(XUns_train[i,j,0], XUns_train[i,j,1], Ys_train[i,j], c=col[i,j], marker=mark[i,j])
    # ax.set_xlabel('x(t)')
    # ax.set_ylabel('u(t)')
    # ax.set_zlabel('x(t+1)')
    # plt.title('DPGMM train clustering')

    # # plot clustered trajectory
    # plt.figure()
    # for i in range(XUns_train.shape[0]):
    #     for j in range(XUns_train.shape[1]):
    #         plt.scatter(tm[j],XUns_train[i,j,0], c=col[i,j], marker=mark[i,j])
    # plt.title('Clustered train trajectories')

    # plot clustered test data
    labels1, counts1 = zip(*sorted(Counter(dpgmm_test_idx).items(), key=operator.itemgetter(0)))
    K = len(labels1)
    colors1 = get_N_HexCol(K)
    colors1 = np.asarray(colors1) / 255.
    marker_set = ['.', 'o', '*', '+', '^', 'x', 'o', 'D', 's']
    marker_set_size = len(marker_set)
    if K < marker_set_size:
        markers1 = marker_set[:K]
    else:
        markers1 = ['o'] * K

    col1 = np.zeros([cluster_test_data.shape[0], 3])
    mark1 = np.array(['None'] * cluster_test_data.shape[0])
    i = 0
    for label in labels1:
        col1[(dpgmm_test_idx == label)] = colors1[i]
        mark1[(dpgmm_test_idx == label)] = markers1[i]
        i += 1

    col1 = col1.reshape(N_test, -1, 3)
    mark1 = mark1.reshape(N_test, -1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(XUns_test.shape[0]):
        for j in range(XUns_test.shape[1]):
            # ax.scatter(XUns_test[i, j, 0], XUns_test[i, j, 1], Ys_test[i, j], c=col1[i, j], marker=mark1[i, j])
            ax.scatter(XUns_test[i, j, 0], XUns_test[i, j, 1], Ys_test[i, j], c=col1[i, j])
            # plt.show()
    ax.set_xlabel('x(t)')
    ax.set_ylabel('u(t)')
    ax.set_zlabel('x(t+1)')
    plt.title('DPGMM test clustering')

    # init x dist estimation
    N = Xs_train.shape[0]
    init_x_table = {}
    for label in labels:
        init_x_table[label] = {'X': [], 'mu': None, 'var': None}
    for n in range(N):
        idx = dpgmm.predict(Xs_train[n])
        for label in labels:
            X_mode = Xs_train[n][(idx == label)]
            init_x_table[label]['X'].append(np.asarray(X_mode[0]))

    for label in labels:
        init_x_table[label]['mu'] = np.mean(init_x_table[label]['X'])
        init_x_table[label]['var'] = np.var(init_x_table[label]['X'])

if fit_moe and cluster:
    start_time = time.time()
    MoE = {}
    for label in labels:
        x_train = XUn_train[(np.logical_and((dpgmm_train_idx == label), (dpgmm_train_y_idx == label)))]
        y_train = Y_train[(np.logical_and((dpgmm_train_idx == label), (dpgmm_train_y_idx == label)))]
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

    # XU svm
    scaler1 = StandardScaler().fit(XUn_train)
    XUn_train_std = scaler1.transform(XUn_train)
    start_time = time.time()
    clf1 = GridSearchCV(SVC(**svm_params), **svm_grid_params)
    clf1.fit(XUn_train_std[:-1, :], dpgmm_train_idx[1:])
    print 'SVM training time:', time.time() - start_time
    print 'Best SVM params:', clf1.best_params_

    XUn_test_std = scaler1.transform(XUn_test)
    svm_test_idx1 = clf1.predict(XUn_test_std[:-1, :])
    svm_train_idx1 = clf1.predict(XUn_train_std[:-1, :])
    total_correct = np.float(np.sum(dpgmm_test_idx[1:] == svm_test_idx1))
    total = np.float(len(dpgmm_test_idx) - 1)
    # total_correct = np.float(np.sum(dpgmm_train_idx[1:] == svm_train_idx))
    # total = np.float(len(dpgmm_train_idx) - 1)
    print 'XU Gating score: ', total_correct / total * 100.0
    plt.figure()
    # plt.plot(svm_train_idx)
    # plt.plot(dpgmm_train_idx[1:])
    plt.plot(svm_test_idx1, label='predicted')
    plt.plot(dpgmm_test_idx[1:], label='actual')
    plt.title('XU')
    plt.legend()

    # # XUG svm
    # XUnG_train = np.concatenate((XUn_train, dpgmm_train_idx[:, np.newaxis]), axis=1)
    # scaler = StandardScaler().fit(XUnG_train)
    # XUnG_train_std = scaler.transform(XUnG_train)
    # start_time = time.time()
    # clf = GridSearchCV(SVC(**svm_params), **svm_grid_params)
    # clf.fit(XUnG_train_std[:-1, :], dpgmm_train_idx[1:].reshape(-1))
    # print 'SVM training time:', time.time()-start_time
    # print 'Best SVM params:', clf.best_params_
    #
    # XUnG_test = np.concatenate((XUn_test[:-1, :], dpgmm_test_idx[:-1][:,np.newaxis]),axis=1)
    # XUnG_test_std = scaler.transform(XUnG_test)
    # svm_test_idx = clf.predict(XUnG_test_std)
    # svm_train_idx = clf.predict(XUnG_train_std)
    # total_correct = np.float(np.sum(dpgmm_test_idx[1:] == svm_test_idx))
    # total = np.float(len(dpgmm_test_idx)-1)
    # # total_correct = np.float(np.sum(dpgmm_train_idx[1:] == svm_train_idx))
    # # total = np.float(len(dpgmm_train_idx) - 1)
    # print 'XUG Gating score: ', total_correct / total * 100.0
    # plt.figure()
    # # plt.plot(svm_train_idx)
    # # plt.plot(dpgmm_train_idx[1:])
    # plt.plot(svm_test_idx, label='predicted')
    # plt.plot(dpgmm_test_idx[1:], label='actual')
    # plt.title('XUG')
    # plt.legend()

    # long term prediction with MoE
    mu_X_pred = np.zeros(H)
    sigma_X_pred = np.zeros(H)
    mode_pred = np.zeros(H)
    mode_xu_pred = np.zeros(H)

    XU_0 = XUs_test[0, :, 0:2]
    x_0 = np.asscalar(XU_0[0, 0])
    u_0 = np.asscalar(XU_0[0, 1])
    xu_0 = np.append(x_0, u_0)
    mu_X_pred[0], sigma_X_pred[0] = x_0, v0

    mu_xu_t = np.append(x_0, u_0)
    sigma_xu_t = np.array([[v0, 0.],
                           [0., w0]])
    # prediction horizon H is almost equal to T, not sure if it can be reduced
    # we always predict along the first test rollout
    # t0 mode prediction
    mode_d0_actual = dpgmm_test_idx[0]  # actual mode
    mode_d0_gate = mode_d1_gate = svm_test_idx1[0]  # we assume this to be same
    assert(mode_d0_actual==mode_d0_gate)
    mode = mode_pred[0] = mode_d0_gate
    start_time = time.time()
    mode_prev = mode
    for t in range(1, H):
        gp_expert = MoE[mode]
        if mode == mode_prev:
            mu_x_t1, sigma_x_t1 = gp_expert.predict_dynamics_1_step(mu_xu_t, sigma_xu_t)
        else:
            mu_x_t1 = init_x_table[mode]['mu']
            sigma_x_t1 = init_x_table[mode]['var']
        # mu_x_t1 = traj_gt[t, 1]
        u_t = np.asscalar(XU_0[t, 1])
        mu_xu_t = np.append(mu_x_t1, u_t)
        wu = np.asscalar(Wu[0, t])
        sigma_xu_t = np.array([[sigma_x_t1, 0.],
                               [0., wu]])
        mu_X_pred[t] = mu_x_t1
        sigma_X_pred[t] = sigma_x_t1
        mode_pred[t] = mode

        mu_xu_t_std = scaler1.transform(mu_xu_t.reshape(1,-1))
        mode_prev = mode
        mode = clf1.predict(mu_xu_t_std.reshape(1,-1))
        mode = int(mode)


        # xu_t_actual = XU_0[t]
        # xu_t_actual_std = scaler1.transform(xu_t_actual.reshape(1, -1))
        # mode_xu = clf1.predict(xu_t_actual_std.reshape(1, -1))
        # mode_xu = int(mode_xu)
        # mode_xu_pred[t] = mode_xu

    print 'Prediction time for MoE(w/o gating) with horizon', H, ':', time.time() - start_time

    dt = sim_1d_params['dt']
    tm = np.array(range(H)) * dt

    plt.figure()
    plt.title('MoE state evolution (testing data)')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    for XUn in XUns_test:
        plt.plot(tm, XUn[:, 0])
    plt.plot(tm, mu_X_pred, color='b', ls='-', marker='s', linewidth='2', label='learned model', markersize=7)
    plt.plot(tm, traj_gt[:-1, 1], color='g', ls='-', marker='^', linewidth='2', label='real system', markersize=7)
    plt.fill_between(tm, mu_X_pred - np.sqrt(sigma_X_pred) * 1.96, mu_X_pred + np.sqrt(sigma_X_pred) * 1.96, alpha=0.2)
    plt.legend()

    plt.figure()
    plt.plot(mode_pred, label='mode_pred')
    plt.plot(svm_test_idx1[:H], label='svm_test')
    plt.plot(dpgmm_test_idx[:H], label='dpgmm_test')
    # plt.plot(mode_xu_pred[:H], label='mode_xu')
    plt.legend()
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
