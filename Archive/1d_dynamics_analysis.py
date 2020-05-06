# import sys
# sys.path.insert(0, '/home/shahbaz/Research/Software/Spyder_ws/gps/python')
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scalar_dynamics_sys import sim_1d
# from scalar_dynamics_sys import MomentMatching
# from scalar_dynamics_sys import UGP
from model_leraning_utils import UGP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.gaussian_process import GaussianProcessRegressor
import GPy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from copy import deepcopy
from Archive.utilities import get_N_HexCol
# from utilities import plot_ellipse
# from utilities import logsum
from sklearn import mixture
from collections import Counter
import operator
from matplotlib.ticker import MaxNLocator
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time
from itertools import compress

# np.random.seed(10) # case in which gp pred ugp goes wrong (single svm) , so good plot
# np.random.seed(6) # good for nonlinear case and changed noise level

sim_1d_params = {
    'x0': 0.,
    'dt': 0.04,
    'T': 1.,
    'num_episodes': 15,
    'mode_c': {'mc':  {
                        'range': (0., 10.),
                        'dynamics': (-2.,10.),
                        'L': -0.2,
                        'target': 10.,
                        'noise': 1.,                # std dev
                        # 'noise': 0.,                # std dev
                        # 'init_x_var': 0.1,          # var
                        'init_x_var': 0.001,
                      }
              },
    'mode_d': { 'm1': {
                        'range': (0., 2.),
                        # 'range': (0., 10.),
                        'dynamics': (-0.5, 10.),
                        'L': -0.05,
                        'target': 10.,
                        'noise': .125*2,                # std dev
                        # 'noise': 0.,
                        'init_x_var': 0.01,          # var
                        # 'init_x_var': 0.0,
                      },
                'm2': {
                        'range': (4., 6.),
                        # 'range': (3., 6.),
                        'dynamics': (-1., 5.),
                        'L': -0.05,
                        'target': 10.,
                        'noise': .25/2,                # std dev
                        # 'noise': 0.,
                        'init_x_var': 0.05,          # var
                        # 'init_x_var': 0.0,
                      },
                'm3': {
                        'range': (8., 10.),
                        # 'range': (7., 10.),
                        'dynamics': (-2, 10.),
                        'L': -0.1,
                        'target': 10.,
                        'noise': .5,                # std dev
                        # 'noise': 0.,
                        'init_x_var': 0.1,          # var
                        # 'init_x_var': 0.2,
                      },
                'm4': {
                        'range': (12., 14.),
                        # 'range': (7., 10.),
                        'dynamics': (-2, 10.),
                        'L': -0.5,
                        'target': 14.,
                        'noise': .5,                # std dev
                        # 'noise': 0.,
                        'init_x_var': 0.1,          # var
                        # 'init_x_var': 0.0,
                      },
               },
}
ugp_params = {
    'alpha': 1.,
    'kappa': 2.,
    'beta': 0.,
}

# type = 'cont'
type = 'disc'
mode_num = 3
mode_seq = ['m1','m2','m3']
# mode_seq = ['m1']
# mode_seq = ['m2', 'm1', 'm3']
gmm_clust = False
global_pred = True
global_gp = True
cluster = True
fit_moe = True
mmgp = False
ugp = True
min_prob_grid = 0.01 # 1%

# generate 1D continuous data
sim_1d_sys = sim_1d(sim_1d_params, type=type, mode_seq=mode_seq, mode_num=mode_num)
traj_gt = sim_1d_sys.sim_episode(noise=False)
num_episodes = sim_1d_params['num_episodes']
traj_list = sim_1d_sys.sim_episodes(num_episodes)
traj_data = np.array(traj_list)

# plot 1D continuous data
# plt.rcParams.update({'font.size': 25})
plt.figure()
# plt.title('1D continuous system data')
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
                    # 'alpha': 0, # alpha=0 when using white kernal
                    'alpha': 1e-6, # alpha=0 when using white kernal
                    # 'kernel': C(1.0, (1e-2,1e2))*RBF(np.ones(dX+dU), (1e-3, 1e3))+W(noise_level=1., noise_level_bounds=(1e-3, 1e-2)),
                    'kernel': C(1.0, (1e-2,1e2))*RBF(np.ones(dX+dU), (1e-3, 1e3)),
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
    # 'horizon': 10,
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
    x0 = np.asscalar(XU_0[0,0])
    u0 = np.asscalar(XU_0[0,1])

    # gp mean long-term prediction without uncertainty
    # X_pred[0] = x0
    # xu_t = np.append(x0,u0)
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



    # # State evolution (training data) with uncertainty propagation
    # mu_X_pred = np.zeros(H)
    # sigma_X_pred = np.zeros(H)
    #
    # XU_0 = XUs_train[0,:,0:2]
    # x0 = np.asscalar(XU_0[0,0])
    # u0 = np.asscalar(XU_0[0,1])
    # xu_0 = np.append(x0,u0)
    # mu_X_pred[0], sigma_X_pred[0] = x0, v0
    #
    # mu_xu_t = np.append(x0,u0)
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
    #     plt.plot(tm, XUn[:H,0])
    # plt.plot(tm, mu_X_pred, color='b', ls='-', marker='s', linewidth='2', label='learned model', markersize=7)
    # plt.plot(tm, traj_gt[:H,1], color='g', ls='-', marker='^', linewidth='2', label='real system', markersize=7)
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
    # print 'GP prediction train data score:', score_cum/float(XUns_train.shape[0])

    if mmgp:
        gp_mm = MomentMatching(gp)
        # State evolution (test data) with uncertainty propagation MM
        mu_X_pred = np.zeros(H)
        sigma_X_pred = np.zeros(H)

        XU_0 = XUs_test[0,:,0:2]
        x0 = np.asscalar(XU_0[0,0])
        # u0 = np.asscalar(XU_0[0,1])
        u0, w0 = sim_1d_sys.get_action(x0)
        xu_0 = np.append(x0,u0)
        mu_X_pred[0], sigma_X_pred[0] = x0, v0

        mu_xu_t = np.append(x0,u0)
        sigma_xu_t = np.array([[v0, 0.],
                                [0., w0]])

        start_time = time.time()
        for t in range(1,H):
            mu_x_t1, sigma_x_t1 = gp_mm.predict_dynamics_1_step(mu_xu_t, sigma_xu_t)
            # u_t = np.asscalar(XU_0[t, 1])
            xt1 = np.random.normal(mu_x_t1, sigma_x_t1)
            u_t, wu = sim_1d_sys.get_action(xt1)
            mu_xu_t = np.append(mu_x_t1, u_t)
            # wu = np.asscalar(Wu[0, t])
            sigma_xu_t = np.array([[sigma_x_t1, 0.],
                                  [0., wu]])
            mu_X_pred[t] = mu_x_t1
            sigma_X_pred[t] = sigma_x_t1
        print 'Prediction time for horizon MM', H,':', time.time()-start_time

        plt.figure()
        plt.title('Long-term prediction with GP')
        plt.xlabel('Time, t')
        plt.ylabel('State, x(t)')
        for XUn in XUns_test:
            plt.plot(tm, XUn[:H,0])
        plt.plot(tm, mu_X_pred, color='b', ls='-', marker='s', linewidth='2', label='Learned dynamics', markersize=7)
        plt.plot(tm, traj_gt[:H,1], color='g', ls='-', marker='^', linewidth='2', label='True dynamics', markersize=7)
        plt.fill_between(tm, mu_X_pred - np.sqrt(sigma_X_pred)*1.96, mu_X_pred + np.sqrt(sigma_X_pred)*1.96, alpha=0.2)
        plt.legend()
        plt.savefig('gp_long-term_mm.pdf')
        plt.savefig('gp_long-term_mm.png', format='png', dpi=1000)

        # compute prediction score MM
        start_index = 0
        # horizon = T # cannot be > T
        horizon = H  # cannot be > T
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
        print 'GP MM prediction test data score:', score_cum/float(XUns_test.shape[0])

    # State evolution (test data) with uncertainty propagation UGP
    if ugp:
        mu_X_pred = np.zeros(H)
        sigma_X_pred = np.zeros(H)

        XU_0 = XUs_test[0, :, 0:2]
        x0 = np.asscalar(XU_0[0, 0])
        # u0 = np.asscalar(XU_0[0, 1])
        u0, w0 = sim_1d_sys.get_action(x0)
        xu_0 = np.append(x0, u0)
        mu_X_pred[0], sigma_X_pred[0] = x0, v0

        mu_xu_t = np.append(x0, u0)
        sigma_xu_t = np.array([[v0, 0.],
                               [0., w0]])

        ugp_dyn = UGP(dX + dU, **ugp_params)
        # ugp_dyn = UGP(dX, **ugp_params)
        start_time = time.time()
        for t in range(1, H):
            mu_x_t1, sigma_x_t1, _, _, _ = ugp_dyn.get_posterior(gp, mu_xu_t, sigma_xu_t)
            # actions of the first roll out
            xt1 = np.random.normal(mu_x_t1, sigma_x_t1)
            u_t, wu = sim_1d_sys.get_action(xt1)
            # u_t = np.asscalar(XU_0[t, 1])
            mu_xu_t = np.append(mu_x_t1, u_t)
            # wu = np.asscalar(Wu[0, t])
            sigma_xu_t = np.array([[sigma_x_t1, 0.],
                                   [0., wu]])
            mu_X_pred[t] = mu_x_t1
            sigma_X_pred[t] = sigma_x_t1
        print 'Prediction time for horizon UGP', H, ':', time.time() - start_time

        # prepare for contour plot
        tm_grid = tm
        grid_size = 0.2
        x_grid = np.arange(-1, 11, grid_size)  # TODO: get the ranges from the mode dict
        Xp, Tp = np.meshgrid(x_grid, tm_grid)
        prob_map = np.zeros((len(tm_grid), len(x_grid)))
        for i in range(len(x_grid)):
            for t in range(len(tm_grid)):
                x = x_grid[i]
                mu = mu_X_pred[t]
                sig = np.sqrt(sigma_X_pred[t])
                prob_val = sp.stats.norm.pdf(x, mu, sig)
                prob_map[t, i] += prob_val

        plt.figure()
        plt.title('Long-term prediction with GP')
        plt.xlabel('Time, t')
        plt.ylabel('State, x(t)')
        for XUn in XUns_test:
            plt.plot(tm, XUn[:H, 0])
        plt.plot(tm, mu_X_pred, color='b', ls='-', marker='s', linewidth='2', label='Learned dynamics', markersize=7)
        plt.plot(tm, traj_gt[:H, 1], color='g', ls='-', marker='^', linewidth='2', label='True dynamics', markersize=7)
        plt.fill_between(tm, mu_X_pred - np.sqrt(sigma_X_pred) * 1.96, mu_X_pred + np.sqrt(sigma_X_pred) * 1.96, alpha=0.2)
        # plt.contourf(Tp, Xp, prob_map, 50, cmap='Blues', alpha=1., vmin=0., vmax=1.5)
        # plt.colorbar()
        plt.legend()
        plt.savefig('gp_long-term_ugp.pdf')
        plt.savefig('gp_long-term_ugp.png', format='png', dpi=1000)

        # compute prediction score UGP
        start_index = 0
        # horizon = T # cannot be > T
        horizon = H  # cannot be > T
        end_index = start_index + horizon
        weight = np.ones(horizon) # weight long term prediction mse error based on time
        score_cum = 0.
        for XUn in XUns_test:
            x_m = XUn[start_index:end_index,0]
            x_m.reshape(-1)
            # bias_term = (x_m - mu_X_pred[start_index:end_index])**2 # assumes mu_X_pred is computed for T
            # var_term = sigma_X_pred[start_index:end_index]
            # mse_ = bias_term + var_term
            # mse_w = mse_*weight
            # score_cum += np.sum(mse_w)
            mu = mu_X_pred[start_index:end_index]
            sig = np.sqrt(sigma_X_pred[start_index:end_index])
            traj_score=0.
            for i in range(len(x_m)):
                xm_ = x_m[i]
                mu_ = mu[i]
                sig_ = sig[i]
                li = sp.stats.norm.logpdf(xm_, mu_, sig_)
                traj_score += li
            # traj_score_avg = traj_score/float(len(x_m))
            # score_cum += traj_score_avg
            score_cum += traj_score
        print 'UGP prediction test data score:', score_cum/float(XUns_test.shape[0])


if cluster:
    # cluster_train_data = XUnY_train
    cluster_train_data = X_train
    cluster_train_op_data = Y_train
    cluster_test_data = X_test
    K = cluster_train_data.shape[0]//3
    dpgmm_params = {
                    'K': K, # cluster size
                    'restarts': 10, # number of restarts
                    # 'alpha': 1e-1, *
                    'alpha': 1e2,
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
    dpgmm_train_y_idx = dpgmm.predict(cluster_train_op_data)  # only work if the dpgmm was trained on only x
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

    # plot cluster components
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.bar(labels,counts,color=colors)
    plt.title('DPGMM clustering')
    plt.ylabel('Cluster sizes')
    plt.xlabel('Cluster labels')
    plt.savefig('dpgmm_1d_dyn_cluster counts.pdf')
    plt.savefig('dpgmm_1d_dyn_cluster counts.png', format='png', dpi=1000)

    # plot clustered train data
    col = np.zeros([cluster_train_data.shape[0],3])
    mark = np.array(['None']*cluster_train_data.shape[0])
    i=0
    for label in labels:
        col[(dpgmm_train_idx==label)] = colors[i]
        mark[(dpgmm_train_idx == label)] = markers[i]
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
    plt.title('DPGMM train clustering')

    # plot clustered trajectory
    plt.figure()
    for i in range(XUns_train.shape[0]):
        for j in range(XUns_train.shape[1]):
            plt.scatter(tm[j],XUns_train[i,j,0], c=col[i,j], marker=mark[i,j])
    plt.title('Clustered train trajectories')

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
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(XUns_test.shape[0]):
    #     for j in range(XUns_test.shape[1]):
    #         # ax.scatter(XUns_test[i, j, 0], XUns_test[i, j, 1], Ys_test[i, j], c=col1[i, j], marker=mark1[i, j])
    #         ax.scatter(XUns_test[i, j, 0], XUns_test[i, j, 1], Ys_test[i, j], c=col1[i, j])
    #         # plt.show()
    # ax.set_xlabel('x(t)')
    # ax.set_ylabel('u(t)')
    # ax.set_zlabel('x(t+1)')
    # plt.title('DPGMM test clustering')

    if gmm_clust:
        K = cluster_train_data.shape[0] // 3
        gmm_params = {
            'K': K,  # cluster size
            'restarts': 10,  # number of restarts
            'enable': False,
        }

        gmm = mixture.GaussianMixture(n_components=gmm_params['K'],
                                      covariance_type='full',
                                      tol=1e-6,
                                      n_init=gmm_params['restarts'],
                                      warm_start=False,
                                      init_params='random',
                                      max_iter=1000)
        gmm.fit(cluster_train_data)
        gmm_train_idx = gmm.predict(cluster_train_data)
        labels2, counts2 = zip(*sorted(Counter(gmm_train_idx).items(), key=operator.itemgetter(0)))
        K = len(labels2)
        colors = get_N_HexCol(K)
        colors = np.asarray(colors) / 255.

        # plot cluster components
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.bar(labels2, counts2, color=colors)
        plt.title('GMM clustering')
        plt.ylabel('Cluster sizes')
        plt.xlabel('Cluster labels')
        plt.savefig('gmm_1d_dyn_cluster counts.pdf')
        plt.savefig('gmm_1d_dyn_cluster counts.png', format='png', dpi=1000)

    # # init x dist estimation
    # N = Xs_train.shape[0]
    # init_x_table = {}
    # for label in labels:
    #     init_x_table[label] = {'X': [], 'mu': None, 'var': None}
    # for n in range(N):
    #     idx = dpgmm.predict(Xs_train[n])
    #     for label in labels:
    #         X_mode = Xs_train[n][(idx == label)]
    #         if X_mode.shape[0] > 0:
    #             init_x_table[label]['X'].append(np.asarray(X_mode[0]))
    #
    # for label in labels:
    #     init_x_table[label]['mu'] = np.mean(init_x_table[label]['X'])
    #     init_x_table[label]['var'] = np.var(init_x_table[label]['X'])

    # transition GP
    trans_gpr_params = {
        'alpha': 0.,  # alpha=0 when using white kernal
        'kernel': C(1.0, (1e-2, 1e2)) * RBF(np.ones(dX+dU), (1e-3, 1e3)) + W(noise_level=1., noise_level_bounds=(1e-2, 1e2)),
        'n_restarts_optimizer': 10,
        'normalize_y': False,  # is not supported in the propogation function
    }
    trans_dicts = {}
    for xu_train in XUns_train:                     # TODO: X_train is used directly not cluster_train_data
        x_train_idx = dpgmm.predict(xu_train[:, :dX])
        iddiff = x_train_idx[:-1] != x_train_idx[1:]
        trans_data = zip(xu_train[:-1, :dX+dU], xu_train[1:, :dX], x_train_idx[:-1], x_train_idx[1:])
        trans_data_p = list(compress(trans_data, iddiff))
        for xu, y, xid, yid in trans_data_p:
            if (xid, yid) not in trans_dicts:
                trans_dicts[(xid, yid)] = {'XU': [], 'Y': [], 'gp': None}
            trans_dicts[(xid, yid)]['XU'].append(xu)
            trans_dicts[(xid, yid)]['Y'].append(y)
    for trans_data in trans_dicts:
        XU = np.array(trans_dicts[trans_data]['XU']).reshape(-1,dX+dU)
        Y = np.array(trans_dicts[trans_data]['Y']).reshape(-1,1)
        gp = GaussianProcessRegressor(**trans_gpr_params)
        gp.fit(XU, Y)
        trans_dicts[trans_data]['gp'] = deepcopy(gp)
        del gp

if fit_moe and cluster:
    start_time = time.time()
    MoE = {}
    MoE_gp = {}
    for label in labels:
        x_train = XUn_train[(np.logical_and((dpgmm_train_idx == label), (dpgmm_train_y_idx == label)))]
        y_train = Y_train[(np.logical_and((dpgmm_train_idx == label), (dpgmm_train_y_idx == label)))]
        gp_ = GaussianProcessRegressor(**gpr_params)
        gp_.fit(x_train, y_train)
        # gp_expert = MomentMatching(gp_)
        # MoE[label] = deepcopy(gp_expert)
        MoE_gp[label] = deepcopy(gp_)
        # del gp_expert
        del gp_
    print 'MoE training time:', time.time() - start_time

    # gating network training
    svm_grid_params = {
                        'param_grid': {"C": np.logspace(-10, 10, endpoint=True, num=11, base=2.),
                                       "gamma": np.logspace(-10, 10, endpoint=True, num=11, base=2.)},
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

    # # XU svm
    # scaler1 = StandardScaler().fit(XUn_train)
    # XUn_train_std = scaler1.transform(XUn_train)
    # start_time = time.time()
    # clf1 = GridSearchCV(SVC(**svm_params), **svm_grid_params)
    # clf1.fit(XUn_train_std[:-1, :], dpgmm_train_idx[1:])
    # print 'SVM training time:', time.time() - start_time
    # print 'Best SVM params:', clf1.best_params_
    #
    # start_time = time.time()
    # XUn_test_std = scaler1.transform(XUn_test)
    # svm_test_idx1 = clf1.predict(XUn_test_std[:-1, :])
    # print 'Gating prediction time for', len(svm_test_idx1), 'samples:', time.time()-start_time
    # total_correct = np.float(np.sum(dpgmm_test_idx[1:] == svm_test_idx1))
    # total = np.float(len(dpgmm_test_idx) - 1)
    # print 'XU Gating score: ', total_correct / total * 100.0
    # plt.figure()
    # # plt.plot(svm_train_idx)
    # # plt.plot(dpgmm_train_idx[1:])
    # plt.plot(svm_test_idx1, label='predicted')
    # plt.plot(dpgmm_test_idx[1:], label='actual')
    # plt.title('XU')
    # plt.legend()

    # svm for each mode
    start_time = time.time()
    scaler1 = StandardScaler().fit(XUn_train)
    XUn_train_std = scaler1.transform(XUn_train)
    SVMs = {}
    # XUnI = np.concatenate((XUn_train_std[:-1,:], dpgmm_train_idx[1:].reshape(-1,1)), axis=1)
    XUnI = zip(XUn_train_std[:-1, :], dpgmm_train_idx[1:])
    for label in labels:
        # xui = XUnI[(dpgmm_train_idx[:-1] == label)]
        xui = list(compress(XUnI, (dpgmm_train_idx[:-1] == label)))
        # xu = xui[:, :dX+dU]
        xu, i = zip(*xui)
        xu = np.array(xu)
        i = list(i)
        # i = xui[:, dX + dU:].reshape(-1)
        clf = GridSearchCV(SVC(**svm_params), **svm_grid_params)
        clf.fit(xu, i)
        SVMs[label] = deepcopy(clf)
        del clf
    print 'SVMs training time:', time.time() - start_time

    # XUG svm
    # XUnG_train = np.concatenate((XUn_train, dpgmm_train_idx[:, np.newaxis]), axis=1)
    # scaler1 = StandardScaler().fit(XUnG_train)
    # XUnG_train_std = scaler1.transform(XUnG_train)
    # start_time = time.time()
    # clf1 = GridSearchCV(SVC(**svm_params), **svm_grid_params)
    # clf1.fit(XUnG_train_std[:-1, :], dpgmm_train_idx[1:].reshape(-1))
    # print 'SVM training time:', time.time()-start_time
    # print 'Best SVM params:', clf1.best_params_
    #
    # XUnG_test = np.concatenate((XUn_test[:-1, :], dpgmm_test_idx[:-1][:,np.newaxis]),axis=1)
    # XUnG_test_std = scaler1.transform(XUnG_test)
    # svm_test_idx1 = clf1.predict(XUnG_test_std)
    # # svm_train_idx = clf1.predict(XUnG_train_std)
    # total_correct = np.float(np.sum(dpgmm_test_idx[1:] == svm_test_idx1))
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
    if mmgp:
        mu_X_pred = np.zeros(H)
        sigma_X_pred = np.zeros(H)
        mode_pred = np.zeros(H)

        XU_0 = XUs_test[0, :, 0:2]
        x0 = np.asscalar(XU_0[0, 0])
        u0 = np.asscalar(XU_0[0, 1])
        xu_0 = np.append(x0, u0)
        v0 = sim_1d_params['mode_d']['m1']['init_x_var']
        mu_X_pred[0], sigma_X_pred[0] = x0, v0

        mu_xu_t = np.append(x0, u0)
        sigma_xu_t = np.array([[v0, 0.],
                               [0., w0]])
        # prediction horizon H is almost equal to T, not sure if it can be reduced
        # we always predict along the first test rollout
        # t0 mode prediction
        mode_d0_actual = dpgmm_test_idx[0]  # actual mode
        mode_d0_gate = svm_test_idx1[0]  # we assume this to be same
        assert(mode_d0_actual==mode_d0_gate)
        mode = mode_pred[0] = mode_d0_gate
        mode_prev = mode

        # long term prediction with moment matching
        start_time = time.time()
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
            # wu = 0.
            sigma_xu_t = np.array([[sigma_x_t1, 0.],
                                   [0., wu]])
            mu_X_pred[t] = mu_x_t1
            sigma_X_pred[t] = sigma_x_t1
            mode_pred[t] = mode

            mu_xu_t_std = scaler1.transform(mu_xu_t.reshape(1,-1))
            mode_prev = mode
            mode = clf1.predict(mu_xu_t_std.reshape(1,-1))
            mode = int(mode)

        print 'Prediction time for MoE MM with horizon', H, ':', time.time() - start_time

        # plot long term prediction results of moment matching
        dt = sim_1d_params['dt']
        tm = np.array(range(H)) * dt

        plt.figure()
        plt.title('Long-term prediction with mixture of GP')
        plt.xlabel('Time, t')
        plt.ylabel('State, x(t)')
        for XUn in XUns_test:
            plt.plot(tm, XUn[:H, 0])
        plt.plot(tm, mu_X_pred, color='b', ls='-', marker='s', linewidth='2', label='Learned dynamics', markersize=7)
        plt.plot(tm, traj_gt[:H, 1], color='g', ls='-', marker='^', linewidth='2', label='True dynamics', markersize=7)
        plt.fill_between(tm, mu_X_pred - np.sqrt(sigma_X_pred) * 1.96, mu_X_pred + np.sqrt(sigma_X_pred) * 1.96, alpha=0.2)
        plt.legend()
        plt.savefig('ME_long-term_mm.pdf')
        plt.savefig('ME_long-term_mm.png', format='png', dpi=1000)

        # plt.figure()
        # plt.title('MM')
        # plt.plot(mode_pred, label='mode_pred')
        # plt.plot(svm_test_idx1[:H], label='svm_test')
        # plt.plot(dpgmm_test_idx[:H], label='dpgmm_test')
        # plt.legend()

        # compute prediction score MM
        start_index = 0
        # horizon = T  # cannot be > T
        horizon = H  # cannot be > T
        end_index = start_index + horizon
        weight = np.ones(horizon)  # weight long term prediction mse error based on time
        score_cum = 0.
        for XUn in XUns_test:
            x_m = XUn[start_index:end_index, 0]
            x_m.reshape(-1)
            bias_term = (x_m - mu_X_pred[start_index:end_index]) ** 2  # assumes mu_X_pred is computed for T
            var_term = sigma_X_pred[start_index:end_index]
            mse_ = bias_term + var_term
            mse_w = mse_ * weight
            score_cum += np.sum(mse_w)
        print 'MoE MM system prediction test data score:', score_cum / float(XUns_test.shape[0])

    # long term prediction with UGP
    if ugp:
        # sigmaIps = np.zeros((H, 2*(dX+dU)+1, dX+dU))
        sigmaOps = np.zeros((H, 2 * (dX + dU) + 1, dX))
        XU_0 = XUs_test[0, :, 0:2]
        x0 = np.asscalar(XU_0[0, 0])
        u0 = np.asscalar(XU_0[0, 1])
        v0 = sim_1d_params['mode_d']['m1']['init_x_var']
        # sigmaOp = np.zeros((2 * (dX + dU) + 1, dX))
        # sigmaOp.fill(x0)

        # t0 mode prediction
        mode_d0_actual = dpgmm_test_idx[0]  # actual mode
        # mode_d0_gate = svm_test_idx1[0]  # we assume this to be same
        # assert (mode_d0_actual == mode_d0_gate)
        # start_mode = mode_d0_gate
        start_mode = mode_d0_actual

        ugp_dyn = UGP(dX + dU, **ugp_params)
        mc_sample_size = (dX+dU)*10 # TODO: put this param in some proper place
        labels1, counts1 = zip(*sorted(Counter(dpgmm_test_idx).items(), key=operator.itemgetter(0)))
        num_modes = len(labels1)
        modes = labels1
        sim_data_s = {mode: np.zeros((H, dX+dX+dU+dU+1)) for mode in modes}  # x_mu, x_sig, u_mu, u_sigma, p
        sim_data_s[start_mode][0] = np.array([x0, v0, u0, w0, 1.])
        start_time = time.time()
        # for t in range(1, H):
        #     # probabilistic gating
        #     for par_mode in sim_data_s.keys(): # for each mode
        #         par = sim_data_s[par_mode][t-1] # parent node in the time evolving graph
        #         par_p = par[4]    # probability of the parent node
        #         if par_p > 1e-4:
        #             par_x = par[0]    # parent state mean
        #             par_v = par[1]    # parent state var
        #
        #             # action from the policy based on sampled input state
        #             x_t_ = np.random.normal(par_x, np.sqrt(par_v))  # sampled state
        #             # par_u, par_w = sim_1d_sys.get_action(x_t_)
        #             par_u, par_w = sim_1d_sys.get_action(par_x)
        #             sim_data_s[par_mode][t - 1][2] = par_u
        #             sim_data_s[par_mode][t - 1][3] = par_w
        #             par_mu_xu = np.append(par_x, par_u)
        #             par_sigma_xu = np.array([[par_v, 0.],
        #                                     [0., par_w]])
        #
        #             xu_t_s = np.random.multivariate_normal(par_mu_xu, par_sigma_xu, mc_sample_size)
        #             assert(xu_t_s.shape==(mc_sample_size,dX+dU))
        #             xu_t_s_std = scaler1.transform(xu_t_s)
        #             clf1 = SVMs[par_mode]
        #             mode_dst = clf1.predict(xu_t_s_std)
        #             mode_counts = Counter(mode_dst).items()
        #             total_samples = 0
        #             mode_prob = dict(zip(labels1, [0]*len(labels1)))
        #             mode_p = {}
        #             for mod in mode_counts:
        #                 if (par_mode == mod[0]) or ((par_mode, mod[0]) in trans_dicts):
        #                     total_samples = total_samples + mod[1]
        #             for mod in mode_counts:
        #                 if (par_mode == mod[0]) or ((par_mode, mod[0]) in trans_dicts):
        #                     prob = float(mod[1])/float(total_samples)
        #                     mode_p[mod[0]] = prob
        #             mode_prob.update(mode_p)
        #
        #
        #             for child_mode in sim_data_s.keys():  # for each mode
        #                 chd = sim_data_s[child_mode][t:t+1]  # child node in the time evolving graph
        #                 chd_p = mode_prob[child_mode]
        #                 if chd_p > 1e-4:
        #                     mode_ = child_mode
        #                     if child_mode == par_mode:
        #                         gp = MoE_gp[mode_]
        #                         mu_x_t, sigma_x_t, _, _, _ = ugp_dyn.get_posterior(gp, par_mu_xu, par_sigma_xu)
        #                     else:
        #                         # mu_x_t = init_x_table[mode_]['mu']
        #                         # sigma_x_t = init_x_table[mode_]['var']
        #                         # if (par_mode, child_mode) in trans_dicts:
        #                         gp_trans = trans_dicts[(par_mode, child_mode)]['gp']
        #                         mu_x_t, sigma_x_t, _, _, _ = ugp_dyn.get_posterior(gp_trans, par_mu_xu, par_sigma_xu)
        #                     curr_chd_p = chd[0,4]
        #                     mu1 = chd[0,0]
        #                     sig1 = chd[0,1]
        #                     new_chd_p = par_p * chd_p
        #                     mu2 = mu_x_t
        #                     sig2 = sigma_x_t
        #                     tot_chd_p = curr_chd_p + new_chd_p
        #                     w1 = curr_chd_p / tot_chd_p
        #                     w2 = new_chd_p / tot_chd_p
        #                     chd[0,0] = w1 * mu1 + w2 * mu2
        #                     chd[0,1] = w1 * sig1 + w2 * sig2 + w1 * mu1 ** 2 + w2 * mu2 ** 2 - chd[0,0] ** 2
        #                     chd[0,4] = curr_chd_p + new_chd_p
        #     # probability check
        #     prob_mode_tot = 0.
        #     for mode_ in sim_data_s.keys():  # for each mode
        #         track_curr = sim_data_s[mode_][t]
        #         p_curr = track_curr[4]
        #         prob_mode_tot += p_curr
        #     if (prob_mode_tot - 1.0) > 1e-4:
        #         assert(False)
        # print 'Prediction time for MoE UGP with horizon', H, ':', time.time() - start_time
        #
        # # plot long term prediction results of UGP
        # dt = sim_1d_params['dt']
        # tm = np.array(range(H)) * dt
        # mu_X = np.zeros(H)
        # for t in range(H):
        #     xp_pairs = [(sim_data_s[mode_][t][0], sim_data_s[mode_][t][4]) for mode_ in sim_data_s]
        #     xp_max = max(xp_pairs, key=lambda x: x[1])
        #     mu_X[t] = xp_max[0]
        #
        # # prepare for contour plot
        # tm_grid = tm
        # grid_size = 0.2
        # x_grid = np.arange(-1, 16, grid_size)      # TODO: get the ranges from the mode dict
        # Xp, Tp = np.meshgrid(x_grid, tm_grid)
        # prob_map = np.zeros((len(tm_grid), len(x_grid)))
        # prob_limit = np.zeros(len(tm_grid))
        # for t in range(len(tm_grid)):
        #     for mode_ in sim_data_s.keys():
        #         w = sim_data_s[mode_][t][4]
        #         if w > 1e-4:
        #             mu = sim_data_s[mode_][t][0]
        #             sig = np.sqrt(sim_data_s[mode_][t][1])
        #             prob_lt = sp.stats.norm.pdf(mu+1.96*sig, mu, sig)*w
        #             if prob_lt > prob_limit[t]:
        #                 prob_limit[t] = prob_lt
        #
        # for i in range(len(x_grid)):
        #     for t in range(len(tm_grid)):
        #         x = x_grid[i]
        #         for mode_ in sim_data_s.keys():
        #             w = sim_data_s[mode_][t][4]
        #             if w > 1e-4:
        #                 mu = sim_data_s[mode_][t][0]
        #                 sig = np.sqrt(sim_data_s[mode_][t][1])
        #                 prob_val = sp.stats.norm.pdf(x, mu, sig)*w
        #                 prob_map[t, i] += prob_val
        #         # if prob_map[t, i]<prob_limit[t]:
        #         #     prob_map[t, i] = 0.
        # # probability check
        # # print prob_map.sum(axis=1)*grid_size
        #
        # # plt.figure()
        # # plt.title('UGP')
        # # plt.plot(mode_pred_ugp, label='mode_pred')
        # # plt.plot(svm_test_idx1[:H], label='svm_test')
        # # plt.plot(dpgmm_test_idx[:H], label='dpgmm_test')
        # # plt.legend()
        #
        # # compute prediction score UGP
        # start_index = 0
        # # horizon = T  # cannot be > T
        # horizon = H  # cannot be > T
        # end_index = start_index + horizon
        #
        # score_cum = 0.
        # for XUn in XUns_test:
        #     x_m = XUn[start_index:end_index, 0]
        #     x_m.reshape(-1)
        #     traj_score = 0.
        #     for t in range(len(x_m)):
        #         pt = 0.
        #         for mode_ in sim_data_s.keys():
        #             w = sim_data_s[mode_][t][4]
        #             if w > 1e-4:
        #                 xm_ = x_m[t]
        #                 mu_ = sim_data_s[mode_][t][0]
        #                 sig_ = np.sqrt(sim_data_s[mode_][t][1])
        #                 prob_comp = sp.stats.norm.pdf(xm_, mu_, sig_) * w
        #                 pt += prob_comp
        #         traj_score += np.log(pt)
        #     # traj_score_avg = traj_score / float(len(x_m))
        #     # score_cum += traj_score_avg
        #     score_cum += traj_score
        # print 'MoE UGP system prediction test data score:', score_cum / float(XUns_test.shape[0])

        # list based tree structure for multiple track in modes
        sim_data_tree = [[[start_mode, -1, x0, v0, u0, w0, 1.]]]
        for t in range(0, H):
            tracks = sim_data_tree[t]
            for track in tracks:
                md, md_prev, mu_xt, var_xt, _, _, p = track
                xt = np.random.normal(mu_xt, np.sqrt(var_xt))
                mu_ut, var_ut = sim_1d_sys.get_action(xt)
                track[4] = mu_ut
                track[5] = var_ut
                mu_xtut = np.append(mu_xt, mu_ut)
                var_xtut = np.array([[var_xt, 0.],
                                         [0., var_ut]])

                xtut_s = np.random.multivariate_normal(mu_xtut, var_xtut, mc_sample_size)
                assert (xtut_s.shape == (mc_sample_size, dX + dU))
                xtut_s_std = scaler1.transform(xtut_s)
                clf1 = SVMs[md]
                mode_dst = clf1.predict(xtut_s_std)
                mode_counts = Counter(mode_dst).items()
                total_samples = 0
                mode_prob = dict(zip(labels1, [0] * len(labels1)))
                mode_p = {}
                for mod in mode_counts:
                    if (md == mod[0]) or ((md, mod[0]) in trans_dicts):
                        total_samples = total_samples + mod[1]
                for mod in mode_counts:
                    if (md == mod[0]) or ((md, mod[0]) in trans_dicts):
                        prob = float(mod[1]) / float(total_samples)
                        mode_p[mod[0]] = prob
                mode_prob.update(mode_p)
                if len(sim_data_tree) == t + 1:
                    sim_data_tree.append([])        # create the next (empty) time step
                for md_next, p_next in mode_prob.iteritems():
                    if p_next > 1e-4:
                        # get the next state
                        if md_next == md:
                            gp = MoE_gp[md]
                            mu_xt_next_new, var_xt_next_new, _, _, _ = ugp_dyn.get_posterior(gp, mu_xtut, var_xtut)
                        else:
                            gp_trans = trans_dicts[(md, md_next)]['gp']
                            mu_xt_next_new, var_xt_next_new, _, _, _ = ugp_dyn.get_posterior(gp_trans, mu_xtut, var_xtut)
                        assert (len(sim_data_tree) == t + 2)
                        tracks_next = sim_data_tree[t + 1]
                        if md == md_next:
                            md_ = md_prev
                        else:
                            md_ = md
                        if len(tracks_next)==0:
                            if p*p_next > 1e-4:
                                sim_data_tree[t+1].append([md_next, md_, mu_xt_next_new, var_xt_next_new, 0., 0., p*p_next])
                        else:
                            md_next_curr_list = [track_next[0] for track_next in tracks_next]
                            if md_next not in md_next_curr_list:
                                # md_next not already in the t+1 time step
                                if p * p_next > 1e-4:
                                    sim_data_tree[t + 1].append(
                                        [md_next, md_, mu_xt_next_new, var_xt_next_new, 0., 0., p * p_next])
                            else:
                                # md_next already in the t+1 time step
                                if md == md_next:
                                    md_ = md_prev
                                else:
                                    md_ = md
                                md_next_curr_trans_list = [(track_next[1], track_next[0]) for track_next in tracks_next]
                                if (md_, md_next) not in md_next_curr_trans_list:
                                    # the same transition track is not present
                                    if p * p_next > 1e-4:
                                        sim_data_tree[t + 1].append(
                                            [md_next, md_, mu_xt_next_new, var_xt_next_new, 0., 0., p * p_next])
                                else:
                                    it = 0
                                    for track_next in tracks_next:
                                        md_next_curr, md_prev_curr, mu_xt_next_curr, var_xt_next_curr, _, _, p_next_curr = track_next
                                        if md_next == md_next_curr:
                                            next_trans = (md_, md_next)
                                            curr_trans = (md_prev_curr, md_next_curr)
                                            if curr_trans == next_trans:
                                                p_next_new = p*p_next
                                                tot_new_p = p_next_curr + p_next_new
                                                w1 = p_next_curr / tot_new_p
                                                w2 = p_next_new / tot_new_p
                                                mu_next_comb = w1 * mu_xt_next_curr + w2 * mu_xt_next_new
                                                var_next_comb = w1 * var_xt_next_curr + w2 * var_xt_next_new + \
                                                                w1 * mu_xt_next_curr ** 2 + w2 * mu_xt_next_new ** 2 - mu_next_comb ** 2
                                                p_next_comb = p_next_curr + p_next_new
                                                if p_next_comb > 1e-4:
                                                    sim_data_tree[t + 1][it] = [md_next, md_, mu_next_comb, var_next_comb, 0., 0., p_next_comb]
                                        it+=1

            # probability check
            prob_mode_tot = 0.
            for track_ in sim_data_tree[t]:
                    prob_mode_tot += track_[6]
            if (prob_mode_tot - 1.0) > 1e-4:
                assert (False)

        print 'Prediction time for MoE UGP with horizon', H, ':', time.time() - start_time



        # plot for tree structure
        # plot long term prediction results of UGP
        dt = sim_1d_params['dt']
        tm = np.array(range(H)) * dt
        mu_X = np.zeros(H)
        for t in range(H):
            tracks = sim_data_tree[t]
            xp_pairs = [[track[2], track[6]] for track in tracks]
            xp_max = max(xp_pairs, key=lambda x: x[1])
            mu_X[t] = xp_max[0]

        # prepare for contour plot
        tm_grid = tm
        grid_size = 0.2
        x_grid = np.arange(-1, 16, grid_size)  # TODO: get the ranges from the mode dict
        Xp, Tp = np.meshgrid(x_grid, tm_grid)
        prob_map = np.zeros((len(tm_grid), len(x_grid)))

        for i in range(len(x_grid)):
            for t in range(len(tm_grid)):
                x = x_grid[i]
                tracks = sim_data_tree[t]
                for track in tracks:
                    w = track[6]
                    if w > 1e-4:
                        mu = track[2]
                        sig = np.sqrt(track[3])
                        prob_val = sp.stats.norm.pdf(x, mu, sig) * w
                        prob_map[t, i] += prob_val
                # if prob_map[t, i]<prob_limit[t]:
                #     prob_map[t, i] = 0.
        # probability check
        print prob_map.sum(axis=1)*grid_size


        min_prob_den = min_prob_grid / grid_size
        plt.figure()
        plt.title('Long-term prediction with mixture of GP')
        plt.xlabel('Time, t')
        plt.ylabel('State, x(t)')
        plt.plot(tm, mu_X, color='b', ls='-', marker='s', linewidth='2', label='Learned dynamics', markersize=7)
        plt.contourf(Tp, Xp, prob_map, colors='b', alpha=.2, levels=[min_prob_den, 10.]) #TODO: levels has to properly set according to some confidence interval
        # plt.contourf(Tp, Xp, prob_map, 20, cmap='Blues',
        #              alpha=1., vmin=1e-4)
        # plt.contourf(Tp, Xp, prob_map, colors='b', alpha=.2,
        #              levels=[1e-4, 4.])
        plt.plot(tm, traj_gt[:H, 1], color='g', ls='-', marker='^', linewidth='2', label='True dynamics', markersize=7)
        # for i in range(sigmaOp.shape[0]):
        #     plt.scatter(tm, sigmaOps[:,i], marker='+', color='k')
        for XUn in XUns_test:
            plt.plot(tm, XUn[:H, 0])
        # plt.colorbar()
        plt.legend()
        plt.savefig('ME_long-term_ugp.pdf')
        plt.savefig('ME_long-term_ugp.png', format='png', dpi=1000)
        # plt.show()

        # plt.figure()
        # plt.title('UGP')
        # plt.plot(mode_pred_ugp, label='mode_pred')
        # plt.plot(svm_test_idx1[:H], label='svm_test')
        # plt.plot(dpgmm_test_idx[:H], label='dpgmm_test')
        # plt.legend()

        # compute prediction score UGP
        start_index = 0
        # horizon = T  # cannot be > T
        horizon = H  # cannot be > T
        end_index = start_index + horizon

        score_cum = 0.
        for XUn in XUns_test:
            x_m = XUn[start_index:end_index, 0]
            x_m.reshape(-1)
            traj_score = 0.
            for t in range(len(x_m)):
                pt = 0.
                tracks = sim_data_tree[t]
                for track in tracks:
                    w = track[6]
                    if w > 1e-4:
                        xm_ = x_m[t]
                        mu_ = track[2]
                        sig_ = np.sqrt(track[3])
                        prob_comp = sp.stats.norm.pdf(xm_, mu_, sig_) * w
                        pt += prob_comp
                traj_score += np.log(pt)
            # traj_score_avg = traj_score / float(len(x_m))
            # score_cum += traj_score_avg
            score_cum += traj_score
        print 'MoE UGP system prediction test data score:', score_cum / float(XUns_test.shape[0])

plt.show()

None
