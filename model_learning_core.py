import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mat_col
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from utilities import get_N_HexCol
from collections import Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from multidim_gp import MultidimGP
from model_leraning_utils import UGP
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from copy import deepcopy
import operator
import datetime
import time
from itertools import compress
import pickle
from blocks_sim import MassSlideWorld

# np.random.seed(6)

blocks_exp = True
mjc_exp = False
yumi_exp = False

global_gp = True
load_gp = True
cluster = True
fit_moe = True

if blocks_exp:
    exp_data = pickle.load( open("./Results/blocks_exp_preprocessed_data.p", "rb" ) )

exp_params = exp_data['exp_params']
Xg = exp_data['Xg']  # sate ground truth
Ug = exp_data['Ug']  # action ground truth
dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']
dX = dP+dV
T = exp_params['T']
dt = exp_params['dt']
n_train = exp_data['n_train']
n_test = exp_data['n_test']

XU_t_train = exp_data['XU_t_train']
dX_t_train = exp_data['dX_t_train']
XUs_train = exp_data['XUs_train']
X_t1_train = exp_data['X_t1_train']
Xs_train = XUs_train[:, :, :dX]

if global_gp:
    # global gp fit
    if not load_gp:
        noise_lower = 1e-3
        noise_upper = 1e-2
        gpr_params = {
            'alpha': 0.,  # alpha=0 when using white kernal
            'kernel': C(1.0, (1e-2, 1e2)) * RBF(np.ones(dX + dU), (1e-3, 1e3)) + W(noise_level=1.,
                                                                                   noise_level_bounds=(noise_lower, noise_upper)),
            'n_restarts_optimizer': 10,
            'normalize_y': False,
        }

        mdgp_glob = MultidimGP(gpr_params, dX)
        start_time = time.time()
        mdgp_glob.fit(XU_t_train, X_t1_train)
        print 'Global GP fit time', time.time() - start_time
        exp_data['mdgp_glob'] = deepcopy(mdgp_glob)
        pickle.dump(exp_data, open("./Results/blocks_exp_preprocessed_data.p", "wb"))
    else:
        if 'mdgp_glob' not in exp_data:
            assert(False)
        else:
            mdgp_glob = exp_data['mdgp_glob']


    # global gp long-term prediction
    H = T       # prediction horizon
    if blocks_exp:
        expl_noise = 5.
        massSlideParams = exp_params['massSlide']
        # policy_params = exp_params['policy']
        policy_params = {
                            'm1': {
                                'L': np.array([.2, 1.]),
                                # 'noise': 7.5*2,
                                'noise': expl_noise,
                                'target': 18.,
                                },
                        } # TODO: the block_sim code assumes only 'm1' mode for control
        massSlideWorld = MassSlideWorld(**massSlideParams)
        massSlideWorld.set_policy(policy_params)
        massSlideWorld.reset()
        mode = 'm1'  # only one mode for control no matter what X

    # ugp_params = {
    #     'alpha': 1.,
    #     'kappa': 2.,
    #     'beta': 0.,
    # }
    ugp_params = {
        'alpha': 1.,
        'kappa': 0.1,
        'beta': 0.,
    }
    ugp_global_dyn = UGP(dX + dU, **ugp_params)
    ugp_global_pol = UGP(dX, **ugp_params)

    x_mu_t = exp_data['X0_mu']
    # x_mu_t = exp_data['X0_mu'] + 0.5
    x_var_t = np.diag(exp_data['X0_var'])
    # x_var_t[0, 0] = 1e-6
    x_var_t[1,1] = 1e-6       # TODO: cholesky failing for zero v0 variance
    X_mu_pred = []
    X_var_pred = []
    X_particles = []
    start_time = time.time()
    for t in range(H):
        x_t = np.random.multivariate_normal(x_mu_t, x_var_t)
        if blocks_exp:
            # _, u_mu_t, u_var_t = massSlideWorld.act(x_t, mode)
            # _, u_mu_t, u_var_t = massSlideWorld.act(x_mu_t, mode)
            u_mu_t, u_var_t, _, _, _ = ugp_global_pol.get_posterior(massSlideWorld, x_mu_t, x_var_t)
        xu_mu_t = np.append(x_mu_t, u_mu_t)
        xu_var_t = np.block([[x_var_t, np.zeros((dX,dU))],
                            [np.zeros((dU,dX)), u_var_t]])
        X_mu_pred.append(x_mu_t)
        X_var_pred.append(x_var_t)
        x_mu_t, x_var_t, Y_mu, _, _ = ugp_global_dyn.get_posterior(mdgp_glob, xu_mu_t, xu_var_t)
        X_particles.append(Y_mu)
    print 'Prediction time for horizon UGP', H, ':', time.time() - start_time

    # compute long-term prediction score
    XUs_test = exp_data['XUs_test']
    assert(XUs_test.shape[0]==n_test)
    X_test_log_ll = np.zeros((H, n_test))
    for t in range(H):      # one data point less than in XU_test
        for i in range(n_test):
            XU_test = XUs_test[i]
            x_t = XU_test[t, :dX]
            x_mu_t = X_mu_pred[t]
            x_var_t = X_var_pred[t]
            X_test_log_ll[t, i] = sp.stats.multivariate_normal.logpdf(x_t, x_mu_t, x_var_t)

    tm = np.array(range(H)) * dt
    plt.figure()
    plt.title('Average NLL of test trajectories w.r.t time ')
    plt.xlabel('Time, t')
    plt.ylabel('NLL')
    plt.plot(tm.reshape(H,1), X_test_log_ll)

    nll_mean = np.mean(X_test_log_ll.reshape(-1))
    nll_std = np.std(X_test_log_ll.reshape(-1))
    print 'NLL mean: ', nll_mean, 'NLL std: ', nll_std


    if blocks_exp:
        X_mu_pred = np.array(X_mu_pred)
        P_sig_pred = np.zeros(H)
        V_sig_pred = np.zeros(H)
        P_sigma_points = np.zeros((2*(dX+dU) + 1,H))
        V_sigma_points = np.zeros((2 * (dX+dU) + 1, H))
        for t in range(H):
            # X_mu_pred[t] = X_mu_pred[t-1] + X_mu_pred[t]
            # X_var_pred[t] = X_var_pred[t-1] + X_var_pred[t]
            P_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[0])
            V_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[1])

        P_mu_pred = X_mu_pred[:, :dP].reshape(-1)
        V_mu_pred = X_mu_pred[:, dP:].reshape(-1)

        for t in range(1,H):
            P_sigma_points[:, t] = X_particles[t-1][:, 0]
            V_sigma_points[:, t] = X_particles[t-1][:, 1]

        tm = np.array(range(H)) * dt
        plt.figure()
        plt.title('Long-term prediction with GP')
        plt.subplot(121)
        plt.xlabel('Time, t')
        plt.ylabel('Position, m')
        plt.plot(tm, P_mu_pred)
        plt.fill_between(tm, P_mu_pred - P_sig_pred * 1.96, P_mu_pred + P_sig_pred * 1.96, alpha=0.2)
        plt.plot(tm, Xg[:,0], linewidth='2')
        for i in range(n_train):
            plt.plot(tm, Xs_train[i, :, :dP], alpha=0.3)
        for p in P_sigma_points:
            plt.scatter(tm, p, marker='+')
        plt.subplot(122)
        plt.xlabel('Time, t')
        plt.ylabel('Velocity, m/s')
        plt.plot(tm, V_mu_pred)
        plt.fill_between(tm, V_mu_pred - V_sig_pred * 1.96, V_mu_pred + V_sig_pred * 1.96, alpha=0.2)
        plt.plot(tm, Xg[:, 1], linewidth='2')
        for i in range(n_train):
            plt.plot(tm, Xs_train[i, :, dP:], alpha=0.3)
        for p in V_sigma_points:
            plt.scatter(tm, p, marker='+')

plt.show()
None
#



