import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mat_col
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from model_leraning_utils import get_N_HexCol
from model_leraning_utils import train_trans_models
from model_leraning_utils import train_SVM_models
from model_leraning_utils import DPGMMCluster
from collections import Counter
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
# from multidim_gp import MultidimGP
from multidim_gp import MdGpyGP as MultidimGP
# from multidim_gp import MdGpySparseGP as MultidimGP
from model_leraning_utils import UGP
from model_leraning_utils import dummySVM
from model_leraning_utils import YumiKinematics
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
from copy import deepcopy
import operator
#import datetime
import time
from itertools import compress
import pickle
from blocks_sim import MassSlideWorld
from mjc_exp_policy import Policy, exp_params_rob, kin_params
from mjc_exp_policy import SimplePolicy
import copy

np.random.seed(1)     # good value for clustering new yumi exp

# logfile = "./Results/yumi_exp_preprocessed_data_2.p"
# logfile = "./Results/yumi_peg_exp_preprocessed_data_1.p"
# logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train.p"    # includes a trained global gp
# logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train_1.p"      # new yumi exp
# logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train_2.p"      # global gp trained and lt pred working with simple policy
logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train_3.p"
# logfile = "./Results/mjc_exp_2_sec_raw_preprocessed.p"

vbgmm_refine = False

global_gp = False
delta_model = False
fit_moe = True

load_all = False
load_gp = True
load_dpgmm = load_all
load_transition_gp = load_all
load_experts = load_all
load_svms = load_all
load_global_lt_pred = load_all


min_prob_grid = 0.001 # 1%
grid_size = 0.005
min_counts = 5 # min number of cluster size.

exp_data = pickle.load( open(logfile, "rb" ) )

exp_params = exp_data['exp_params']
dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']
dX = dP+dV
dEP = 6
dEV = 6
dEX = 12
dF = 6
T = exp_params['T'] - 1
dt = exp_params['dt']
n_train = exp_data['n_train']
# n_test = exp_data['n_test']-1 # TODO: remove -1 this is done to fix a bug in the logfile but already fixed in the code.
n_test = exp_data['n_test']
# data set for joint space
XUs_t_train = exp_data['XUs_t_train']
XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1])
XU_scaler = StandardScaler().fit(XU_t_train)
XU_t_std_train = XU_scaler.transform(XU_t_train)
# XU_t_train_avg = np.mean(XUs_t_train, axis=0)
XU_t_train_avg = exp_data['XU_t_train_avg']
# Xs_t1_train_avg = exp_data['Xs_t1_train_avg']
Xrs_t_train = exp_data['Xrs_t_train']

Xs_t_train = exp_data['Xs_t_train']
X_t_train = Xs_t_train.reshape(-1, Xs_t_train.shape[-1])
X_scaler = StandardScaler().fit(X_t_train)
X_t_std_train = X_scaler.transform(X_t_train)
w_vel = 1.0
X_t_std_weighted_train = X_t_std_train
X_t_std_weighted_train[:, dP:dP+dV] = X_t_std_weighted_train[:, dP:dP+dV] * w_vel

Xs_t1_train = exp_data['Xs_t1_train']
X_t1_train = Xs_t1_train.reshape(-1, Xs_t1_train.shape[-1])
X_t1_std_train = X_scaler.transform(X_t1_train)
X_t1_std_weighted_train = X_t1_std_train
X_t1_std_weighted_train[:, dP:dP+dV] = X_t1_std_weighted_train[:, dP:dP+dV] * w_vel

dX_t_train = X_t1_train - X_t_train

EXs_t_train = exp_data['EXs_t_train']
EX_t_train = EXs_t_train.reshape(-1, EXs_t_train.shape[-1])

XUs_t_test = exp_data['XUs_t_test']

# data set for cartesian space
EXFs_train = exp_data['EXFs_train']
EXF_train = EXFs_train.reshape(-1, EXFs_train.shape[-1])
EX_train = EXF_train[:, :dEX]
EX_scaler = StandardScaler().fit(EX_train)
EX_std_train = EX_scaler.transform(EX_train)
EX_std_train[:,dEP:dEP+dEV] = EX_std_train[:,dEP:dEP+dEV]*2.0

EXFs_t_train = exp_data['EXFs_t_train']
EXF_t_train = EXFs_t_train.reshape(-1, EXFs_t_train.shape[-1])
EXF_scaler = StandardScaler().fit(EXF_t_train)
EXF_t_std_train = EXF_scaler.transform(EXF_t_train)

EXs_t_train = exp_data['EXs_t_train']
EX_t_train = EXs_t_train.reshape(-1, EXs_t_train.shape[-1])
EX_t_scaler = StandardScaler().fit(EX_t_train)
EX_t_std_train = EX_t_scaler.transform(EX_t_train)
w_vel = 1.0
EX_t_std_weighted_train = EX_t_std_train
EX_t_std_weighted_train[:, dEP:dEP+dEV] = EX_t_std_weighted_train[:, dEP:dEP+dEV] * w_vel

EXs_t1_train = exp_data['EXs_t1_train']
EX_t1_train = EXs_t1_train.reshape(-1, EXs_t1_train.shape[-1])
EX_t1_std_train = EX_t_scaler.transform(EX_t1_train)
EX_t1_std_weighted_train = EX_t1_std_train
EX_t1_std_weighted_train[:, dEP:dEP+dEV] = EX_t1_std_weighted_train[:, dEP:dEP+dEV] * w_vel

dEX_t_train = EX_t1_train - EX_t_train
Fs_t_train = exp_data['Fs_t_train']
F_t_train = Fs_t_train.reshape(-1, Fs_t_train.shape[-1])
EX_1_EX_F_t_train = np.concatenate((EX_t1_train, EX_t_train, F_t_train), axis=1)
EX_1_EX_F_scaler = StandardScaler().fit(EX_1_EX_F_t_train)
EX_1_EX_F_t_std_train = EX_1_EX_F_scaler.transform(EX_1_EX_F_t_train)

Us_t_train = exp_data['Us_t_train']
U_t_train = Us_t_train.reshape(-1, Us_t_train.shape[-1])
EXU_t_train = np.concatenate((EX_t_train, U_t_train), axis=1)
EXU_scaler = StandardScaler().fit(EXU_t_train)
EXU_t_std_train = EXU_scaler.transform(EXU_t_train)

EXFs_t_test = exp_data['EXFs_t_test']


ugp_params = {
    'alpha': 1.,
    'kappa': 2.,
    'beta': 0.,
}

agent_hyperparams = {
    'dt': 0.05,
    'T': 50,
    'smooth_noise': False,
    'smooth_noise_var': 1.,
    'smooth_noise_renormalize': False
}

# exp_params['joint_space_noise'] = 0.

# gpr_params = {
#         'alpha': 0.,  # alpha=0 when using white kernal
#         'kernel': C(1.0, (1e-2, 1e2)) * RBF(np.ones(dX + dU), (1e-2, 1e2)) + W(noise_level=1.,
#                                                                                noise_level_bounds=(1e-4, 1e1)),
#         'n_restarts_optimizer': 10,
#         'normalize_y': False,  # is not supported in the propogation function
#     }
p_noise_var = np.full(7, 6.25e-4)
v_noise_var = np.full(7, 6.25e-2)
# p_noise_var = np.full(7, 6.25e-6)
# v_noise_var = np.full(7, 6.25e-6)
gpr_params = {
        'noise_var': np.concatenate((p_noise_var, p_noise_var)),
        'normalize': True,
    }
# expl_noise = 3.
H = T  # prediction horizon

if global_gp:
    gpr_params_list = []
    for i in range(dX):
        gpr_params_list.append(gpr_params)

    # global gp fit
    if not load_gp:
        mdgp_glob = MultidimGP(gpr_params_list, dX)
        start_time = time.time()
        if not delta_model:
            mdgp_glob.fit(XU_t_train, X_t1_train)
        else:
            mdgp_glob.fit(XU_t_train, dX_t_train)
        print 'Global GP fit time', time.time() - start_time
        exp_data['mdgp_glob'] = deepcopy(mdgp_glob)
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'mdgp_glob' not in exp_data:
            assert(False)
        else:
            mdgp_glob = exp_data['mdgp_glob']

    if not load_global_lt_pred:
        # global gp long-term prediction
        pol = Policy(agent_hyperparams, exp_params_rob)
        # pol1 = Policy(agent_hyperparams, exp_params_rob)
        # pol2 = SimplePolicy(Xrs_t_train, Us_t_train, exp_params_rob)
        sim_pol = SimplePolicy(Xrs_t_train, Us_t_train, exp_params_rob)

        ugp_global_dyn = UGP(dX + dU, **ugp_params)
        ugp_global_pol = UGP(dX, **ugp_params)

        x_mu_t = exp_data['X0_mu']
        # x_mu_t = exp_data['X0_mu'] + 0.5
        x_var_t = np.diag(exp_data['X0_var'])
        # x_var_t[0, 0] = 1e-6
        # x_var_t[1,1] = 1e-6       # TODO: cholesky failing for zero v0 variance
        Y_mu = np.zeros((2*(dX + dU) + 1, dX))
        X_mu_pred = []
        X_var_pred = []
        U_mu_pred = []
        U_mu_pred_x_avg = []
        U2_mu_pred = []
        U_mu_pred_sp = []
        U_var_pred = []
        U_mu_pred_avg = []
        X_particles = []
        start_time = time.time()
        for t in range(H):
            # standard case
            x_t = np.random.multivariate_normal(x_mu_t, x_var_t)
            # u_mu_t, u_var_t, _, _, xu_cov = ugp_global_pol.get_posterior(pol, x_mu_t, x_var_t, t)
            u_mu_t, u_var_t, _, _, xu_cov = ugp_global_pol.get_posterior(sim_pol, x_mu_t, x_var_t, t)
            U_mu_pred.append(u_mu_t)
            U_var_pred.append(u_var_t)
            X_mu_pred.append(x_mu_t)
            X_var_pred.append(x_var_t)
            # X_particles.append(Y_mu)
            # xu_mu_t = np.append(x_mu_t, u_mu_t)
            xu_var_t = np.block([[x_var_t, xu_cov],
                                 [xu_cov.T, u_var_t]])

            # simple policy evaluation
            _, u_mu_t_sp = sim_pol.act(x_mu_t, t)
            U_mu_pred_sp.append(u_mu_t_sp)


            # fix u with mean u data
            u_mu_t_avg = XU_t_train_avg[t, dX:dX + dU]
            U_mu_pred_avg.append(u_mu_t_avg)
            xu_mu_t = np.append(x_mu_t, u_mu_t_avg)

            # to test the policy with mean state data, the action should correspond to mean action data
            # x_mu_t_avg = XU_t_train_avg[t, :dX]
            ############ TODO: remove after debugging
            # x_mu_t_avg = np.array([-1.3048, -1.35466, 0.947929, 0.317889, 2.06793, 1.49044, -2.14021, 0.000531959, 0.00055548, -0.000337065, -7.55786e-05, 0.00385989, -0.000255539, -0.00792514])
            ############
            # u_mu_t_x_avg, _ = pol1.predict(x_mu_t_avg.reshape(1, -1), t)
            # u_mu_t_x_avg = pol1.act(x_mu_t_avg, None, t, noise=None)
            # u_mu_t_x_avg = u_mu_t_x_avg.reshape(-1)
            # U_mu_pred_x_avg.append(u_mu_t_x_avg)

            if not delta_model:
                x_mu_t, x_var_t, Y_mu, _, _ = ugp_global_dyn.get_posterior(mdgp_glob, xu_mu_t, xu_var_t)
            else:
                dx_mu_t, dx_var_t, dY_mu, _, xudx_covar = ugp_global_dyn.get_posterior(mdgp_glob, xu_mu_t, xu_var_t)
                xdx_covar = xudx_covar[:dX, :]
                x_mu_t = X_mu_pred[t] + dx_mu_t
                x_var_t = X_var_pred[t] + dx_var_t + xdx_covar + xdx_covar.T
                # Y_mu = X_particles[t] + dY_mu
        print 'Global GP prediction time for horizon', H, ':', time.time() - start_time
        exp_data['global_lt_pred'] = {'X_mu_pred': X_mu_pred, 'X_var_pred': X_var_pred, 'U_mu_pred': U_mu_pred,'X_particles': X_particles}
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'global_lt_pred' not in exp_data:
            assert (False)
        else:
            X_mu_pred = exp_data['global_lt_pred']['X_mu_pred']
            X_var_pred = exp_data['global_lt_pred']['X_var_pred']
            U_mu_pred = exp_data['global_lt_pred']['U_mu_pred']
            # X_particles = exp_data['global_lt_pred']['X_particles']




    # compute long-term prediction score
    XUs_t_test = exp_data['XUs_t_test']
    assert(XUs_t_test.shape[0]==n_test)
    X_test_log_ll = np.zeros((H, n_test))
    for t in range(H):      # one data point less than in XU_test
        for i in range(n_test):
            XU_test = XUs_t_test[i]
            x_t = XU_test[t, :dX]
            x_mu_t = X_mu_pred[t]
            x_var_t = X_var_pred[t]
            X_test_log_ll[t, i] = sp.stats.multivariate_normal.logpdf(x_t, x_mu_t, x_var_t)

    tm = np.array(range(H)) * dt
    # plt.figure()
    # plt.title('Average NLL of test trajectories w.r.t time ')
    # plt.xlabel('Time, t')
    # plt.ylabel('NLL')
    # plt.plot(tm.reshape(H,1), X_test_log_ll)

    nll_mean = np.mean(X_test_log_ll.reshape(-1))
    nll_std = np.std(X_test_log_ll.reshape(-1))
    print 'NLL mean: ', nll_mean, 'NLL std: ', nll_std

    # plot long-term prediction
    X_mu_pred = np.array(X_mu_pred)
    U_mu_pred = np.array(U_mu_pred)
    U_mu_pred_x_avg = np.array(U_mu_pred_x_avg)
    # U2_mu_pred = np.array(U2_mu_pred)
    U_mu_pred_avg = np.array(U_mu_pred_avg)
    U_mu_pred_sp = np.array(U_mu_pred_sp)

    P_mu_pred = X_mu_pred[:, :dP]
    V_mu_pred = X_mu_pred[:, dP:]
    P_sig_pred = np.zeros((H,dP))
    V_sig_pred = np.zeros((H,dV))
    U_sig_pred = np.zeros((H,dU))
    P_sigma_points = np.zeros((H, 2 * (dX + dU) + 1, dP))
    V_sigma_points = np.zeros((H, 2 * (dX + dU) + 1, dV))
    for t in range(H):
        P_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[:dP])
        V_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[dP:])
        U_sig_pred[t] = np.sqrt(np.diag(U_var_pred[t]))
        # P_sigma_points[t, :, :] = X_particles[t][:, :dP]
        # V_sigma_points[t, :, :] = X_particles[t][:, dP:]

    # tm = np.array(range(H)) * dt
    tm = np.array(range(H))
    plt.figure()
    plt.title('Long-term prediction with GP')
    # jPos
    for j in range(dP):
        plt.subplot(3, 7, 1 + j)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint Pos (rad)')
        plt.title('j%dPos' % (j + 1))
        plt.plot(tm, Xs_t_train[:, :H, j].T, alpha=0.2)
        plt.plot(tm, P_mu_pred[:H, j], color='g', marker='s', markersize=2,)
        plt.fill_between(tm, P_mu_pred[:H, j] - P_sig_pred[:H, j] * 1.96, P_mu_pred[:H, j] + P_sig_pred[:H, j] * 1.96, alpha=0.2, color='g')
    # jVel
    for j in range(dV):
        plt.subplot(3, 7, 8 + j)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint Vel (rad/s)')
        plt.title('j%dVel' % (j + 1))
        plt.plot(tm, Xs_t_train[:, :H, dP+j].T, alpha=0.2)
        plt.plot(tm, V_mu_pred[:H, j], color='b', marker='s', markersize=2,)
        plt.fill_between(tm, V_mu_pred[:H, j] - V_sig_pred[:H, j] * 1.96, V_mu_pred[:H, j] + V_sig_pred[:H, j] * 1.96,
                         alpha=0.2, color='b')
    for j in range(dV):
        plt.subplot(3, 7, 15 + j)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint Trq (Nm)')
        plt.title('j%dTrq' % (j + 1))
        plt.plot(tm, XUs_t_train[:, :H, dX+j].T, alpha=0.2)
        plt.plot(tm, U_mu_pred[:H, j], color='r', marker='s', markersize=2, label='mean pred')
        plt.fill_between(tm, U_mu_pred[:H, j] - U_sig_pred[:H, j] * 1.96, U_mu_pred[:H, j] + U_sig_pred[:H, j] * 1.96,
                         alpha=0.2, color='r')
        plt.plot(tm, U_mu_pred_avg[:H, j], color='r', linestyle='--', label='mean data')
        # plt.plot(tm, U_mu_pred_x_avg[:H, j], color='r', linestyle='-.', label='avg state based')
        plt.plot(tm, U_mu_pred_sp[:H, j], color='r', linestyle='-.', label='simple pol')
    plt.legend()
    plt.show(block=False)


if fit_moe:
    if not load_dpgmm:
        # K = X_t_std_weighted_train.shape[0] // 3
        K = 20
        dpgmm_params = {
            'n_components': K,  # cluster size
            'covariance_type': 'full',
            'tol': 1e-6,
            'n_init': 10,
            'max_iter': 200,
            'weight_concentration_prior_type': 'dirichlet_process',
            'weight_concentration_prior':1e-3,
            'mean_precision_prior':None,
            'mean_prior': None,
            'degrees_of_freedom_prior': 2+dEX,
            'covariance_prior': None,
            'warm_start': False,
            'init_params': 'random',
            'verbose': 1
        }
        dpgmm = mixture.BayesianGaussianMixture(**dpgmm_params)
        start_time = time.time()
        # dpgmm.fit(X_t_std_weighted_train)
        # dpgmm.fit(X_t_train)
        # dpgmm.fit(EX_t_train)
        dpgmm.fit(EX_std_train)
        # dpgmm.fit(EX_1_EX_F_t_std_train)
        print 'DPGMM clustering time:', time.time() - start_time
        print 'Converged DPGMM', dpgmm.converged_, 'on', dpgmm.n_iter_, 'iterations with lower bound', dpgmm.lower_bound_
        exp_data['dpgmm'] = deepcopy(dpgmm)
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'dpgmm' not in exp_data:
            assert (False)
        else:
            dpgmm = exp_data['dpgmm']
    dpgmm_EX_train_labels = dpgmm.predict(EX_std_train)
    log_prob = dpgmm._estimate_weighted_log_prob(EX_std_train)
    labels, counts = zip(*sorted(Counter(dpgmm_EX_train_labels).items(), key=operator.itemgetter(0)))
    for i in range(len(counts)):
        if counts[i] < min_counts:
            array_idx_label = (dpgmm_EX_train_labels == labels[i])
            log_prob_label = log_prob[array_idx_label]
            reassigned_labels = np.zeros(log_prob_label.shape[0])
            for j in range(log_prob_label.shape[0]):
                sorted_idx = np.argsort(log_prob_label[j, :])
                reassigned_labels[j] = int(sorted_idx[-2])
            dpgmm_EX_train_labels[array_idx_label] = reassigned_labels

    dpgmm_EXs_train_labels = dpgmm_EX_train_labels.reshape(n_train,-1)
    dpgmm_EXs_t_train_labels = dpgmm_EXs_train_labels[:, :-1]
    dpgmm_EXs_t1_train_labels = dpgmm_EXs_train_labels[:, 1:]
    dpgmm_EX_t_train_labels = dpgmm_EXs_t_train_labels.reshape(-1)
    dpgmm_EX_t1_train_labels = dpgmm_EXs_t1_train_labels.reshape(-1)
    # dpgmm_Xt_train_labels = dpgmm.predict(EX_t_std_weighted_train)
    # dpgmm_Xt_train_labels = dpgmm.predict(EX_1_EX_F_t_train)
    # dpgmm_Xt1_train_labels = dpgmm.predict(X_t1_std_weighted_train)

    labels, counts = zip(*sorted(Counter(dpgmm_EX_t_train_labels).items(), key=operator.itemgetter(0)))
    K = len(labels)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors) / 255.
    if 'clust_result' not in exp_data:
        exp_data['clust_result'] = {'assign': dpgmm_EXs_t_train_labels, 'labels': labels, }
        pickle.dump(exp_data, open(logfile, "wb"))
    # ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.bar(labels, counts, color=colors)
    # plt.title('DPGMM clustering')
    # plt.ylabel('Cluster sizes')
    # plt.xlabel('Cluster labels')
    # plt.savefig('dpgmm_blocks_cluster counts.pdf')
    # plt.savefig('dpgmm_1d_dyn_cluster counts.png', format='png', dpi=1000)

    # pi = dpgmm.dpgmm.weights_
    # ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.bar(labels, pi[list(labels)], color=colors)
    # plt.title('DPGMM clustering')
    # plt.ylabel('Cluster weights')
    # plt.xlabel('Cluster labels')

    # plot clustered trajectory

    col = np.zeros([EX_t_train.shape[0], 3])
    i = 0
    for label in labels:
        col[(dpgmm_EX_t_train_labels == label)] = colors[i]
        i += 1
    cols = col.reshape(n_train, T, -1)
    label_col_dict = d = dict(zip(labels, colors))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter3D(EX_t_train[:,0], EX_t_train[:,1], EX_t_train[:,2], c=col)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('DPGMM clustering')
    plt.show(block=False)

    if not load_transition_gp:
        # transition GP
        trans_gpr_params = gpr_params
        trans_gp_param_list = []
        for i in range(dX):
            trans_gp_param_list.append(gpr_params)
        trans_dicts = train_trans_models(trans_gp_param_list, XUs_t_train, dpgmm_EXs_t_train_labels, dX, dU)
        exp_data['transition_gp'] = deepcopy(trans_dicts)
        pickle.dump(exp_data, open(logfile, "wb"))

    else:
        if 'transition_gp' not in exp_data:
            assert(False)
        else:
            trans_dicts = exp_data['transition_gp']

    if not load_experts:
        # expert training
        expert_gpr_params = gpr_params
        # expert_gpr_params = {
        #     # 'alpha': 1e-2,  # alpha=0 when using white kernal
        #     'alpha': 0.,  # alpha=0 when using white kernal
        #     'kernel': C(1.0, (1e-2, 1e2)) * RBF(np.ones(dX + dU), (1e-1, 1e1)) + W(noise_level=1.,
        #                                                                            noise_level_bounds=(1e-4, 1e-2)),
        #     # 'kernel': C(1.0, (1e-1, 1e1)) * RBF(np.ones(dX + dU), (1e-1, 1e1)),
        #     'n_restarts_optimizer': 10,
        #     'normalize_y': False,  # is not supported in the propogation function
        # }
        expert_gp_param_list = []
        for i in range(dX):
            expert_gp_param_list.append(gpr_params)
        experts = {}
        start_time = time.time()
        for label in labels:
            x_train = XU_t_train[(np.logical_and((dpgmm_EX_t_train_labels == label), (dpgmm_EX_t1_train_labels == label)))]
            y_train = X_t1_train[(np.logical_and((dpgmm_EX_t_train_labels == label), (dpgmm_EX_t1_train_labels == label)))]
            mdgp = MultidimGP(expert_gp_param_list, y_train.shape[1])
            mdgp.fit(x_train, y_train)
            experts[label] = deepcopy(mdgp)
            del mdgp
        print 'Experts training time:', time.time() - start_time
        exp_data['experts'] = deepcopy(experts)
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'experts' not in exp_data:
            assert(False)
        else:
            experts = exp_data['experts']

    if not load_svms:
        # gating network training
        svm_grid_params = {
                            'param_grid': {"C": np.logspace(-10, 10, endpoint=True, num=11, base=2.),
                                           "gamma": np.logspace(-10, 10, endpoint=True, num=11, base=2.)},
                            'scoring': 'accuracy',
                            # 'cv': 5,
                            'n_jobs':-1,
                            'iid': False,
                            'cv':3,
        }
        svm_params = {

            'kernel': 'rbf',
            'decision_function_shape': 'ovr',
            'tol': 1e-06,
        }
        # svm for each mode
        EXUs_t_std_train = EXU_t_std_train.reshape(n_train, T, -1)
        SVMs = train_SVM_models(svm_grid_params, svm_params, EXUs_t_std_train, dpgmm_EXs_t_train_labels, labels)
        exp_data['svm'] = deepcopy(SVMs)
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'svm' not in exp_data:
            assert (False)
        else:
            SVMs = exp_data['svm']

    yumiKin = YumiKinematics(kin_params)

    # long-term prediction for MoE method
    pol = Policy(agent_hyperparams, exp_params_rob)

    ugp_experts_dyn = UGP(dX + dU, **ugp_params)
    ugp_experts_pol = UGP(dX, **ugp_params)

    x_mu_t = exp_data['X0_mu']
    x_var_t = np.diag(exp_data['X0_var'])
    # x_var_t[0, 0] = 1e-6
    # x_var_t[1, 1] = 1e-6  # TODO: cholesky failing for zero v0 variance
    ex_mu_t = exp_data['EX0_mu']
    ex_mu_t_std = EX_scaler.transform(ex_mu_t.reshape(1, -1))
    mode0 = dpgmm.predict(ex_mu_t_std.reshape(1, -1)) # TODO: vel multiplier?
    mode0 = np.asscalar(mode0)
    mc_sample_size = (dX + dU) * 5  # TODO: put this param in some proper place
    num_modes = len(labels)
    modes = labels
    X_mu_pred = []
    X_var_pred = []
    X_particles = []
    sim_data_tree = [[[mode0, -1, x_mu_t, x_var_t, None, None, 1., pol]]]
    start_time = time.time()
    for t in range(H):
        tracks = sim_data_tree[t]
        for track in tracks:
            md = track[0]
            md_prev = track[1]
            x_mu_t = track[2]
            x_var_t = track[3]
            p = track[6]
            pi = track[7]
            assert(pi is not None)
            u_mu_t, u_var_t, _, _, xu_cov = ugp_experts_pol.get_posterior(pi, x_mu_t, x_var_t, t)
            # u_mu_t = Us_t_train[0][t]
            # u_var_t = np.zeros((dU,dU))
            xu_mu_t = np.append(x_mu_t, u_mu_t)
            # xu_var_t = np.block([[x_var_t, np.zeros((dX,dU))],
            #                     [np.zeros((dU,dX)), u_var_t]])
            xu_var_t = np.block([[x_var_t, xu_cov],
                                 [xu_cov.T, u_var_t]])
            track[4] = u_mu_t
            track[5] = u_var_t
            xtut_s = np.random.multivariate_normal(xu_mu_t, xu_var_t, mc_sample_size)
            assert (xtut_s.shape == (mc_sample_size, dX + dU))
            ext_s = yumiKin.forward(xtut_s[:, :dX])
            ut_s = xtut_s[:,dX:]
            extut_s = np.concatenate((ext_s, ut_s), axis=1)
            extut_s_std = EXU_scaler.transform(extut_s)
            clf = SVMs[md]
            mode_dst = clf.predict(extut_s_std)
            mode_counts = Counter(mode_dst).items()
            total_samples = 0
            mode_prob = dict(zip(labels, [0] * len(labels)))
            # mode_p = {}
            for mod in mode_counts:
                if (md == mod[0]) or ((md, mod[0]) in trans_dicts):
                    total_samples = total_samples + mod[1]
            # for mod in mode_counts:
            #     if (md == mod[0]) or ((md, mod[0]) in trans_dicts):
            #         prob = float(mod[1]) / float(total_samples)
            #         mode_p[mod[0]] = prob
            # mode_prob.update(mode_p)
            # alternate mode_prob with state values also
            mode_pred_dict = {}
            for label in labels:
                mode_pred_dict[label] = {'p': 0., 'mu': None, 'var': None}
            for mod in mode_counts:
                if (md == mod[0]) or ((md, mod[0]) in trans_dicts):
                    prob = float(mod[1]) / float(total_samples)
                    mode_pred_dict[mod[0]]['p'] = prob
                    XU_mode = np.array(list(compress(xtut_s, (mode_dst==mod[0]))))
                    mode_pred_dict[mod[0]]['mu'] = np.mean(XU_mode, axis=0)
                    if XU_mode.shape[0]==1:
                        mode_pred_dict[mod[0]]['var'] = np.diag(np.full(dX+dU, 1e-6))
                    else:
                        mode_pred_dict[mod[0]]['var'] = np.cov(XU_mode, rowvar=False)
                    mode_pred_dict[mod[0]]['XU'] = XU_mode
            if len(sim_data_tree) == t + 1:
                sim_data_tree.append([])        # create the next (empty) time step
            # for md_next, p_next in mode_prob.iteritems():
            for mode_pred_key in mode_pred_dict:
                mode_pred = mode_pred_dict[mode_pred_key]
                md_next = mode_pred_key
                p_next = mode_pred['p']
                xu_mu_s_ = mode_pred['mu']
                xu_var_s_ = mode_pred['var']
                if p_next > 1e-4:
                    # get the next state
                    if md_next == md:
                        md_ = md_prev
                        pi_next = pi
                        gp = experts[md]
                        x_mu_t_next_new, x_var_t_next_new, _, _, _ = ugp_experts_dyn.get_posterior(gp, xu_mu_t, xu_var_t)
                    else:
                        md_ = md
                        gp_trans = trans_dicts[(md, md_next)]['mdgp']
                        # x_mu_t_next_new, x_var_t_next_new, _, _, _ = ugp_experts_dyn.get_posterior(gp_trans, xu_mu_t,
                        #                                                                            xu_var_t)
                        xu_var_s_= xu_var_s_ + np.diag(np.diag(xu_var_s_) + 1e-6)
                        x_mu_t_next_new, x_var_t_next_new, _, _, _ = ugp_experts_dyn.get_posterior(gp_trans, xu_mu_s_, xu_var_s_)
                        exp_params_ = deepcopy(exp_params_rob)
                        exp_params_['x0'] = x_mu_t_next_new
                        pi_next = Policy(agent_hyperparams, exp_params_)
                    assert (len(sim_data_tree) == t + 2)
                    tracks_next = sim_data_tree[t + 1]
                    if len(tracks_next)==0:
                        if p*p_next > 1e-4:
                            sim_data_tree[t+1].append([md_next, md_, x_mu_t_next_new, x_var_t_next_new, 0., 0., p*p_next, pi_next])
                    else:
                        md_next_curr_list = [track_next[0] for track_next in tracks_next]
                        if md_next not in md_next_curr_list:
                            # md_next not already in the t+1 time step
                            if p * p_next > 1e-4:
                                sim_data_tree[t + 1].append(
                                    [md_next, md_, x_mu_t_next_new, x_var_t_next_new, 0., 0., p * p_next, pi_next])
                        else:
                            # md_next already in the t+1 time step
                            # if md == md_next:
                            #     md_ = md_prev
                            #     pi_next = pi
                            # else:
                            #     md_ = md
                            #     pi_next = None
                            md_next_curr_trans_list = [(track_next[1], track_next[0]) for track_next in tracks_next]
                            if (md_, md_next) not in md_next_curr_trans_list:
                                # the same transition track is not present
                                if p * p_next > 1e-4:
                                    sim_data_tree[t + 1].append(
                                        [md_next, md_, x_mu_t_next_new, x_var_t_next_new, 0., 0., p * p_next, pi_next])
                            else:
                                it = 0
                                for track_next in tracks_next:
                                    md_next_curr = track_next[0]
                                    md_prev_curr = track_next[1]
                                    x_mu_t_next_curr = track_next[2]
                                    x_var_t_next_curr = track_next[3]
                                    p_next_curr = track_next[6]
                                    pi_next = track_next[7]
                                    if md_next == md_next_curr:
                                        next_trans = (md_, md_next)
                                        curr_trans = (md_prev_curr, md_next_curr)
                                        if curr_trans == next_trans:
                                            p_next_new = p*p_next
                                            tot_new_p = p_next_curr + p_next_new
                                            w1 = p_next_curr / tot_new_p
                                            w2 = p_next_new / tot_new_p
                                            mu_next_comb = w1 * x_mu_t_next_curr + w2 * x_mu_t_next_new
                                            var_next_comb = w1 * x_var_t_next_curr + w2 * x_var_t_next_new + \
                                                            w1 * np.outer(x_mu_t_next_curr,x_mu_t_next_curr) + \
                                                            w2 * np.outer(x_mu_t_next_new, x_mu_t_next_new) -\
                                                            np.outer(mu_next_comb,mu_next_comb)
                                            p_next_comb = p_next_curr + p_next_new
                                            if p_next_comb > 1e-4:
                                                sim_data_tree[t + 1][it] = \
                                                    [md_next, md_, mu_next_comb, var_next_comb, 0., 0., p_next_comb, pi_next]
                                    it+=1

        # probability check
        prob_mode_tot = 0.
        for track_ in sim_data_tree[t]:
                prob_mode_tot += track_[6]
        if (prob_mode_tot - 1.0) > 1e-4:
            assert (False)

    print 'Prediction time for MoE UGP with horizon', H, ':', time.time() - start_time


    # plot each path (in mode) separately
    # path is assumed to be a path arising out from a unique transtions
    # different paths arising out of the same transition at different time is allowed in our model not here
    tm = np.array(range(H)) * dt

    path_dict = {}
    for i in range(H):
        t = tm[i]
        tracks = sim_data_tree[i]
        for track in tracks:
            path = (track[0], track[1])
            if path not in path_dict:
                path_dict[path] = {'time':[] ,'X':[], 'X_var':[], 'X_std':[], 'U':[], 'U_var':[], 'U_std':[], 'prob':[], 'col':label_col_dict[path[0]]}
            path_dict[path]['time'].append(t)
            path_dict[path]['X'].append(track[2])
            path_dict[path]['X_var'].append(track[3])
            path_dict[path]['U'].append(track[4])
            path_dict[path]['U_var'].append(track[5])
            path_dict[path]['X_std'].append(np.sqrt(np.diag(track[3])))
            path_dict[path]['U_std'].append(np.sqrt(np.diag(track[5])))
            path_dict[path]['prob'].append(track[6])


    # plot for tree structure
    # plot long term prediction results of UGP
    plt.figure()
    for path_key in path_dict:
    # path_key = (10,-1)
        path = path_dict[path_key]
        time = np.array(path['time'])
        pos = np.array(path['X'])[:,:dP]
        pos_std = np.array(path['X_std'])[:, :dP]
        vel = np.array(path['X'])[:, dP:]
        vel_std = np.array(path['X_std'])[:, dP:]
        trq = np.array(path['U'])
        trq_std = np.array(path['U_std'])

        prob = np.array(path['prob']).reshape(-1,1)
        col = np.tile(path['col'], (time.shape[0],1))
        rbga_col = np.concatenate((col, prob), axis=1)
        for j in range(dP):
            plt.subplot(3, 7, 1 + j)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Joint Pos (rad)')
            plt.title('j%dPos' % (j + 1))
            plt.scatter(time, pos[:, j], color=col, s=3, marker='s')
            plt.fill_between(time, pos[:, j] - pos_std[:, j] * 1.96, pos[:, j] + pos_std[:, j] * 1.96,
                             alpha=0.2, color=col)
            for i in range(n_train):
                x = Xs_t_train[i, :, j]
                cl = cols[i]
                plt.scatter(tm, x, alpha=0.5, color=cl, s=1)
        for j in range(dV):
            plt.subplot(3, 7, 8 + j)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Joint Pos (rad)')
            plt.title('j%dVel' % (j + 1))
            plt.scatter(time, vel[:, j], color=col, s=3, marker='s')
            plt.fill_between(time, vel[:, j] - vel_std[:, j] * 1.96, vel[:, j] + vel_std[:, j] * 1.96,
                             alpha=0.2, color=col)
            for i in range(n_train):
                x = Xs_t_train[i, :, dP+j]
                cl = cols[i]
                plt.scatter(tm, x, alpha=0.5, color=cl, s=1)
        for j in range(dU):
            plt.subplot(3, 7, 15 + j)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Joint Pos (rad)')
            plt.title('j%dTrq' % (j + 1))
            plt.scatter(time, trq[:, j], color=col, s=3, marker='s')
            plt.fill_between(time, trq[:, j] - trq_std[:, j] * 1.96, trq[:, j] + trq_std[:, j] * 1.96,
                             alpha=0.2, color=col)
            for i in range(n_train):
                u = XUs_t_train[i, :, dX+j]
                cl = cols[i]
                plt.scatter(tm, u, alpha=0.5, color=cl, s=1)
        plt.show(block=False)

    # compute long-term prediction score
    XUs_t_test = exp_data['XUs_t_test']
    assert (XUs_t_test.shape[0] == n_test)
    X_test_log_ll = np.zeros((H, n_test))
    for i in range(n_test):
        XU_test = XUs_t_test[i]
        for t in range(H):
            x_t = XU_test[t, :dX]
            tracks = sim_data_tree[t]
            prob_mix = 0.
            for track in tracks:
                prob_mix += sp.stats.multivariate_normal.pdf(x_t, track[2], track[3])*track[6]
            X_test_log_ll[t, i] = np.log(prob_mix)

    tm = np.array(range(H)) * dt
    # plt.figure()
    # plt.title('Average NLL of test trajectories w.r.t time ')
    # plt.xlabel('Time, s')
    # plt.ylabel('NLL')
    # plt.plot(tm.reshape(H, 1), X_test_log_ll)

    nll_mean = np.mean(X_test_log_ll.reshape(-1))
    nll_std = np.std(X_test_log_ll.reshape(-1))
    print 'NLL mean: ', nll_mean, 'NLL std: ', nll_std

    # plt.show()

    # plot only mode of multimodal dist
    # tm = np.array(range(H))
    # P_mu = np.zeros(H)
    # V_mu = np.zeros(H)
    # Xs_mu_pred = []
    # for t in range(H):
    #     tracks = sim_data_tree[t]
    #     xp_pairs = [[track[2], track[6]] for track in tracks]
    #     xs = [track[2] for track in tracks]
    #     Xs_mu_pred.append(xs)
    #     xp_max = max(xp_pairs, key=lambda x: x[1])
    #     P_mu[t] = xp_max[0][0]
    #     V_mu[t] = xp_max[0][1]
    #
    # # prepare for contour plot
    # tm_grid = tm
    # x_grid = np.arange(-1, 4, grid_size)  # TODO: get the ranges from the mode dict
    # Xp, Tp = np.meshgrid(x_grid, tm_grid)
    # prob_map_pos = np.zeros((len(tm_grid), len(x_grid)))
    # prob_map_vel = np.zeros((len(tm_grid), len(x_grid)))
    #
    # for i in range(len(x_grid)):
    #     for t in range(len(tm_grid)):
    #         x = x_grid[i]
    #         tracks = sim_data_tree[t]
    #         for track in tracks:
    #             w = track[6]
    #             # if w > 1e-4:
    #             mu = track[2][:dP]
    #             var = track[3][:dP, :dP]
    #             prob_val = sp.stats.norm.pdf(x, mu, np.sqrt(var)) * w
    #             prob_map_pos[t, i] += prob_val
    #             mu = track[2][dP:dP+dV]
    #             var = track[3][dP:dP+dV, dP:dP+dV]
    #             prob_val = sp.stats.norm.pdf(x, mu, np.sqrt(var)) * w
    #             prob_map_vel[t, i] += prob_val
    #         # if prob_map[t, i]<prob_limit[t]:
    #         #     prob_map[t, i] = 0.
    # # probability check
    # print 'prob_map_pos', prob_map_pos.sum(axis=1) * grid_size
    # print 'prob_map_vel', prob_map_vel.sum(axis=1) * grid_size
    #
    # min_prob_den = min_prob_grid / grid_size
    # plt.figure()
    # plt.subplot(121)
    # plt.title('Long-term prediction for position with ME')
    # plt.xlabel('Time, t')
    # plt.ylabel('State, x(t)')
    # plt.plot(tm, P_mu, color='g', ls='-', marker='s', linewidth='2', label='Position', markersize=5)
    # plt.contourf(Tp, Xp, prob_map_pos, colors='g', alpha=.2,
    #              levels=[min_prob_den, 100.])  # TODO: levels has to properly set according to some confidence interval
    # # plt.plot(tm, traj_gt[:H, 1], color='g', ls='-', marker='^', linewidth='2', label='True dynamics', markersize=7)
    # for x in Xs_t_train:
    #     plt.plot(tm, x[:H, :dP])
    # # plt.colorbar()
    # plt.legend()
    # plt.subplot(122)
    # plt.title('Long-term prediction for velocity with ME')
    # plt.xlabel('Time, t')
    # plt.ylabel('State, x(t)')
    # plt.plot(tm, V_mu, color='b', ls='-', marker='s', linewidth='2', label='Velocity', markersize=5)
    # plt.contourf(Tp, Xp, prob_map_vel, colors='b', alpha=.2,
    #              levels=[min_prob_den, 50.])  # TODO: levels has to properly set according to some confidence interval
    # # plt.plot(tm, traj_gt[:H, 1], color='g', ls='-', marker='^', linewidth='2', label='True dynamics', markersize=7)
    # for x in Xs_t_train:
    #     plt.plot(tm, x[:H, dP:dP+dV])
    # # plt.colorbar()
    # plt.legend()

plt.show(block=False)
None




