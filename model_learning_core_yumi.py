import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mat_col
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from model_leraning_utils import get_N_HexCol
from model_leraning_utils import train_trans_models
# from model_leraning_utils import SVMmodePrediction
from model_leraning_utils import SVMmodePredictionGlobal as SVMmodePrediction
from model_leraning_utils import DPGMMCluster
from collections import Counter
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
# from multidim_gp import MultidimGP
from multidim_gp import MdGpyGPwithNoiseEst as MultidimGP
# from multidim_gp import MdGpyGP as MultidimGP
# from multidim_gp import MdGpySparseGP as MultidimGP
from model_leraning_utils import UGP, SimplePolicy
from model_leraning_utils import dummySVM, traj_with_globalgp
from YumiKinematics import YumiKinematics
from model_leraning_utils import logsum
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
# from mjc_exp_policy import SimplePolicy

import copy
# np.random.seed(1)     # good value for clustering new yumi exp
# np.random.seed(3)
np.random.seed(4)       # good for big data wom10 without normalizing for clustering

# logfile = "./Results/yumi_exp_preprocessed_data_2.p"
# logfile = "./Results/yumi_peg_exp_preprocessed_data_1.p"
# logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train.p"    # includes a trained global gp
# logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train_1.p"      # new yumi exp
# logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train_2.p"      # global gp trained and lt pred working with simple policy
# logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train_3.p"          # with EX_ee points, also lt moe working with simple policy and has perturbed exp data
# logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train_4.p"          # first full result with small data
logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train_big_data_wom10.p"
logfile_test_m2 = "./Results/yumi_peg_exp_new_preprocessed_data_test_m2_1.p"
logfile_test_m5 = "./Results/yumi_peg_exp_new_preprocessed_data_test_m5_1.p"
logfile_test_m10 = "./Results/yumi_peg_exp_new_preprocessed_data_test_m10_1.p"
logfile_test_p2 = "./Results/yumi_peg_exp_new_preprocessed_data_test_p2_1.p"
logfile_test_p5 = "./Results/yumi_peg_exp_new_preprocessed_data_test_p5_1.p"
logfile_test_p10 = "./Results/yumi_peg_exp_new_preprocessed_data_test_p10_1.p"

gp_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_gp_bigdata.p"
moe_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_moe_bigdata.p"

# gp_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_gp_smalldata.p"
# moe_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_moe_smalldata.p"

# gp_results = {}
gp_results = pickle.load(open(gp_result_file, "rb")) # global gp training done with 15 base policy trials
# moe_results = {}
moe_results = pickle.load( open(moe_result_file, "rb" ) )

# logfile = "./Results/mjc_exp_2_sec_raw_preprocessed.p"

vbgmm_refine = False

global_gp = True
delta_model = True
fit_moe = True

load_all = True
load_gp = False
load_dpgmm = True
load_transition_gp = True
load_experts = True
load_svms = True
load_global_lt_pred = False


min_prob_grid = 0.001 # 1%
grid_size = 0.005
min_counts = 5 # min number of cluster size.
prob_min = 1e-3
mc_factor = 10
min_mc_particles = 3
# both pos and vel var was set to 6.25e-4 initially
p_noise_var = np.full(7, 1e-6)
v_noise_var = np.full(7, 2.5e-3)
# pol_per_facor = -0.02
jitter_var_tl = 1e-6
# jitter_var_tl = 0
num_tarj_samples = 50

exp_data = pickle.load( open(logfile, "rb" ) )
exp_test_m2 = pickle.load( open(logfile_test_m2, "rb" ) )
exp_test_m5 = pickle.load( open(logfile_test_m5, "rb" ) )
exp_test_m10 = pickle.load( open(logfile_test_m10, "rb" ) )
exp_test_p2 = pickle.load( open(logfile_test_p2, "rb" ) )
exp_test_p5 = pickle.load( open(logfile_test_p5, "rb" ) )
exp_test_p10 = pickle.load( open(logfile_test_p10, "rb" ) )

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
# n_train = exp_data['n_train']
# n_test = exp_data['n_test']-1 # TODO: remove -1 this is done to fix a bug in the logfile but already fixed in the code.
# n_test = exp_data['n_test']
# data set for joint space
XUs_t_train = exp_data['XUs_t_train']  #TODO: reverse the restricted training trials count (for global gp training)
XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1])
n_train, _, _ = XUs_t_train.shape
# XU_t_train_avg = exp_data['XU_t_train_avg']
# Xs_t1_train_avg = exp_data['Xs_t1_train_avg']
# Xrs_t_train = exp_data['Xrs_t_train']
Xrs_t_test = exp_data['Xrs_t_test']
# Xrs_t_test_m2 = exp_test_m2['Xrs_t_test']
# Xrs_t_test_m5 = exp_test_m5['Xrs_t_test']
# Xrs_t_test_m10 = exp_test_m10['Xrs_t_test']
# Xrs_t_test_p2 = exp_test_p2['Xrs_t_test']
# Xrs_t_test_p5 = exp_test_p5['Xrs_t_test']
# Xrs_t_test_p10 = exp_test_p10['Xrs_t_test']

Xs_t_train = exp_data['Xs_t_train']
X_t_train = Xs_t_train.reshape(-1, Xs_t_train.shape[-1])

Xs_t1_train = exp_data['Xs_t1_train']
X_t1_train = Xs_t1_train.reshape(-1, Xs_t1_train.shape[-1])

EXs_t_train = exp_data['EXs_t_train']
EX_t_train = EXs_t_train.reshape(-1, EXs_t_train.shape[-1])

# data set for cartesian space
EXFs_t_train = exp_data['EXFs_t_train']
EXF_t_train = EXFs_t_train.reshape(-1, EXFs_t_train.shape[-1])

EXs_t_train = exp_data['EXs_t_train']
EX_t_train = EXs_t_train.reshape(-1, EXs_t_train.shape[-1])

EXs_t1_train = exp_data['EXs_t1_train']
EX_t1_train = EXs_t1_train.reshape(-1, EXs_t1_train.shape[-1])

Fs_t_train = exp_data['Fs_t_train']
F_t_train = Fs_t_train.reshape(-1, Fs_t_train.shape[-1])
EX_1_EX_F_t_train = np.concatenate((EX_t1_train, EX_t_train, F_t_train), axis=1)

Us_t_train = exp_data['Us_t_train']
Us_t_test = exp_data['Us_t_test']
Xs_t_test = exp_data['Xs_t_test']
U_t_train = Us_t_train.reshape(-1, Us_t_train.shape[-1])
EXU_t_train = np.concatenate((EX_t_train, U_t_train), axis=1)

Us_t_test_m2 = exp_test_m2['Us_t_test']
Us_t_test_m5 = exp_test_m5['Us_t_test']
Us_t_test_m10 = exp_test_m10['Us_t_test']
Us_t_test_p2 = exp_test_p2['Us_t_test']
Us_t_test_p5 = exp_test_p5['Us_t_test']
Us_t_test_p10 = exp_test_p10['Us_t_test']

dX_t_train = X_t1_train - X_t_train

# EXFs_t_test = exp_data['EXFs_t_test']

EXs_ee_t_train = exp_data['EXs_ee_t_train']
EX_ee_t_train = EXs_ee_t_train.reshape(-1, EXs_ee_t_train.shape[-1])

# # unperturbed simple policy
# # Xrs_data = Xrs_t_train
# # Us_data = Us_t_train
# Xrs_data = Xrs_t_test
# Us_data = Us_t_test
# XUs_t_test = exp_data['XUs_t_test']
# XUs_test_data = XUs_t_test
# exp_params_data = deepcopy(exp_params_rob)

# m10 simple policy
# Xrs_data = Xrs_t_train
# Us_data = Us_t_train
Xrs_data = Xrs_t_test
Us_data = Us_t_test
XUs_t_test = exp_data['XUs_t_test']
n_test, _, _, = XUs_t_test.shape
XUs_test_data = XUs_t_test
Kp = exp_params_rob['Kp']
pol_per_facor = -0.1
exp_params_data = deepcopy(exp_params_rob)
exp_params_data['Kp'] = Kp + Kp * pol_per_facor


# # select perturbed experiment
# XUs_test_data = exp_test_m10['XUs_t_test'][3:, :, :]
# Kp = exp_params_rob['Kp']
# pol_per_facor = -0.1
# exp_params_data = deepcopy(exp_params_rob)
# exp_params_data['Kp'] = Kp + Kp * pol_per_facor
# Xrs_data = Xrs_t_test_m10[:3, :, :]
# Us_data = Us_t_test_m10[:3, :, :]



# filter vel signal for estimating vel noise variance
# plt.figure()
# tm = range(T)
# for i in range(n_train):
#     x = Xs_t_train[i, :, dP + 6]
#     x_fil = sp.ndimage.gaussian_filter1d(x, 4)
#     plt.plot(tm, x, alpha=0.2)
#     plt.plot(tm, x_fil)
#     plt.show()

# filter vel signal for estimating pos noise variance
# plt.figure()
# tm = range(T)
# for i in range(n_train):
#     x = Xs_t_train[i, :, 6]
#     x_fil = sp.ndimage.gaussian_filter1d(x, 1.)
#     plt.plot(tm, x, alpha=1.)
#     plt.plot(tm, x_fil)
#     plt.show()

# estimate vel noise variance for each joint
# v_res_s = np.zeros((n_train, T, dV))
# v_var_s = np.zeros((n_train, 7))
# for i in range(n_train):
#     for j in range(7):
#         v = Xs_t_train[i, :, dP + j]
#         v_f = sp.ndimage.gaussian_filter1d(v, 4)
#         v_res_s[i, :, j] = v - v_f
#         v_var_s[i, j] = np.var(v_res_s[i, :, j])
# v_res = v_res_s.reshape(-1, v_res_s.shape[-1])
# v_var = np.var(v_res, axis=0)
# v_std = np.sqrt(v_var)
# v_var_mean = np.mean(v_var)

# # estimate pos noise variance for each joint
# p_res_s = np.zeros((n_train, T, dP))
# p_var_s = np.zeros((n_train, 7))
# for i in range(n_train):
#     for j in range(7):
#         p = Xs_t_train[i, :, j]
#         p_f = sp.ndimage.gaussian_filter1d(p, 1)
#         p_res_s[i, :, j] = p - p_f
#         p_var_s[i, j] = np.var(p_res_s[i, :, j])
# p_res = p_res_s.reshape(-1, p_res_s.shape[-1])
# p_var = np.var(p_res, axis=0)
# p_std = np.sqrt(p_var)
# p_var_mean = np.mean(p_var)

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

gpr_params_global = {
        'normalize': True,
        'constrain_ls': True,
        'ls_b_mul': (0.1, 10.),
        'constrain_sig_var': True,
        'sig_var_b_mul': (0.1, 10.),
        'noise_var': np.concatenate([p_noise_var, v_noise_var]),
        # 'noise_var': None,
        'constrain_noise_var': True,
        'noise_var_b_mul': (1e-2, 10.),
        'fix_noise_var': False,
        'restarts': 1,
    }
# expl_noise = 3.
H = T  # prediction horizon

if global_gp:

    # global gp fit
    if not load_gp:
        mdgp_glob = MultidimGP(gpr_params_global, dX)
        start_time = time.time()
        if not delta_model:
            mdgp_glob.fit(XU_t_train, X_t1_train)
        else:
            mdgp_glob.fit(XU_t_train, dX_t_train)
        gp_training_time = time.time() - start_time
        print 'Global GP fit time', gp_training_time
        gp_results['gp_training_time'] = gp_training_time
        exp_data['mdgp_glob'] = deepcopy(mdgp_glob)
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'mdgp_glob' not in exp_data:
            assert(False)
        else:
            mdgp_glob = exp_data['mdgp_glob']
    global_gp_noise_level = []
    global_gp_ls = []
    global_gp_variance = []
    for gp in mdgp_glob.gp_list:
        global_gp_noise_level.append(gp.parameters[1].parameters[0])
        global_gp_ls.append(gp.parameters[0].parameters[1])
        global_gp_variance.append(gp.parameters[0].parameters[0])
    print('Pos noise', global_gp_noise_level[:dP])
    print('Vel noise', global_gp_noise_level[dP:])
    print('Pos variance', global_gp_variance[dP:])
    print('Vel variance', global_gp_variance[:dP])
    print('Pos ls', global_gp_ls[dP:])
    print('Vel ls', global_gp_ls[:dP])
    if not load_global_lt_pred:
        # global gp long-term prediction
        # long-term prediction for MoE method
        # original simple policy

        pol = Policy(agent_hyperparams, exp_params_rob)
        # pol1 = Policy(agent_hyperparams, exp_params_rob)
        # sim_pol = SimplePolicy(Xrs_t_train, Us_t_train, exp_params_rob)
        sim_pol = SimplePolicy(Xrs_data, Us_data, exp_params_data)

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
            # x_t = np.random.multivariate_normal(x_mu_t, x_var_t)
            # u_mu_t, u_var_t, _, _, xu_cov = ugp_global_pol.get_posterior(pol, x_mu_t, x_var_t, t)
            u_mu_t, u_var_t, _, _, xu_cov = ugp_global_pol.get_posterior(sim_pol, x_mu_t, x_var_t, t)
            U_mu_pred.append(u_mu_t)
            U_var_pred.append(u_var_t)
            X_mu_pred.append(x_mu_t)
            X_var_pred.append(x_var_t)
            # X_particles.append(Y_mu)
            xu_mu_t = np.append(x_mu_t, u_mu_t)
            xu_var_t = np.block([[x_var_t, xu_cov],
                                 [xu_cov.T, u_var_t]])

            # simple policy evaluation
            _, u_mu_t_sp = sim_pol.act(x_mu_t, t)
            U_mu_pred_sp.append(u_mu_t_sp)


            # fix u with mean u data
            # u_mu_t_avg = XU_t_train_avg[t, dX:dX + dU]
            # U_mu_pred_avg.append(u_mu_t_avg)
            # xu_mu_t = np.append(x_mu_t, u_mu_t_avg)

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
            x_var_t = x_var_t + np.eye(dX, dX) * jitter_var_tl   # to prevent collapse of the Gaussian
        gp_pred_time = time.time() - start_time
        print 'Global GP prediction time for horizon', H, ':', gp_pred_time
        gp_results['gp_pred_time'] = gp_pred_time
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

    # tm = np.array(range(H)) * dt
    # plt.figure()
    # plt.title('Average NLL of test trajectories w.r.t time ')
    # plt.xlabel('Time, t')
    # plt.ylabel('NLL')
    # plt.plot(tm.reshape(H,1), np.mean(X_test_log_ll, axis=1).reshape(H, 1))

    nll_mean = np.mean(X_test_log_ll.reshape(-1))
    nll_std = np.std(X_test_log_ll.reshape(-1))
    print 'UT NLL mean: ', nll_mean, 'UT NLL std: ', nll_std

    # plot long-term prediction
    X_mu_pred = np.array(X_mu_pred)
    U_mu_pred = np.array(U_mu_pred)
    # U_mu_pred_x_avg = np.array(U_mu_pred_x_avg)
    # U2_mu_pred = np.array(U2_mu_pred)
    # U_mu_pred_avg = np.array(U_mu_pred_avg)
    # U_mu_pred_sp = np.array(U_mu_pred_sp)

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
        plt.plot(tm, Xs_t_test[:, :H, j].T, alpha=0.2)
        plt.plot(tm, P_mu_pred[:H, j], color='g', marker='s', markersize=2,)
        plt.fill_between(tm, P_mu_pred[:H, j] - P_sig_pred[:H, j] * 1.96, P_mu_pred[:H, j] + P_sig_pred[:H, j] * 1.96, alpha=0.2, color='g')
    # jVel
    for j in range(dV):
        plt.subplot(3, 7, 8 + j)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint Vel (rad/s)')
        plt.title('j%dVel' % (j + 1))
        plt.plot(tm, Xs_t_test[:, :H, dP+j].T, alpha=0.2)
        plt.plot(tm, V_mu_pred[:H, j], color='b', marker='s', markersize=2,)
        plt.fill_between(tm, V_mu_pred[:H, j] - V_sig_pred[:H, j] * 1.96, V_mu_pred[:H, j] + V_sig_pred[:H, j] * 1.96,
                         alpha=0.2, color='b')
    for j in range(dV):
        plt.subplot(3, 7, 15 + j)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint Trq (Nm)')
        plt.title('j%dTrq' % (j + 1))
        plt.plot(tm, XUs_t_test[:, :H, dX+j].T, alpha=0.2)
        plt.plot(tm, U_mu_pred[:H, j], color='r', marker='s', markersize=2, label='mean pred')
        plt.fill_between(tm, U_mu_pred[:H, j] - U_sig_pred[:H, j] * 1.96, U_mu_pred[:H, j] + U_sig_pred[:H, j] * 1.96,
                         alpha=0.2, color='r')
        # plt.plot(tm, U_mu_pred_avg[:H, j], color='r', linestyle='--', label='mean data')
        # plt.plot(tm, U_mu_pred_x_avg[:H, j], color='r', linestyle='-.', label='avg state based')
        # plt.plot(tm, U_mu_pred_sp[:H, j], color='r', linestyle='-.', label='simple pol')
    plt.legend()
    plt.show(block=False)

    # trajectory sampling approach (for consistency )
    x_mu_0 = exp_data['X0_mu']
    x_var_0 = np.diag(exp_data['X0_var'])
    traj_with_globalgp_ = traj_with_globalgp(x_mu_0, x_var_0, mdgp_glob, sim_pol, dlt_mdl=delta_model)
    traj_samples = traj_with_globalgp_.sample(num_tarj_samples, H)
    gp_results['traj_samples'] = traj_samples
    traj_mean = np.mean(traj_samples, axis=0)
    traj_std = np.sqrt(np.var(traj_samples, axis=0))
    traj_covar = np.zeros((H, dX, dX))
    for t in range(H):
        traj_covar[t] = np.cov(traj_samples[:, t, :], rowvar=False)
    # tm = np.array(range(H)) * dt
    tm = np.array(range(H))
    plt.figure()
    plt.title('Long-term prediction with GP (traj sampling)')
    # jPos
    for j in range(dP):
        plt.subplot(3, 7, 1 + j)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint Pos (rad)')
        plt.title('j%dPos' % (j + 1))
        plt.autoscale(True)
        plt.plot(tm, Xs_t_train[:, :H, j].T, alpha=0.2, color='k')
        # plt.autoscale(False)
        plt.plot(tm, traj_samples[:, :H, j].T, alpha=0.2, color='g')
        plt.plot(tm, traj_mean[:H, j], color='g')
        plt.fill_between(tm, traj_mean[:H, j] - traj_std[:H, j] * 1.96, traj_mean[:H, j] + traj_std[:H, j] * 1.96,
                         alpha=0.2, color='g')

    # jVel
    for j in range(dV):
        plt.subplot(3, 7, 8 + j)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint Vel (rad/s)')
        plt.title('j%dVel' % (j + 1))
        plt.autoscale(True)
        plt.plot(tm, Xs_t_train[:, :H, dP + j].T, alpha=0.2, color='k')
        # plt.autoscale(False)
        plt.plot(tm, traj_samples[:, :H, dP+j].T, alpha=0.2, color='b')
        plt.plot(tm, traj_mean[:H, dP + j], color='b')
        plt.fill_between(tm, traj_mean[:H, dP + j] - traj_std[:H, dP + j] * 1.96,
                         traj_mean[:H, dP + j] + traj_std[:H, dP + j] * 1.96,
                         alpha=0.2, color='b')

    for j in range(dV):
        plt.subplot(3, 7, 15 + j)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint Trq (Nm)')
        plt.title('j%dTrq' % (j + 1))
        plt.autoscale(True)
        plt.plot(tm, XUs_t_train[:, :H, dX + j].T, alpha=0.2, color='k')
    plt.legend()
    plt.show(block=False)

    # loglikelihood score
    XUs_t_test = exp_data['XUs_t_test']
    assert (XUs_t_test.shape[0] == n_test)
    X_test_log_ll = np.zeros((H, n_test))
    X_test_SE = np.zeros((H, n_test))
    for t in range(H):  # one data point less than in XU_test
        for i in range(n_test):
            XU_test = XUs_t_test[i]
            x_t = XU_test[t, :dX].reshape(-1)
            x_mu_t = traj_mean[t].reshape(-1)
            x_var_t = traj_covar[t]
            # x_var_t = traj_covar[t] + np.eye(dX) * jitter_var_tl
            x_var_t = np.diag(np.diag(x_var_t))
            X_test_log_ll[t, i] = sp.stats.multivariate_normal.logpdf(x_t, x_mu_t, x_var_t)
            X_test_SE[t, i] = np.dot((x_t - x_mu_t), (x_t - x_mu_t))

    tm = np.array(range(H)) * dt
    plt.figure()
    plt.title('Average NLL of test trajectories w.r.t time ')
    plt.xlabel('Time, t')
    plt.ylabel('NLL')
    plt.plot(tm.reshape(H,1), np.mean(X_test_log_ll, axis=1).reshape(H, 1))

    tm = np.array(range(H)) * dt
    plt.figure()
    plt.title('Average RMSE of test trajectories w.r.t time ')
    plt.xlabel('Time, t')
    plt.ylabel('NLL')
    plt.plot(tm.reshape(H, 1), np.mean(X_test_SE, axis=1).reshape(H, 1))

    nll_mean = np.mean(X_test_log_ll.reshape(-1))
    nll_std = np.std(X_test_log_ll.reshape(-1))
    rmse = np.sqrt(np.mean(X_test_SE.reshape(-1)))
    print('Yumi exp GP', 'NLL mean: ', nll_mean, 'NLL std: ', nll_std, 'RMSE:', rmse)

    gp_results['rmse'] = rmse
    gp_results['nll'] = (nll_mean, nll_std)
    pickle.dump(gp_results, open(gp_result_file, "wb"))




if fit_moe:
    if not load_dpgmm:
        # clust_data = X_t_train
        # clust_data = EX_t_train
        clust_data = EX_ee_t_train
        dof = clust_data.shape[1] + 2
        dpgmm_params = {
            'n_components': 20,  # cluster size
            'covariance_type': 'full',
            'tol': 1e-6,
            'n_init': 10,
            'max_iter': 300,
            'weight_concentration_prior_type': 'dirichlet_process',
            'weight_concentration_prior':1e-1,
            'mean_precision_prior':None,
            'mean_prior': None,
            'degrees_of_freedom_prior': dof,
            'covariance_prior': None,
            'warm_start': False,
            'init_params': 'random',
            'verbose': 1,
        }
        dpgmm_params_extra = {
                'min_clust_size': 50,
                # 'min_clust_size': 20, # for new yumi exp
                'standardize': False,
                'vbgmm_refine': False,
                'min_size_filter': True,
                'seg_filter': True,
                'n_train': n_train,
        }
        ##########Clustering notes for yumi exp###########

        ##################################################

        dpgmm = DPGMMCluster(dpgmm_params, dpgmm_params_extra, clust_data)
        start_time = time.time()
        clustered_labels, labels, counts = dpgmm.cluster()
        cluster_time = time.time() - start_time
        moe_results['cluster_time'] = cluster_time
        print('Clustering time:', cluster_time)
        exp_data['dpgmm'] = deepcopy(dpgmm)
        exp_data['clust_result'] = {'assign': clustered_labels, 'labels': labels, 'counts': counts}
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'dpgmm' not in exp_data:
            assert (False)
        else:
            dpgmm = exp_data['dpgmm']
            clustered_labels = exp_data['clust_result']['assign']
            labels = exp_data['clust_result']['labels']
            counts = exp_data['clust_result']['counts']

    clustered_labels_t = clustered_labels
    clustered_labels_t_s = clustered_labels_t.reshape(n_train, -1)
    clustered_labels_t1 = np.append(clustered_labels[1:], clustered_labels[-1])
    clustered_labels_t1_s = clustered_labels_t1.reshape(n_train, -1)

    K = len(labels)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors) / 255.

    fig = plt.figure()
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.bar(labels, counts, color=colors)
    plt.title('DPGMM clustering')
    plt.ylabel('Cluster sizes')
    plt.xlabel('Cluster labels')
    plt.show(block=False)
    plt.savefig('dpgmm_yumi_cluster counts.pdf')
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
        col[(clustered_labels == label)] = colors[i]
        i += 1
    cols = col.reshape(n_train, T, -1)
    label_col_dict = dict(zip(labels, colors))
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
        # gpr_params_trans = {
        #     'noise_var': np.concatenate((p_noise_var, v_noise_var)),
        #     # 'noise_var': None,
        #     'normalize': True,
        # }
        gpr_params_trans = {
            'normalize': True,
            'constrain_ls': False,
            'ls_b_mul': (0.01, 10.),
            'constrain_sig_var': False,
            'sig_var_b_mul': (0.01, 10.),
            # 'noise_var': np.concatenate([p_noise_var, v_noise_var]),
            'noise_var': None,
            'constrain_noise_var': False,
            'noise_var_b_mul': (0.1, 10.),
            'fix_noise_var': False,
            'restarts': 1,
        }
        start_time = time.time()
        trans_dicts = train_trans_models(gpr_params_trans, XUs_t_train, clustered_labels_t_s, dX, dU)
        trans_gp_time = time.time() - start_time
        moe_results['trans_gp_time'] = trans_gp_time
        print ('Transition GP training time:', trans_gp_time)
        exp_data['transition_gp'] = deepcopy(trans_dicts)
        pickle.dump(exp_data, open(logfile, "wb"))

    else:
        if 'transition_gp' not in exp_data:
            assert(False)
        else:
            trans_dicts = exp_data['transition_gp']

    if not load_experts:
        # expert training
        # gpr_params_experts = {
        #     'noise_var': np.concatenate((p_noise_var, v_noise_var)),
        #     'normalize': True,
        # }
        # gpr_params_experts = {
        #     'normalize': False,
        #     'constrain_ls': True,
        #     'ls_b_mul': (0.01, 10.),
        #     'constrain_sig_var': True,
        #     'sig_var_b_mul': (0.01, 10.),
        #     # 'noise_var': np.concatenate([p_noise_var, v_noise_var]),
        #     'noise_var': None,
        #     'constrain_noise_var': True,
        #     'noise_var_b_mul': (0.01, 10.),
        #     'fix_noise_var': False,
        #     'restarts': 1,
        # }
        gpr_params_experts = {
            'normalize': True,
            'constrain_ls': True,
            'ls_b_mul': (0.1, 10.),
            'constrain_sig_var': True,
            'sig_var_b_mul': (0.1, 10.),
            'noise_var': np.concatenate([p_noise_var, v_noise_var]),
            # 'noise_var': None,
            'constrain_noise_var': True,
            'noise_var_b_mul': (1e-2, 10.),
            'fix_noise_var': False,
            'restarts': 1,
        }
        experts = {}
        start_time = time.time()
        for label in labels:
            expert_idx = np.logical_and((clustered_labels_t == label), (clustered_labels_t1 == label))
            x_train = XU_t_train[expert_idx]
            y_train = X_t1_train[expert_idx]
            if delta_model:
                y_train = y_train - x_train[:, :dX]
            mdgp = MultidimGP(gpr_params_experts, y_train.shape[1])
            mdgp.fit(x_train, y_train)
            experts[label] = deepcopy(mdgp)
            del mdgp
        expert_train_time = time.time() - start_time
        moe_results['expert_train_time'] = expert_train_time
        print 'Experts training time:', expert_train_time
        exp_data['experts'] = deepcopy(experts)
        pickle.dump(exp_data, open(logfile, "wb"))

    else:
        if 'experts' not in exp_data:
            assert(False)
        else:
            experts = exp_data['experts']

    # plot experts data
    # plt.figure()
    # tm = np.tile(np.array(range(H)), n_train)
    # for label in labels:
    #     expert_idx = np.logical_and((clustered_labels_t == label), (clustered_labels_t1 == label))
    #     x_train = XU_t_train[expert_idx]
    #     y_train = X_t1_train[expert_idx]
    #     tm_exp = tm[expert_idx]
    #     for j in range(7):
    #         plt.subplot(3, 7, 1+j)
    #         plt.scatter(tm_exp, x_train[:, j], s=2)
    #         plt.subplot(3, 7, 8 + j)
    #         plt.scatter(tm_exp, x_train[:, 7+j], s=2)
    #         plt.subplot(3, 7, 15 + j)
    #         plt.scatter(tm_exp, x_train[:, 14 + j], s=2)
            # plt.show(block=False)

    if not load_svms:
        # gating network training
        svm_grid_params = {
                            'param_grid': {"C": np.logspace(-12, 12, endpoint=True, num=11, base=2.),
                                           "gamma": np.logspace(-12, 12, endpoint=True, num=11, base=2.)},
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
        mode_prediction_data_t = XUs_t_train
        mode_predictor = SVMmodePrediction(svm_grid_params, svm_params)
        start_time = time.time()
        mode_predictor.train(mode_prediction_data_t, clustered_labels_t_s, labels)
        svm_train_time = time.time() - start_time
        moe_results['svm_train_time'] = svm_train_time
        print 'SVM training time:', svm_train_time
        exp_data['mode_predictor'] = deepcopy(mode_predictor)
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'mode_predictor' not in exp_data:
            assert (False)
        else:
            mode_predictor = exp_data['mode_predictor']

    yumiKin = YumiKinematics(kin_params)

    # long-term prediction for MoE method
    # pol = Policy(agent_hyperparams, exp_params_rob)
    pol = SimplePolicy(Xrs_data, Us_data, exp_params_data)

    ugp_experts_dyn = UGP(dX + dU, **ugp_params)
    ugp_experts_pol = UGP(dX, **ugp_params)

    x_mu_t = exp_data['X0_mu']
    x_var_t = np.diag(exp_data['X0_var'])
    # x_var_t[0, 0] = 1e-6
    # x_var_t[1, 1] = 1e-6  # TODO: cholesky failing for zero v0 variance
    # ex_mu_t = exp_data['EX0_mu']

    # mode0 = dpgmm.predict(x_mu_t.reshape(1, -1)) # TODO: vel multiplier?
    mode0 = clustered_labels[0]  # TODO: vel multiplier?
    mode0 = np.asscalar(mode0)
    mc_sample_size = (dX + dU) * mc_factor  # TODO: put this param in some proper place
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
            # u_mu_t = XU_t_train_avg[t, dX:]
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
            # ext_s = yumiKin.forward(xtut_s[:, :dX])
            # ut_s = xtut_s[:,dX:]
            # extut_s = np.concatenate((ext_s, ut_s), axis=1)
            # extut_s_std = EXU_scaler.transform(extut_s)
            # clf = SVMs[md]
            # mode_dst = clf.predict(extut_s_std)
            mode_dst = mode_predictor.predict(xtut_s, md)
            mode_counts = Counter(mode_dst).items()
            # mode_counts_ = copy.deepcopy(mode_counts)
            # for i in range(len(mode_counts_)):
            #     if mode_counts_[i][1] < min_mc_particles:
            #         del(mode_counts[i])

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
                        # mode_pred_dict[mod[0]]['var'] = np.diag(np.full(dX+dU, 1e-6))
                        mode_pred_dict[mod[0]]['var'] = np.diag(np.concatenate((p_noise_var, v_noise_var, np.full(dU, 1e-6))))   # TODO: check this again and update in blocks
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
                if p_next > prob_min:
                    # get the next state
                    if md_next == md:
                        md_ = md_prev
                        pi_next = pi
                        gp = experts[md]
                        if not delta_model:
                            x_mu_t_next_new, x_var_t_next_new, _, _, _ = ugp_experts_dyn.get_posterior(gp, xu_mu_t, xu_var_t)
                        else:
                            dx_mu_t_next_new, dx_var_t_next_new, _, _, xudx_covar = ugp_experts_dyn.get_posterior(gp,
                                                                                                                  xu_mu_t,
                                                                                                                  xu_var_t)
                            dx_var_t_next_new = dx_var_t_next_new + np.eye(dX,dX)*jitter_var_tl
                            xdx_covar = xudx_covar[:dX, :]
                            x_mu_t_next_new = x_mu_t + dx_mu_t_next_new
                            x_var_t_next_new = x_var_t + dx_var_t_next_new + xdx_covar + xdx_covar.T
                            # x_var_t_next_new = x_var_t + dx_var_t
                            # x_var_t_next_new = x_var_t
                    else:
                        md_ = md
                        gp_trans = trans_dicts[(md, md_next)]['mdgp']
                        # x_mu_t_next_new, x_var_t_next_new, _, _, _ = ugp_experts_dyn.get_posterior(gp_trans, xu_mu_t,
                        #                                                                            xu_var_t)
                        xu_var_s_= xu_var_s_ + np.eye(dX+dU) * jitter_var_tl
                        x_mu_t_next_new, x_var_t_next_new, _, _, _ = ugp_experts_dyn.get_posterior(gp_trans, xu_mu_s_, xu_var_s_)
                        # exp_params_ = deepcopy(exp_params_rob)
                        # exp_params_['x0'] = x_mu_t_next_new
                        # pi_next = Policy(agent_hyperparams, exp_params_)
                        pi_next = pi
                    assert (len(sim_data_tree) == t + 2)
                    tracks_next = sim_data_tree[t + 1]
                    if len(tracks_next)==0:
                        if p*p_next > prob_min:
                            sim_data_tree[t+1].append([md_next, md_, x_mu_t_next_new, x_var_t_next_new, 0., 0., p*p_next, pi_next])
                    else:
                        md_next_curr_list = [track_next[0] for track_next in tracks_next]
                        if md_next not in md_next_curr_list:
                            # md_next not already in the t+1 time step
                            if p * p_next > prob_min:
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
                                if p * p_next > prob_min:
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
                                            if p_next_comb > prob_min:
                                                sim_data_tree[t + 1][it] = \
                                                    [md_next, md_, mu_next_comb, var_next_comb, 0., 0., p_next_comb, pi_next]
                                    it+=1

        # probability check
        prob_mode_tot = 0.
        for track_ in sim_data_tree[t]:
                prob_mode_tot += track_[6]
        if (prob_mode_tot - 1.0) > 1e-4:
            assert (False)

    moe_pred_time = time.time() - start_time
    moe_results['moe_pred_time'] = moe_pred_time
    print 'Prediction time for MoE UGP with horizon', H, ':', moe_pred_time
    moe_results['track_data'] = sim_data_tree

    # plot each path (in mode) separately
    # path is assumed to be a path arising out from a unique transtions
    # different paths arising out of the same transition at different time is allowed in our model not here
    tm = np.array(range(H)) * dt

    path_dict = {}
    col_mode = np.zeros((H, 3))
    for i in range(H):
        t = tm[i]
        tracks = sim_data_tree[i]
        prob = 0.
        label_mode=np.zeros(H, dtype=int)
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
            p = track[6]
            if p>prob:
                prob = p
                col_mode[i] = label_col_dict[path[0]]
                label_mode[i] = path[0]
    moe_results['path_data'] = path_dict
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
        prob = np.clip(prob, 0., 1.)
        col = np.tile(path['col'], (time.shape[0],1))
        rbga_col = np.concatenate((col, prob), axis=1)
        for j in range(dP):
            plt.subplot(3, 7, 1 + j)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Joint Pos (rad)')
            plt.title('j%dPos' % (j + 1))
            plt.scatter(time, pos[:, j], color=rbga_col, s=3, marker='s')
            plt.fill_between(time, pos[:, j] - pos_std[:, j] * 1.96, pos[:, j] + pos_std[:, j] * 1.96,
                             alpha=0.2, color=rbga_col)
            for i in range(n_test):
                x = Xs_t_test[i, :, j]
                cl = cols[i]
                plt.scatter(tm, x, alpha=0.1, color=cl, s=1)
        for j in range(dV):
            plt.subplot(3, 7, 8 + j)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Joint Pos (rad)')
            plt.title('j%dVel' % (j + 1))
            plt.scatter(time, vel[:, j], color=rbga_col, s=3, marker='s')
            plt.fill_between(time, vel[:, j] - vel_std[:, j] * 1.96, vel[:, j] + vel_std[:, j] * 1.96,
                             alpha=0.2, color=rbga_col)
            for i in range(n_test):
                x = Xs_t_test[i, :, dP+j]
                cl = cols[i]
                plt.scatter(tm, x, alpha=0.1, color=cl, s=1)
        for j in range(dU):
            plt.subplot(3, 7, 15 + j)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Joint Pos (rad)')
            plt.title('j%dTrq' % (j + 1))
            plt.scatter(time, trq[:, j], color=rbga_col, s=3, marker='s')
            plt.fill_between(time, trq[:, j] - trq_std[:, j] * 1.96, trq[:, j] + trq_std[:, j] * 1.96,
                             alpha=0.2, color=rbga_col)
            for i in range(n_test):
                u = XUs_t_test[i, :, dX+j]
                cl = cols[i]
                plt.scatter(tm, u, alpha=0.1, color=cl, s=1)
        plt.show(block=False)

    # plot only mode of multimodal dist
    tm = np.array(range(H))
    P_mu = np.zeros((H, dP))
    V_mu = np.zeros((H, dV))
    Xs_mu_pred = []
    for t in range(H):
        tracks = sim_data_tree[t]
        xp_pairs = [[track[2], track[6]] for track in tracks]
        xs = [track[2] for track in tracks]
        Xs_mu_pred.append(xs)
        xp_max = max(xp_pairs, key=lambda x: x[1])
        P_mu[t] = xp_max[0][:dP]
        V_mu[t] = xp_max[0][dP:dP+dV]

    plt.figure()
    for j in range(dP):
        plt.subplot(2, 7, 1 + j)
        plt.title('j%dPos' % (j + 1))
        plt.plot(tm, P_mu[:, j])
        for i in range(n_test):
            x = Xs_t_test[i, :, j]
            cl = cols[i]
            plt.scatter(tm, x, alpha=0.1, color=cl, s=1)

    for j in range(dV):
        plt.subplot(2, 7, 8 + j)
        plt.title('j%dVel' % (j + 1))
        plt.plot(tm, V_mu[:, j])
        for i in range(n_test):
            x = Xs_t_test[i, :, dP + j]
            cl = cols[i]
            plt.scatter(tm, x, alpha=0.1, color=cl, s=1)
    plt.show(block=False)


    # # compute long-term prediction score
    # XUs_t_test = XUs_test_data
    # assert (XUs_t_test.shape[0] == n_test)
    # X_test_log_ll = np.zeros((H, n_test))
    # for i in range(n_test):
    #     XU_test = XUs_t_test[i]
    #     for t in range(H):
    #         x_t = XU_test[t, :dX]
    #         tracks = sim_data_tree[t]
    #         prob_mix = 0.
    #         for track in tracks:
    #             prob_mix += sp.stats.multivariate_normal.pdf(x_t, track[2], track[3])*track[6]
    #         X_test_log_ll[t, i] = np.log(prob_mix)

    # compute long-term prediction score with logsum
    XUs_t_test= XUs_test_data
    n_test, _, _ = XUs_t_test.shape
    # assert (XUs_t_test.shape[0] == n_test)
    X_test_log_ll = np.zeros((H, n_test))
    X_test_rmse = np.zeros((H, n_test))
    for i in range(n_test):
        XU_test = XUs_t_test[i]
        for t in range(H):
            x_t = XU_test[t, :dX]
            x_t = x_t.reshape(-1)
            tracks = sim_data_tree[t]
            log_prob_track_t = np.zeros(len(tracks))
            for k in range(len(tracks)):
                track = tracks[k]
                x_mu = track[2]
                x_var = track[3]
                # x_var = track[3] + np.eye(dX) * jitter_var_tl
                x_var = np.diag(np.diag(x_var))
                log_prob_track_t[k] = sp.stats.multivariate_normal.logpdf(x_t, x_mu, x_var) + np.log(track[6])
            X_test_log_ll[t, i] = logsum(log_prob_track_t)
            max_comp_id = np.argmax(log_prob_track_t)
            track_max = tracks[max_comp_id]
            x_mu_pred = track_max[2].reshape(-1)
            X_test_rmse[t, i] = np.dot((x_mu_pred - x_t), (x_mu_pred - x_t))

    tm = np.array(range(H)) * dt
    plt.figure()
    plt.title('Average NLL of test trajectories w.r.t time ')
    plt.xlabel('Time, s')
    plt.ylabel('NLL')
    plt.plot(tm.reshape(H, 1), np.mean(X_test_log_ll, axis=1).reshape(H, 1))

    tm = np.array(range(H)) * dt
    plt.figure()
    plt.title('Average RMSE of test trajectories w.r.t time ')
    plt.xlabel('Time, s')
    plt.ylabel('NLL')
    plt.plot(tm.reshape(H, 1), np.mean(X_test_rmse, axis=1).reshape(H, 1))
    # plt.plot(tm.reshape(H, 1), X_test_rmse.reshape(H, -1))

    nll_mean = np.mean(X_test_log_ll.reshape(-1))
    nll_std = np.std(X_test_log_ll.reshape(-1))
    rmse = np.sqrt(np.mean(X_test_rmse.reshape(-1)))
    # rmse = np.mean(X_test_rmse.reshape(-1))
    print 'YUMI exp MOE, NLL mean: ', nll_mean, 'NLL std: ', nll_std, 'RMSE', rmse
    moe_results['rmse'] = rmse
    moe_results['nll'] = (nll_mean, nll_std)
    pickle.dump(moe_results, open(moe_result_file, "wb"))

plt.show(block=False)
None




