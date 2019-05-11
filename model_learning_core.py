import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mat_col
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from model_leraning_utils import get_N_HexCol
from collections import Counter
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn import mixture
# from multidim_gp import MultidimGP
from multidim_gp import MdGpyGP
from multidim_gp import MdGpyGPwithNoiseEst
from model_leraning_utils import UGP
from model_leraning_utils import dummySVM
from model_leraning_utils import SVMmodePredictionGlobal as SVMmodePrediction
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from copy import deepcopy
import operator
#import datetime
import time
from itertools import compress
import pickle
from blocks_sim import MassSlideWorld
from model_leraning_utils import print_experts_gp, print_global_gp, print_transition_gp
from model_leraning_utils import traj_with_moe, traj_with_globalgp
MgGP_global_gp = MdGpyGPwithNoiseEst
MgGP_expert_gp = MdGpyGPwithNoiseEst
MgGP_trans_gp = MdGpyGPwithNoiseEst


# np.random.seed(4)         # good result for the new blocks exp and with noise estimation
# np.random.seed(1)
plt.rcParams.update({'font.size': 15})
# logfile = "./Results/blocks_exp_preprocessed_data_rs_1.dat"
# logfile = "./Results/blocks_exp_preprocessed_data_rs_1.p"     # with global gp saved, scikit_gp
# logfile = "./Results/blocks_exp_preprocessed_data_rs_1_gpy.p"
logfile = "./Results/blocks_exp_preprocessed_data_rs_1_mm.p"


blocks_exp = True
mjc_exp = False
yumi_exp = False

load_all = False

global_gp = True
delta_model = False
load_gp = load_all
load_dpgmm = load_all
load_transition_gp = load_all
load_experts = load_all
load_svms = load_all

fit_moe = True
gp_shuffle_data = False
min_prob_grid = 0.001 # 1%
grid_size = 0.005
# p_noise_var = 0.0026
# p_noise_var = 0.
p_noise_var = 1e-5
# v_noise_var = 1e-3
# v_noise_var = 0.0326
# v_noise_var = 0.
v_noise_var = 1e-4
prob_min = 1e-3
mc_factor = 10
num_tarj_samples = 100

exp_data = pickle.load( open(logfile, "rb" ) )
gp_file = open('./heuristics_gp_params_file', 'w+')
# gp_file = open('./original_gp_params_file', 'w+')

exp_params = exp_data['exp_params']
# Xg = exp_data['Xg']  # sate ground truth
# Ug = exp_data['Ug']  # action ground truth
dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']
dX = dP+dV
T = exp_params['T'] - 1
dt = exp_params['dt']
n_train = exp_data['n_train']
n_test = exp_data['n_test']

XUs_t_train = exp_data['XUs_t_train']
XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1])
XU_scaler = StandardScaler().fit(XU_t_train)
XU_t_std_train = XU_scaler.transform(XU_t_train)

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

XUs_t_test = exp_data['XUs_t_test']

ugp_params = {
    'alpha': 1.,
    'kappa': 2.,
    'beta': 0.,
}

# K = X_t_std_weighted_train.shape[0] // 3
dpgmm_params = {
    'n_components': 10,  # cluster size
    'covariance_type': 'full',
    'tol': 1e-6,
    'n_init': 10,
    'max_iter': 300,
    'weight_concentration_prior_type': 'dirichlet_process',
    'weight_concentration_prior': 1e-2,
    'mean_precision_prior': None,
    'mean_prior': None,
    'degrees_of_freedom_prior': 2 + 2,
    'covariance_prior': None,
    'warm_start': False,
    'init_params': 'random',
}

policy_params = exp_params['policy'] # TODO: the block_sim code assumes only 'm1' mode for control
expl_noise = policy_params['m1']['noise_pol']
# expl_noise = 3.
H = T  # prediction horizon

if global_gp:
    # gpr_params = {
    #     # 'alpha': 1e-2,  # alpha=0 when using white kernal
    #     'alpha': 0.,  # alpha=0 when using white kernal
    #     'kernel': C(1.0, (1e-2, 1e2)) * RBF(np.ones(dX + dU), (1e-1, 1e1)) + W(noise_level=1.,
    #                                                                            noise_level_bounds=(1e-4, 1e1)),
    #     'n_restarts_optimizer': 10,
    #     'normalize_y': False,  # is not supported in the propogation function
    # }

    gpr_params = {
        'noise_var': np.array([p_noise_var, v_noise_var]),
        'normalize': True,
    }

    # global gp fit
    if not load_gp:
        mdgp_glob = MgGP_global_gp(gpr_params, dX)
        start_time = time.time()
        if not delta_model:
            mdgp_glob.fit(XU_t_train, X_t1_train)
        else:
            mdgp_glob.fit(XU_t_train, dX_t_train)
        print 'Global GP fit time', time.time() - start_time
        exp_data['mdgp_glob'] = deepcopy(mdgp_glob)
        print_global_gp(mdgp_glob, gp_file)
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'mdgp_glob' not in exp_data:
            assert(False)
        else:
            mdgp_glob = exp_data['mdgp_glob']

    # global gp long-term prediction
    massSlideParams = exp_params['massSlide']
    # policy_params = exp_params['policy']
    massSlideWorld = MassSlideWorld(**massSlideParams)
    massSlideWorld.set_policy(policy_params)
    massSlideWorld.reset()
    mode = 'm1'  # only one mode for control no matter what X

    ugp_global_dyn = UGP(dX + dU, **ugp_params)
    ugp_global_pol = UGP(dX, **ugp_params)

    x_mu_0 = exp_data['X0_mu']
    x_mu_t = x_mu_0
    # x_mu_t = exp_data['X0_mu'] + 0.5
    x_var_0 = np.diag(exp_data['X0_var'])
    x_var_0[1, 1] = 1e-6  # TODO: cholesky failing for zero v0 variance
    x_var_t = x_var_0
    # x_var_t[0, 0] = 1e-6

    Y_mu = np.zeros((2*(dX + dU) + 1, dX))
    X_mu_pred = []
    X_var_pred = []
    X_particles = []
    start_time = time.time()
    for t in range(H):
        x_t = np.random.multivariate_normal(x_mu_t, x_var_t)
        if blocks_exp:
            # _, u_mu_t, u_var_t = massSlideWorld.act(x_t, mode)
            # _, u_mu_t, u_var_t = massSlideWorld.act(x_mu_t, mode)
            u_mu_t, u_var_t, _, _, xu_cov = ugp_global_pol.get_posterior(massSlideWorld, x_mu_t, x_var_t)
        xu_mu_t = np.append(x_mu_t, u_mu_t)
        # xu_var_t = np.block([[x_var_t, np.zeros((dX,dU))],
        #                     [np.zeros((dU,dX)), u_var_t]])
        xu_var_t = np.block([[x_var_t, xu_cov],
                             [xu_cov.T, u_var_t]])
        X_mu_pred.append(x_mu_t)
        X_var_pred.append(x_var_t)
        X_particles.append(Y_mu)
        if not delta_model:
            x_mu_t, x_var_t, Y_mu, _, _ = ugp_global_dyn.get_posterior(mdgp_glob, xu_mu_t, xu_var_t)
        else:
            dx_mu_t, dx_var_t, dY_mu, _, xudx_covar = ugp_global_dyn.get_posterior(mdgp_glob, xu_mu_t, xu_var_t)
            xdx_covar = xudx_covar[:dX, :]
            x_mu_t = X_mu_pred[t] + dx_mu_t
            x_var_t = X_var_pred[t] + dx_var_t + xdx_covar + xdx_covar.T
            # Y_mu = X_particles[t] + dY_mu
    print 'Global GP prediction time for horizon', H, ':', time.time() - start_time

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
    print 'NLL mean (um): ', nll_mean, 'NLL std (um): ', nll_std

    X_mu_pred = np.array(X_mu_pred)
    P_sig_pred = np.zeros(H)
    V_sig_pred = np.zeros(H)
    P_sigma_points = np.zeros((2*(dX+dU) + 1,H))
    V_sigma_points = np.zeros((2 * (dX+dU) + 1, H))
    for t in range(H):
        P_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[0])
        V_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[1])

    P_mu_pred = X_mu_pred[:, :dP].reshape(-1)
    V_mu_pred = X_mu_pred[:, dP:].reshape(-1)

    for t in range(0,H):
        P_sigma_points[:, t] = X_particles[t][:, 0]
        V_sigma_points[:, t] = X_particles[t][:, 1]

    # tm = np.array(range(H)) * dt
    tm = np.array(range(H))
    Xs_t_test = XUs_t_test[:, :, :dX]
    plt.figure()
    plt.title('Long-term prediction with GP')
    plt.subplot(121)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.plot(tm, P_mu_pred, marker='s', label='Pos mean', color='g', linewidth='2')
    plt.fill_between(tm, P_mu_pred - P_sig_pred * 1.96, P_mu_pred + P_sig_pred * 1.96, alpha=0.2, color='g')
    # plt.plot(tm, Xg[:H,0], linewidth='2')
    plt.plot(tm, Xs_t_test[0, :H, :dP], ls='--', color='g', alpha=0.2, label='Training data')
    for i in range(1, n_test):
        plt.plot(tm, Xs_t_test[i, :H, :dP], ls='--', color='g', alpha=0.2)
    # for p in P_sigma_points:
    #     plt.scatter(tm, p, marker='+')
    plt.legend()
    plt.subplot(122)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.plot(tm, V_mu_pred, marker='s', label='Vel mean', color='b', linewidth='2')
    plt.fill_between(tm, V_mu_pred - V_sig_pred * 1.96, V_mu_pred + V_sig_pred * 1.96, alpha=0.2, color='b')
    # plt.plot(tm, Xg[:H, 1], linewidth='2')
    plt.plot(tm, Xs_t_test[0, :H, dP:], ls='--', color='b', alpha=0.2, label='Training data')
    for i in range(1, n_test):
        plt.plot(tm, Xs_t_test[i, :H, dP:], ls='--', color='b', alpha=0.2)
    # for p in V_sigma_points:
    #     plt.scatter(tm, p, marker='+')
    plt.legend()
    plt.savefig('gp_long-term.pdf')

    massSlideWorld.reset()
    num_samples = num_tarj_samples
    traj_with_globalgp_ = traj_with_globalgp(x_mu_0, x_var_0, mdgp_glob, massSlideWorld)
    _ = traj_with_globalgp_.sample(num_samples, H)
    traj_with_globalgp_.plot_samples()
    params = deepcopy(dpgmm_params)
    params['n_components'] = 2
    params['n_init'] = 3
    nll_mean, nll_std =  traj_with_globalgp_.estimate_gmm_traj_density(params, Xs_t_test)
    print 'NLL mean (mm): ', nll_mean, 'NLL std (mm): ', nll_std

    plt.show(block=False)
if fit_moe:
    if not load_dpgmm:
        dpgmm = mixture.BayesianGaussianMixture(**dpgmm_params)
        start_time = time.time()
        dpgmm.fit(X_t_std_weighted_train)
        print 'DPGMM clustering time:', time.time() - start_time
        print 'Converged DPGMM', dpgmm.converged_, 'on', dpgmm.n_iter_, 'iterations with lower bound', dpgmm.lower_bound_
        exp_data['dpgmm'] = deepcopy(dpgmm)
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'dpgmm' not in exp_data:
            assert (False)
        else:
            dpgmm = exp_data['dpgmm']
    dpgmm_Xt_train_labels = dpgmm.predict(X_t_std_weighted_train)
    dpgmm_Xt1_train_labels = dpgmm.predict(X_t1_std_weighted_train)

    # get labels and counts
    labels, counts = zip(*sorted(Counter(dpgmm_Xt_train_labels).items(), key=operator.itemgetter(0)))
    K = len(labels)
    colors = np.zeros((K,4))
    colors = get_N_HexCol(K)
    colors = np.asarray(colors) / 255.
    # colors_itr = iter(cm.rainbow(np.linspace(0, 1, K)))
    # for i in range(K):
    #     colors[i] = next(colors_itr)
    # colors=colors[:,:3]

    marker_set = ['.', 'o', '*', '+', '^', 'x', 'o', 'D', 's']
    marker_set_size = len(marker_set)
    if K < marker_set_size:
        markers = marker_set[:K]
    else:
        markers = ['o'] * K
    # plot cluster components
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.bar(labels, counts, color=colors)
    # plt.title('DPGMM clustering')
    plt.ylabel('Cluster sizes')
    plt.xlabel('Cluster labels')
    plt.savefig('dpgmm_blocks_cluster counts.pdf')
    # plt.savefig('dpgmm_1d_dyn_cluster counts.png', format='png', dpi=1000)

    # plot clustered trajectory
    col = np.zeros([X_t_train.shape[0], 3])
    mark = np.array(['None'] * X_t_train.shape[0])
    i = 0
    for label in labels:
        col[(dpgmm_Xt_train_labels == label)] = colors[i]
        mark[(dpgmm_Xt_train_labels == label)] = markers[i]
        i += 1

    label_col_dict = d = dict(zip(labels, colors))

    col = col.reshape(n_train, -1, 3)
    mark = mark.reshape(n_train, -1)
    plt.figure()
    plt.title('Clustered train trajectories')
    plt.subplot(211)
    for i in range(XUs_t_train.shape[0]):
        for j in range(XUs_t_train.shape[1]):
            plt.scatter(tm[j], XUs_t_train[i, j, :dP], c=col[i, j], marker=mark[i, j])
    plt.xlabel('Time, t')
    plt.ylabel('Position, m')
    plt.subplot(212)
    for i in range(XUs_t_train.shape[0]):
        for j in range(XUs_t_train.shape[1]):
            plt.scatter(tm[j], XUs_t_train[i, j, dP:dP+dV], c=col[i, j], marker=mark[i, j])
    plt.xlabel('Time, t')
    plt.ylabel('Velocity, m/s')
    plt.savefig('clustered_trajs.pdf')
    plt.show(block=False)

    if not load_transition_gp:
        # transition GP
        trans_gpr_params = gpr_params
        # trans_gpr_params = {
        #     # 'alpha': 1e-2,  # alpha=0 when using white kernal
        #     'alpha': 0.,  # alpha=0 when using white kernal
        #     'kernel': C(1.0, (1e-2, 1e2)) * RBF(np.ones(dX + dU), (1e-1, 1e1)) + W(noise_level=1.,
        #                                                                            noise_level_bounds=(1e-4, 1e-2)),
        #     # 'kernel': C(1.0, (1e-1, 1e1)) * RBF(np.ones(dX + dU), (1e-1, 1e1)),
        #     'n_restarts_optimizer': 10,
        #     'normalize_y': False,  # is not supported in the propogation function
        # }

        trans_dicts = {}
        start_time = time.time()
        for xu in XUs_t_train:
            x = xu[:, :dX]
            x_std = X_scaler.transform(x)
            x_labels = dpgmm.predict(x_std)
            iddiff = x_labels[:-1] != x_labels[1:]
            trans_data = zip(tm[:-1], xu[:-1, :dX + dU], xu[1:, :dX], x_labels[:-1], x_labels[1:])
            trans_data_p = list(compress(trans_data, iddiff))
            for t, xu_, y, xid, yid in trans_data_p:
                if (xid, yid) not in trans_dicts:
                    trans_dicts[(xid, yid)] = {'t': [], 'XU': [], 'Y': [], 'mdgp': None}
                trans_dicts[(xid, yid)]['XU'].append(xu_)
                trans_dicts[(xid, yid)]['Y'].append(y)
                trans_dicts[(xid, yid)]['t'].append(t)
        for trans_data in trans_dicts:
            XU = np.array(trans_dicts[trans_data]['XU']).reshape(-1, dX + dU)
            trans_dicts[trans_data]['XU'] = XU
            Y = np.array(trans_dicts[trans_data]['Y']).reshape(-1, dX)
            trans_dicts[trans_data]['Y'] = Y
            mdgp = MgGP_trans_gp(trans_gpr_params, Y.shape[1])
            mdgp.fit(XU, Y)
            trans_dicts[trans_data]['mdgp'] = deepcopy(mdgp)
            del mdgp
        print 'Transition GP training time:', time.time() - start_time
        exp_data['transition_gp'] = deepcopy(trans_dicts)
        # print_transition_gp(trans_dicts, gp_file)
        pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'transition_gp' not in exp_data:
            assert(False)
        else:
            trans_dicts = exp_data['transition_gp']

    plt.figure()
    # plt.title('Transition points')
    plt.title('Trial trajectories')
    plt.subplot(211)
    for xu in XUs_t_train:
        plt.plot(tm, xu[:,0])
    for trans_data in trans_dicts:
        trans_t = np.array(trans_dicts[trans_data]['t'])
        trans_p = trans_dicts[trans_data]['XU'][:,0]
        trans_p1 = trans_dicts[trans_data]['Y'][:, 0]
        plt.scatter(trans_t, trans_p)
        plt.scatter(trans_t + 1., trans_p1)
    plt.xlabel('Time, t')
    plt.ylabel('Position, m')
    plt.subplot(212)
    for xu in XUs_t_train:
        plt.plot(tm, xu[:, 1])
    for trans_data in trans_dicts:
        trans_t = np.array(trans_dicts[trans_data]['t'])
        trans_v = trans_dicts[trans_data]['XU'][:,1]
        trans_v1 = trans_dicts[trans_data]['Y'][:, 1]
        plt.scatter(trans_t, trans_v)
        plt.scatter(trans_t + 1., trans_v1)
    plt.xlabel('Time, t')
    plt.ylabel('Velocity, m/s')
    plt.savefig('transition_points.pdf')
    plt.show(block=False)


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

        experts = {}
        start_time = time.time()
        for label in labels:
            x_train = XU_t_train[(np.logical_and((dpgmm_Xt_train_labels == label), (dpgmm_Xt1_train_labels == label)))]
            y_train = X_t1_train[(np.logical_and((dpgmm_Xt_train_labels == label), (dpgmm_Xt1_train_labels == label)))]
            mdgp = MgGP_expert_gp(expert_gpr_params, y_train.shape[1])
            mdgp.fit(x_train, y_train)
            experts[label] = deepcopy(mdgp)
            del mdgp
        print 'Experts training time:', time.time() - start_time
        exp_data['experts'] = deepcopy(experts)
        print_experts_gp(experts, gp_file)
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
        start_time = time.time()

        # SVMs = {}
        # XUs_t_std_train = XU_t_std_train.reshape(n_train, T, -1)
        # dpgmm_Xts_train_labels = dpgmm_Xt_train_labels.reshape(n_train, T)
        # XUnI_svm = []
        # dpgmm_Xts_train_labels_svm = []
        # for i in range(n_train):
        #     xu_t_std_train = XUs_t_std_train[i]
        #     dpgmm_xt_train_labels = dpgmm_Xts_train_labels[i]
        #     dpgmm_Xts_train_labels_svm.extend(dpgmm_xt_train_labels[:-1])
        #     xuni = zip(xu_t_std_train[:-1, :], dpgmm_xt_train_labels[1:])
        #     XUnI_svm.extend(xuni)
        # dpgmm_Xts_train_labels_svm = np.array(dpgmm_Xts_train_labels_svm)
        # for label in labels:
        #     xui = list(compress(XUnI_svm, (dpgmm_Xts_train_labels_svm == label)))
        #     xu, i = zip(*xui)
        #     xu = np.array(xu)
        #     i = list(i)
        #     cnts_list = Counter(i).items()
        #     svm_check_ok = True
        #     for cnts in cnts_list:
        #         if cnts[1] < svm_grid_params['cv']:
        #             svm_check_ok = True #TODO: this check is disabled.
        #     if len(cnts_list)>1 and svm_check_ok==True:
        #         clf = GridSearchCV(SVC(**svm_params), **svm_grid_params)
        #         clf.fit(xu, i)
        #         SVMs[label] = deepcopy(clf)
        #         del clf
        #     else:
        #         print 'detected dummy svm:', label
        #         dummy_svm = dummySVM(cnts_list[0][0])
        #         SVMs[label] = deepcopy(dummy_svm)
        #         del dummy_svm
        #
        # print 'SVMs training time:', time.time() - start_time
        dpgmm_Xts_train_labels = dpgmm_Xt_train_labels.reshape(n_train, T)
        mode_predictor = SVMmodePrediction(svm_grid_params, svm_params)
        mode_predictor.train(XUs_t_train, dpgmm_Xts_train_labels, labels)
        exp_data['mode_predictor'] = deepcopy(mode_predictor)
        pickle.dump(exp_data, open(logfile, "wb"))

        # exp_data['svm'] = deepcopy(SVMs)
        # pickle.dump(exp_data, open(logfile, "wb"))
    else:
        if 'mode_predictor' not in exp_data:
            assert (False)
        else:
            mode_predictor = exp_data['mode_predictor']

    # long-term prediction for MoE method
    if blocks_exp:
        massSlideParams = exp_params['massSlide']
        # policy_params = exp_params['policy']
        massSlideWorld = MassSlideWorld(**massSlideParams)
        massSlideWorld.set_policy(policy_params)
        massSlideWorld.reset()
        mode = 'm1'  # only one mode for control no matter what X

    ugp_experts_dyn = UGP(dX + dU, **ugp_params)
    ugp_experts_pol = UGP(dX, **ugp_params)

    x_mu_t = exp_data['X0_mu']
    # x_mu_t = exp_data['X0_mu'] + 0.5
    x_var_t = np.diag(exp_data['X0_var'])
    # x_var_t[0, 0] = 1e-6
    x_var_t[1, 1] = 1e-6  # TODO: cholesky failing for zero v0 variance
    x_mu_t_std = X_scaler.transform(x_mu_t.reshape(1, -1))
    mode0 = dpgmm.predict(x_mu_t_std.reshape(1, -1))
    mode0 = np.asscalar(mode0)
    mc_sample_size = (dX + dU) * mc_factor  # TODO: put this param in some proper place
    num_modes = len(labels)
    modes = labels
    Y_mu = np.zeros((2 * (dX + dU) + 1, dX))
    X_mu_pred = []
    X_var_pred = []
    X_particles = []
    sim_data_tree = [[[mode0, -1, x_mu_t, x_var_t, None, None, 1.]]]
    start_time = time.time()
    for t in range(H):
        # print(t)
        tracks = sim_data_tree[t]
        for track in tracks:
            md, md_prev, x_mu_t, x_var_t, _, _, p = track
            if blocks_exp:
                u_mu_t, u_var_t, _, _, xu_cov = ugp_experts_pol.get_posterior(massSlideWorld, x_mu_t, x_var_t)
            xu_mu_t = np.append(x_mu_t, u_mu_t)
            # xu_var_t = np.block([[x_var_t, np.zeros((dX,dU))],
            #                     [np.zeros((dU,dX)), u_var_t]])
            xu_var_t = np.block([[x_var_t, xu_cov],
                                 [xu_cov.T, u_var_t]])
            track[4] = u_mu_t
            track[5] = u_var_t
            xtut_s = np.random.multivariate_normal(xu_mu_t, xu_var_t, mc_sample_size)
            assert (xtut_s.shape == (mc_sample_size, dX + dU))
            # xtut_s_std = XU_scaler.transform(xtut_s)
            # clf = SVMs[md]
            # mode_dst = clf.predict(xtut_s_std)
            mode_dst = mode_predictor.predict(xtut_s, md)
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
                if p_next > prob_min:
                    # get the next state
                    if md_next == md:
                        gp = experts[md]
                        x_mu_t_next_new, x_var_t_next_new, _, _, _ = ugp_experts_dyn.get_posterior(gp, xu_mu_t, xu_var_t)
                    else:
                        gp_trans = trans_dicts[(md, md_next)]['mdgp']
                        # x_mu_t_next_new, x_var_t_next_new, _, _, _ = ugp_experts_dyn.get_posterior(gp_trans, xu_mu_t,
                        #                                                                            xu_var_t)
                        xu_var_s_= xu_var_s_ + np.diag(np.diag(xu_var_s_) + 1e-6)
                        x_mu_t_next_new, x_var_t_next_new, _, _, _ = ugp_experts_dyn.get_posterior(gp_trans, xu_mu_s_, xu_var_s_)
                    assert (len(sim_data_tree) == t + 2)
                    tracks_next = sim_data_tree[t + 1]
                    if md == md_next:
                        md_ = md_prev
                    else:
                        md_ = md
                    if len(tracks_next)==0:
                        if p*p_next > prob_min:
                            sim_data_tree[t+1].append([md_next, md_, x_mu_t_next_new, x_var_t_next_new, 0., 0., p*p_next])
                    else:
                        md_next_curr_list = [track_next[0] for track_next in tracks_next]
                        if md_next not in md_next_curr_list:
                            # md_next not already in the t+1 time step
                            if p * p_next > prob_min:
                                sim_data_tree[t + 1].append(
                                    [md_next, md_, x_mu_t_next_new, x_var_t_next_new, 0., 0., p * p_next])
                        else:
                            # md_next already in the t+1 time step
                            if md == md_next:
                                md_ = md_prev
                            else:
                                md_ = md
                            md_next_curr_trans_list = [(track_next[1], track_next[0]) for track_next in tracks_next]
                            if (md_, md_next) not in md_next_curr_trans_list:
                                # the same transition track is not present
                                if p * p_next > prob_min:
                                    sim_data_tree[t + 1].append(
                                        [md_next, md_, x_mu_t_next_new, x_var_t_next_new, 0., 0., p * p_next])
                            else:
                                it = 0
                                for track_next in tracks_next:
                                    md_next_curr, md_prev_curr, x_mu_t_next_curr, x_var_t_next_curr, _, _, p_next_curr = track_next
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
                                                    [md_next, md_, mu_next_comb, var_next_comb, 0., 0., p_next_comb]
                                    it+=1

        # probability check
        prob_mode_tot = 0.
        for track_ in sim_data_tree[t]:
            prob_mode_tot += track_[6]
            # print(prob_mode_tot)
        if (prob_mode_tot - 1.0) > prob_min:
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
                path_dict[path] = {'time':[] ,'X':[], 'X_var':[], 'prob':[], 'col':label_col_dict[path[0]]}
            path_dict[path]['time'].append(t)
            path_dict[path]['X'].append(track[2])
            path_dict[path]['X_var'].append(track[3])
            path_dict[path]['prob'].append(track[6])

    # plot probabilities
    tot_prob = np.zeros(H)
    plt.figure()
    for pathkey in path_dict:
        path = path_dict[pathkey]
        t = path['time']
        p = path['prob']
        c = path['col']
        plt.plot(t, p, color=c)
    plt.plot(block=False)

    # plot for tree structure
    # plot long term prediction results of UGP
    plt.figure()
    plt.subplot(121)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.subplot(122)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    for path_key in path_dict:
        path = path_dict[path_key]
        time = np.array(path['time'])
        pos = np.array(path['X'])[:,:dP].reshape(-1)
        pos_std = np.sqrt(np.array(path['X_var'])[:, :dP, :dP]).reshape(time.shape[0])
        vel = np.array(path['X'])[:, dP:dX].reshape(-1)
        vel_std = np.sqrt(np.array(path['X_var'])[:, dP:dX, dP:dX]).reshape(time.shape[0])
        prob = np.array(path['prob']).reshape(-1,1)
        prob = np.clip(prob, 0., 1.)
        col = np.tile(path['col'], (time.shape[0],1))
        rbga_col = np.concatenate((col, prob), axis=1)
        plt.subplot(121)
        plt.scatter(time, pos, c=rbga_col, marker='s', label='M'+str(path_key[0])+' mean')
        plt.fill_between(time, pos - pos_std * 1.96, pos + pos_std * 1.96, alpha=0.2, color=path['col'])
        plt.subplot(122)
        plt.scatter(time, vel, c=rbga_col, marker='s', label='M'+str(path_key[0])+' mean')
        plt.fill_between(time, vel - vel_std * 1.96, vel + vel_std * 1.96, alpha=0.2, color=path['col'])

    # plot training data
    x = Xs_t_test[0]
    plt.subplot(121)
    plt.plot(tm, x[:H, :dP], ls='--', color='k', alpha=0.2, label='Training data')
    plt.subplot(122)
    plt.plot(tm, x[:H, dP:dP + dV], ls='--', color='k', alpha=0.2, label='Training data')
    for x in Xs_t_test[1:]:
        plt.subplot(121)
        plt.plot(tm, x[:H, :dP], ls='--', color='k', alpha=0.2)
        # plt.legend()
        plt.subplot(122)
        plt.plot(tm, x[:H, dP:dP+dV], ls='--', color='k', alpha=0.2)
        # plt.legend()
    plt.savefig('method_result.pdf')

    massSlideWorld.reset()
    num_samples = num_tarj_samples
    traj_with_moe_ = traj_with_moe(sim_data_tree, experts, trans_dicts, massSlideWorld)
    _ = traj_with_moe_.sample(num_samples, H)
    traj_with_moe_.plot_samples()
    params = deepcopy(dpgmm_params)
    params['n_components'] = 2
    params['n_init'] = 3
    traj_with_moe_.estimate_gmm_traj_density(params, Xs_t_test)

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
    #             # if w > prob_min:
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
