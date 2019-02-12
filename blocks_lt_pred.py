import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from multidim_gp import MultidimGP
from model_leraning_utils import UGP
import time
import pickle
from blocks_sim import MassSlideWorld

# np.random.seed(2)

logfile = "./Results/blocks_exp_preprocessed_data.dat"
exp_data = pickle.load( open(logfile, "rb" ) )

exp_params = exp_data['exp_params']
Xg = exp_data['Xg']  # sate ground truth
Ug = exp_data['Ug']  # action ground truth
dP = exp_params['dP'] # pos dim
dV = exp_params['dV'] # vel dim
dU = exp_params['dU'] # act dim
dX = dP+dV # state dim
T = exp_params['T'] - 1 # total time steps
dt = exp_params['dt'] # sampling time
n_train = exp_data['n_train'] # number or trials in training data
n_test = exp_data['n_test'] # number or trials in testing data

XU_t_train = exp_data['XU_t_train'] # shape: n_train*T, dXU, state-action, sequential data
XUs_t_train = exp_data['XUs_t_train'] # shape: n_train, T, dXU, state-action, sequential data
X_t1_train = exp_data['X_t1_train'] # shape: n_train*T, dX, next state, sequential data
X_t_train = exp_data['X_t_train'] # shape: n_train*T, dX, current state, sequential data
X_t_std_weighted_train = exp_data['X_t_std_weighted_train'] # same as X_t_train but standardized
X_t1_std_weighted_train = exp_data['X_t1_std_weighted_train'] # same as X_t1_train but standardized
X_t_test = exp_data['X_t_test']
Xs_t_train = XUs_t_train[:, :, :dX]
X_scaler = exp_data['X_scaler']
XU_t_std_train = exp_data['XU_t_std_train']
XU_scaler = exp_data['XU_scaler']

ugp_params = {
    'alpha': 1.,
    'kappa': 2.,
    'beta': 0.,
}

policy_params = exp_params['policy'] # TODO: the block_sim code assumes only 'm1' mode for control
expl_noise = policy_params['m1']['noise']
H = T  # prediction horizon

gpr_params = {
            # 'alpha': 1e-2,  # alpha=0 when using white kernal
            'alpha': 0.,  # alpha=0 when using white kernal
            'kernel': C(1.0, (1e-2, 1e2)) * RBF(np.ones(dX + dU), (1e-1, 1e1)) + W(noise_level=1.,
                                                                                   noise_level_bounds=(1e-4, 1e-1)),
            # 'kernel': C(1.0, (1e-1, 1e1)) * RBF(np.ones(dX + dU), (1e-1, 1e1)),
            'n_restarts_optimizer': 10,
            'normalize_y': False,  # is not supported in the propogation function
        }

gpr_params_list = []
gpr_params_list.append(gpr_params)
gpr_params_list.append(gpr_params)
# gpr_params_list.append(gpr_params_p_d)
# gpr_params_list.append(gpr_params_v_d)
mdgp_glob = MultidimGP(gpr_params_list, dX) # init a multidim GP with output dim X
start_time = time.time()
mdgp_glob.fit(XU_t_train, X_t1_train)   # fit the gp model with dynamcis data
print 'Global GP fit time', time.time() - start_time

# global gp long-term prediction
massSlideParams = exp_params['massSlide']
# policy_params = exp_params['policy']
massSlideWorld = MassSlideWorld(**massSlideParams)
massSlideWorld.set_policy(policy_params)
massSlideWorld.reset()
mode = 'm1'  # only one mode for control no matter what X

ugp_global_dyn = UGP(dX + dU, **ugp_params) # initialize unscented transform for dynamics
ugp_global_pol = UGP(dX, **ugp_params) # initialize unscented transform for policy

x_mu_t = exp_data['X0_mu']  # mean of initial state distr
x_var_t = np.diag(exp_data['X0_var'])
x_var_t[1,1] = 1e-6   # setting small variance for initial vel    # TODO: cholesky failing for zero v0 variance
X_mu_pred = []  # list for collecting state mean
X_var_pred = [] # list for collecting state var

############ Policy assumptions #######
'''
X # input state
L = np.array([.2, 1.])
Xtrg =  18.
noise = 3.
dX = np.array([Xtrg, 0.]).reshape(1,2) - X
U = np.dot(dX, L) # simple linear controller
U = U.reshape(X.shape[0],1)
if return_std:
    U_noise = np.full((U.shape), np.sqrt(noise))
return U, U_noise
'''

start_time = time.time()
for t in range(H):
    # UT method on stochastic policy, policy is deterministic controller plus exploration noise
    u_mu_t, u_var_t, _, _, xu_cov = ugp_global_pol.get_posterior_pol(massSlideWorld, x_mu_t, x_var_t)
    # form joint state action distribution
    xu_mu_t = np.append(x_mu_t, u_mu_t)
    # xu_var_t = np.block([[x_var_t, np.zeros((dX,dU))],
    #                     [np.zeros((dU,dX)), u_var_t]])
    # TODO: xu_cov may not be correct so disable below and enable above later
    xu_var_t = np.block([[x_var_t, xu_cov],
                         [xu_cov.T, u_var_t]])
    X_mu_pred.append(x_mu_t)
    X_var_pred.append(x_var_t)
    # UT method for one step dynamics prediction
    x_mu_t, x_var_t, _, _, _ = ugp_global_dyn.get_posterior(mdgp_glob, xu_mu_t, xu_var_t)

print 'Global GP prediction time for horizon', H, ':', time.time() - start_time

# compute long-term prediction score using test data
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
plt.figure()
plt.title('NLL of test trajectories w.r.t time ')
plt.xlabel('Time, t')
plt.ylabel('NLL')
plt.plot(tm.reshape(H,1), X_test_log_ll)

nll_mean = np.mean(X_test_log_ll.reshape(-1))
nll_std = np.std(X_test_log_ll.reshape(-1))
print 'NLL mean: ', nll_mean, 'NLL std: ', nll_std

# for plotting purposes
X_mu_pred = np.array(X_mu_pred)
P_sig_pred = np.zeros(H)
V_sig_pred = np.zeros(H)

for t in range(H):
    P_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[0])
    V_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[1])

P_mu_pred = X_mu_pred[:, :dP].reshape(-1)
V_mu_pred = X_mu_pred[:, dP:].reshape(-1)

# tm = np.array(range(H)) * dt
tm = np.array(range(H))
plt.figure()
plt.title('Long-term prediction with GP')
plt.subplot(121)
plt.xlabel('Time, t')
plt.ylabel('Position, m')
# predicted mean
plt.plot(tm, P_mu_pred)
# predicted var
plt.fill_between(tm, P_mu_pred - P_sig_pred * 1.96, P_mu_pred + P_sig_pred * 1.96, alpha=0.2)
# # ground truth
# plt.plot(tm, Xg[:H,0], linewidth='2')
# training trial data
for i in range(n_train):
    plt.plot(tm, Xs_t_train[i, :H, :dP], alpha=0.3)

plt.subplot(122)
plt.xlabel('Time, t')
plt.ylabel('Velocity, m/s')
# predicted mean
plt.plot(tm, V_mu_pred)
# predicted var
plt.fill_between(tm, V_mu_pred - V_sig_pred * 1.96, V_mu_pred + V_sig_pred * 1.96, alpha=0.2)
# # ground truth
# plt.plot(tm, Xg[:H, 1], linewidth='2')
# training trial data
for i in range(n_train):
    plt.plot(tm, Xs_t_train[i, :H, dP:], alpha=0.3)
plt.show()