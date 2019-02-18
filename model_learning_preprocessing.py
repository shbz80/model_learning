import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

blocks_exp = True
mjc_exp = False
yumi_exp = False

# logfile = "_data_disc_rs_1.p"
logfile_ip = "_data_disc_obs_noise_1.p"
# logfile_op = "_data_disc_obs_noise_w2.p"
# logfile_op = "_1.dat"
logfile_op = "_data_disc_obs_noise_1.p"

if blocks_exp:
    # exp_data = pickle.load( open("./Results/blocks_exp_raw_data_rs_4_disc.p", "rb" ) )
    exp_data = pickle.load(open("./Results/blocks_exp_raw"+logfile_ip, "rb"))

exp_params = exp_data['exp_params']
Xs = exp_data['X']  # state
Us = exp_data['U']  # action
Xg = exp_data['Xg']  # sate ground truth
Ug = exp_data['Ug']  # action ground truth

dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']
dX = dP+dV
T = exp_params['T']
dt = exp_params['dt']

n_trials, n_time_steps, dim_state = Xs.shape
_, _, dim_action = Us.shape
assert(dX==dim_state)
assert(n_time_steps==T)
assert(dU==dim_action)

# plot blocks exp
if blocks_exp:
    plt.figure()
    plt.title('Position')
    plt.xlabel('t')
    plt.ylabel('q(t)')
    tm = np.linspace(0,T*dt,T)
    # plot positions
    plt.plot(tm, Xg[:,:dP], ls='-', marker='^')
    for s in range(n_trials):
        plt.plot(tm,Xs[s,:,0])
    plt.figure()
    plt.xlabel('t')
    plt.ylabel('q_dot(t)')
    plt.title('Velocity')
    plt.plot(tm, Xg[:,dP:dP+dV], ls='-', marker='^')
    # plot velocities
    for s in range(n_trials):
        plt.plot(tm,Xs[s,:,1])
    plt.figure()
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.title('Action')
    plt.plot(tm, Ug, ls='-', marker='^')
    # plot actions
    for s in range(n_trials):
        plt.plot(tm,Us[s,:,0])
    plt.show()

n_train = n_trials//3 * 2
n_test = n_trials - n_train
exp_data['n_train'] = n_train
exp_data['n_test'] = n_test
XUs = np.concatenate((Xs, Us), axis=2)
XUs_train = XUs[:n_train, :, :]
XUs_t_train = XUs_train[:,:-1,:]
exp_data['XUs_t_train'] = XUs_t_train
Xs_t1_train = XUs_train[:,1:,:dX]
Xs_t_train = XUs_train[:,:-1,:dX]
dXs_t_train = Xs_t1_train - Xs_t_train

XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1])
exp_data['XU_t_train'] = XU_t_train
X_t1_train = Xs_t1_train.reshape(-1, Xs_t1_train.shape[-1])
exp_data['X_t1_train'] = X_t1_train
X_t_train = Xs_t_train.reshape(-1, Xs_t_train.shape[-1])
exp_data['X_t_train'] = X_t_train
dX_t_train = dXs_t_train.reshape(-1, dXs_t_train.shape[-1])
exp_data['dX_t_train'] = dX_t_train
X_scaler = StandardScaler().fit(X_t_train)
exp_data['X_scaler'] = X_scaler
X_t_std_train = X_scaler.transform(X_t_train)
w_vel = 1.0                               # velocity scaler for giving more weight to velocity
exp_data['w_vel'] = w_vel
X_t_std_weighted_train = X_t_std_train
X_t_std_weighted_train[:, dP:dP+dV] = X_t_std_weighted_train[:, dP:dP+dV] * w_vel
exp_data['X_t_std_weighted_train'] = X_t_std_weighted_train
X_t1_std_train = X_scaler.transform(X_t1_train)
X_t1_std_weighted_train = X_t1_std_train
X_t1_std_weighted_train[:, dP:dP+dV] = X_t1_std_weighted_train[:, dP:dP+dV] * w_vel
exp_data['X_t1_std_weighted_train'] = X_t1_std_weighted_train
XU_scaler = StandardScaler().fit(XU_t_train)
exp_data['XU_scaler'] = XU_scaler
XU_t_std_train = XU_scaler.transform(XU_t_train)
exp_data['XU_t_std_train'] = XU_t_std_train

XUs_test = XUs[n_train:, :, :]
XUs_t_test = XUs_test[:,:-1,:]
exp_data['XUs_t_test'] = XUs_t_test
Xs_t1_test = XUs_test[:,1:,:dX]
Xs_t_test = XUs_test[:,:-1,:dX]
dXs_t_test = Xs_t1_test - Xs_t_test
XU_t_test = XUs_t_test.reshape(-1, XUs_t_test.shape[-1])
exp_data['XU_t_test'] = XU_t_test
X_t1_test = Xs_t1_test.reshape(-1, Xs_t1_test.shape[-1])
exp_data['X_t1_test'] = X_t1_test
X_t_test = Xs_t_test.reshape(-1, Xs_t_test.shape[-1])
exp_data['X_t_test'] = X_t_test
dX_t_test = dXs_t_test.reshape(-1, dXs_t_test.shape[-1])
exp_data['dX_t_test'] = dX_t_test

X0s = XUs_train[:, 0, :dX]
X0_mu = np.mean(X0s, axis=0)
exp_data['X0_mu'] = X0_mu
X0_var = np.var(X0s, axis=0)
exp_data['X0_var'] = X0_var

# pickle.dump(exp_data, open("./Results/blocks_exp_preprocessed_data_rs_4_disc.p", "wb"))
pickle.dump(exp_data, open("./Results/blocks_exp_preprocessed"+logfile_op, "wb"))