import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle
from sklearn.preprocessing import StandardScaler

logfile_ip = "./Results/blocks_exp_raw_data_rs_1.p"
# logfile_ip = "./Results/yumi_exp_preprocessed_data_1.dat"
# logfile_op = "./Results/blocks_exp_preprocessed_data_rs_1.p"
# logfile_op = "./Results/blocks_exp_preprocessed_data_rs_1_gpy.p"
logfile_op = "./Results/blocks_exp_preprocessed_data_rs_1_gpflow.p"

exp_data = pickle.load(open(logfile_ip, "rb"))

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
N = exp_params['num_samples']

n_trials, n_time_steps, dim_state = Xs.shape
_, _, dim_action = Us.shape
assert(dX==dim_state)
assert(n_time_steps==T)
assert(dU==dim_action)

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
Xs_t_train = XUs_train[:,:-1,:dX]
exp_data['Xs_t_train'] = Xs_t_train
Xs_t1_train = XUs_train[:,1:,:dX]
exp_data['Xs_t1_train'] = Xs_t1_train

XUs_test = XUs[n_train:, :, :]
XUs_t_test = XUs_test[:,:-1,:]
exp_data['XUs_t_test'] = XUs_t_test
Xs_t1_test = XUs_test[:,1:,:dX]
exp_data['Xs_t1_test'] = Xs_t1_test
Xs_t_test = XUs_test[:,:-1,:dX]
exp_data['Xs_t_test'] = Xs_t_test

X0s = XUs_train[:, 0, :dX]
X0_mu = np.mean(X0s, axis=0)
exp_data['X0_mu'] = X0_mu
X0_var = np.var(X0s, axis=0)
exp_data['X0_var'] = X0_var

pickle.dump(exp_data, open(logfile_op, "wb"))