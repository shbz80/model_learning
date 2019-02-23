import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle
from sklearn.preprocessing import StandardScaler

# logfile_ip = "./Results/yumi_exp_raw_data_2.p"
# logfile_op = "./Results/yumi_exp_preprocessed_data_2.p"
logfile_ip_25 = "./Results/yumi_peg_exp_raw_data_25.p"
logfile_ip_26 = "./Results/yumi_peg_exp_raw_data_26.p"
logfile_ip_27 = "./Results/yumi_peg_exp_raw_data_27.p"
logfile_ip_28 = "./Results/yumi_peg_exp_raw_data_28.p"
logfile_ip_29 = "./Results/yumi_peg_exp_raw_data_29.p"
logfile_ip_30 = "./Results/yumi_peg_exp_raw_data_30.p"
logfile_ip_31 = "./Results/yumi_peg_exp_raw_data_31.p"
logfile_ip_32 = "./Results/yumi_peg_exp_raw_data_32.p"
logfile_ip_33 = "./Results/yumi_peg_exp_raw_data_33.p"
logfile_ip_34 = "./Results/yumi_peg_exp_raw_data_34.p"
logfile_op = "./Results/yumi_peg_exp_preprocessed_data_1.p"

# reject_1 = [12, 36, 8, 28, 16, 0, 20, 4, 35, 2, 25, 10, 23, 17, 13, 30, 38, 6, 24, 7, 15, 14, 1, 27, 39]
# reject_3 = [12, 36, 28, 37, 36, 8, 24, 0, 16, 14, 17, 20, 2, 35, 13, 38, 7, 23, 10, 4, 9, 30, 25, 26, 39]
# reject_2 = [12, 38, 36, 8, 28, 0, 16, 26, 10, 37, 35, 7, 13, 17, 27, 4, 24, 15, 39]
reject = []


exp_data_25 = pickle.load(open(logfile_ip_25, "rb"))
exp_data_26 = pickle.load(open(logfile_ip_26, "rb"))
exp_data_27 = pickle.load(open(logfile_ip_27, "rb"))
exp_data_28 = pickle.load(open(logfile_ip_28, "rb"))
exp_data_29 = pickle.load(open(logfile_ip_29, "rb"))
exp_data_30 = pickle.load(open(logfile_ip_30, "rb"))
exp_data_31 = pickle.load(open(logfile_ip_31, "rb"))
exp_data_32 = pickle.load(open(logfile_ip_32, "rb"))
exp_data_33 = pickle.load(open(logfile_ip_33, "rb"))
exp_data_34 = pickle.load(open(logfile_ip_34, "rb"))

exp_data_list = [exp_data_30, exp_data_31, exp_data_32, exp_data_33]
# exp_data_list = [exp_data_34]
Xs = exp_data_list[0]['X']  # state
Us = exp_data_list[0]['U']
EXs = exp_data_list[0]['EX']  # state
Fs = exp_data_list[0]['F']
for exp_data in exp_data_list[1:]:
    Xs_ = exp_data['X']  # state
    Us_ = exp_data['U']
    Xs = np.concatenate((Xs, Xs_), axis= 0)
    Us = np.concatenate((Us, Us_), axis=0)
    EXs_ = exp_data['EX']  # state
    Fs_ = exp_data['F']
    EXs = np.concatenate((EXs, EXs_), axis=0)
    Fs = np.concatenate((Fs, Fs_), axis=0)

exp_params = exp_data_list[0]['exp_params']
# Xs = exp_data['X']  # state
# Us = exp_data['U']  # action
# Xg = exp_data['Xg']  # sate ground truth
# Ug = exp_data['Ug']  # action ground truth

dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']
dX = dP+dV
T = exp_params['T']
dt = exp_params['dt']
# N = exp_params['num_samples']

n_trials, n_time_steps, dim_state = Xs.shape
N = n_trials
_, _, dim_action = Us.shape
assert(dX==dim_state)
assert(n_time_steps==T)
assert(dU==dim_action)

# plot yumi exp
# EXs = exp_data['EX']
# Fs = exp_data['F']

# plt.figure()
# tm = np.linspace(0, T * dt, T)
# # jPos
# for j in range(7):
#     plt.subplot(3, 7, 1 + j)
#     plt.title('j%dPos' % (j + 1))
#     plt.plot(tm, Xs[:,:, j].T, color='g')
#
# # jVel
# for j in range(7):
#     plt.subplot(3, 7, 8 + j)
#     plt.title('j%dVel' % (j + 1))
#     plt.plot(tm, Xs[:, :, dP+j].T, color='b')
#
# # jTrq
# for j in range(7):
#     plt.subplot(3, 7, 15 + j)
#     plt.title('j%dTrq' % (j + 1))
#     plt.plot(tm, Us[:, :, j].T, color='r')
#
# plt.show()


id = [True]*N
for i in range(N):
    if i in reject:
        id[i] = False
EXs_fil = EXs[id]
Xs_fil = Xs[id]
Us_fil = Us[id]
Fs_fil = Fs[id]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
for i in range(N):
    if i not in reject:
        ax.scatter3D(EXs[i,:,0], EXs[i,:,1], EXs[i,:,2], marker='$'+str(i)+'$', s=20)
        # plt.show()
plt.show()

# n_trials = 15           # adjusted after filtering
n_train = n_trials//3 * 2
n_test = n_trials - n_train
exp_data['n_train'] = n_train
exp_data['n_test'] = n_test
Xs = Xs_fil
Us = Us_fil
EXs = EXs_fil
Fs = Fs_fil
XUs = np.concatenate((Xs, Us), axis=2)
XUs_train = XUs[:n_train, :, :]
XUs_t_train = XUs_train[:,:-1,:]
exp_data['XUs_t_train'] = XUs_t_train
Xs_t_train = XUs_train[:,:-1,:dX]
exp_data['Xs_t_train'] = Xs_t_train
Xs_t1_train = XUs_train[:,1:,:dX]
exp_data['Xs_t1_train'] = Xs_t1_train
Us_t_train = XUs_train[:,:-1,dX:dX+dU]
exp_data['Us_t_train'] = Us_t_train

EXFs = np.concatenate((EXs, Fs), axis=2)
EXFs_train = EXFs[:n_train, :, :]
EXFs_t_train = EXFs_train[:,:-1,:]
exp_data['EXFs_t_train'] = EXFs_t_train
EXs_t_train = EXFs_train[:,:-1,:12]
exp_data['EXs_t_train'] = EXs_t_train
EXs_t1_train = EXFs_train[:,1:,:12]
exp_data['EXs_t1_train'] = EXs_t1_train
Fs_t_train = EXFs_train[:,:-1,12:]
exp_data['Fs_t_train'] = Fs_t_train

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

EXFs_test = EXFs[n_train:, :, :]
EXFs_t_test = EXFs_test[:,:-1,:]
exp_data['EXFs_t_test'] = EXFs_t_test
EXs_t1_test = EXFs_test[:,1:,:12]
exp_data['EXs_t1_test'] = EXs_t1_test
EXs_t_test = EXFs_test[:,:-1,:12]
exp_data['EXs_t_test'] = EXs_t_test

EX0s = EXFs_train[:, 0, :12]
EX0_mu = np.mean(EX0s, axis=0)
exp_data['EX0_mu'] = EX0_mu
EX0_var = np.var(EX0s, axis=0)
exp_data['EX0_var'] = EX0_var

pickle.dump(exp_data, open(logfile_op, "wb"))