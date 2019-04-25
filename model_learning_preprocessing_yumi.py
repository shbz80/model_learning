import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle
from sklearn.preprocessing import StandardScaler
from YumiKinematics import YumiKinematics
from mjc_exp_policy import kin_params, exp_params_rob
# from pykdl_utils.kdl_kinematics import *
from model_leraning_utils import obtian_joint_space_policy, get_ee_points

# data from yumi exp
logfile_ip = "./Results/yumi_peg_exp_new_raw_data_train.p"
# logfile_ip = "./Results/yumi_peg_exp_new_raw_data_test_m2.p"

# logfile_op = "./Results/yumi_peg_exp_new_preprocessed_data_train_2.p"   # global gp trained and lt pred working with simple policy
# logfile_op = "./Results/yumi_peg_exp_new_preprocessed_data_train_3.p"   # with EX_ee points
logfile_op = "./Results/yumi_peg_exp_new_preprocessed_data_train_4.p"   # noise not fixed
# logfile_op = "./Results/yumi_peg_exp_new_preprocessed_data_test_m2.p"

test_flag = False

# data from mjc exp
# logfile_ip = "./Results/mjc_exp_2_sec_raw.p"
# logfile_op = "./Results/mjc_exp_2_sec_raw_preprocessed.p"

yumiKin = YumiKinematics(kin_params)

# Overwrite wrist data with no action
overwrite_j7_data = False
p_7 = -2.1547
v_7 = 0.0
u_7 = 0.0

reject = [0]

exp_data = pickle.load(open(logfile_ip, "rb"))
Xs = exp_data['X']  # state
Us = exp_data['U']
# EXs = exp_data['EX']  # state
# Fs = exp_data['F']
exp_params = exp_data['exp_params']
if overwrite_j7_data:
    Xs[:, :, 6:7] = p_7
    Xs[:, :, 13:14] = v_7
    Us[:, :, 6:7] = u_7
    exp_data['X'] = Xs
    exp_data['U'] = Us

dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']

N, T, dX = Xs.shape
Qts = Xs[:, :, :dP]
Qts_d = Xs[:, :, dP:dP+dV]
Uts = Us
Ets = np.zeros((N, T, 6))
Ets_d = np.zeros((N, T, 6))
Ets_ee = np.zeros((N, T, 9))
Ets_d_ee = np.zeros((N, T, 9))
Fts = np.zeros((N, T, 6))
for n in range(N):
    for i in range(T):
        Ets[n, i] = yumiKin.fwd_pose(Qts[n,i])
        J_A = yumiKin.get_analytical_jacobian(Qts[n,i])
        Ets_d[n,i] = J_A.dot(Qts_d[n,i])
        Fts[n,i] = np.linalg.pinv(J_A.T).dot(Uts[n,i])

        ee_offsets = kin_params['ee_offsets']
        epos, erot = yumiKin.get_fwd_mat(Qts[n, i])
        ee_points = get_ee_points(ee_offsets, epos.reshape(1, -1), erot)
        ee_points = ee_points.T
        Ets_ee[n, i] = ee_points.reshape(-1)
        J_ee_1 = yumiKin.kdl_kin.jacobian(Qts[n, i], ee_points[0])
        J_ee_2 = yumiKin.kdl_kin.jacobian(Qts[n, i], ee_points[1])
        J_ee_3 = yumiKin.kdl_kin.jacobian(Qts[n, i], ee_points[2])
        J_ee_1 = J_ee_1[:3, :]
        J_ee_2 = J_ee_2[:3, :]
        J_ee_3 = J_ee_3[:3, :]
        ee_vel_1 = J_ee_1.dot(Qts_d[n, i])
        ee_vel_2 = J_ee_2.dot(Qts_d[n, i])
        ee_vel_3 = J_ee_3.dot(Qts_d[n, i])
        Ets_d_ee[n, i] = np.concatenate((ee_vel_1, ee_vel_2, ee_vel_3), axis=1)



EXs = np.concatenate((Ets,Ets_d),axis=2)
Fs = Fts
EXs_ee = np.concatenate((Ets_ee,Ets_d_ee),axis=2)
# exp_data['EX'] = EXs
# exp_data['F'] = Fs


# Xs = exp_data['X']  # state
# Us = exp_data['U']  # action
# Xg = exp_data['Xg']  # sate ground truth
# Ug = exp_data['Ug']  # action ground truth


# T = exp_params['T']
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

plt.figure()
tm = np.linspace(0, T * dt, T)
# jPos
for j in range(7):
    plt.subplot(3, 7, 1 + j)
    plt.title('j%dPos' % (j + 1))
    plt.plot(tm, Xs[:,:, j].T, color='g', alpha=0.3)
    plt.plot(tm, np.mean(Xs[:, :, j], axis=0), color='g', linestyle='--')

# jVel
for j in range(7):
    plt.subplot(3, 7, 8 + j)
    plt.title('j%dVel' % (j + 1))
    plt.plot(tm, Xs[:, :, dP+j].T, color='b', alpha=0.3)
    plt.plot(tm, np.mean(Xs[:, :, dP + j], axis=0), color='b', linestyle='--')

# jTrq
for j in range(7):
    plt.subplot(3, 7, 15 + j)
    plt.title('j%dTrq' % (j + 1))
    plt.plot(tm, Us[:, :, j].T, color='r', alpha=0.3)
    plt.plot(tm, np.mean(Us[:, :, j], axis=0), color='r', linestyle='--')

# plt.show()


id = [True]*N
for i in range(N):
    if i in reject:
        id[i] = False
EXs_fil = EXs[id]
EXs_ee_fil = EXs_ee[id]
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
        # ax.scatter3D(EXs[i,:,0], EXs[i,:,1], EXs[i,:,2], marker='$'+str(i)+'$', s=20)
        # ax.scatter3D(EXs[i, :, 0], EXs[i, :, 1], EXs[i, :, 2], color='r')
        ax.scatter3D(EXs_ee[i, :, 0], EXs_ee[i, :, 1], EXs_ee[i, :, 2], color='b')
        ax.scatter3D(EXs_ee[i, :, 3], EXs_ee[i, :, 4], EXs_ee[i, :, 5], color='g')
        ax.scatter3D(EXs_ee[i, :, 6], EXs_ee[i, :, 7], EXs_ee[i, :, 8], color='m')
        #plt.show()
# plt.show()

n_trials = n_trials - len(reject)           # adjusted after filtering
if test_flag:
    n_train = 0
else:
    n_train = n_trials//3 * 2
n_test = n_trials - n_train
exp_data['n_train'] = n_train
exp_data['n_test'] = n_test
Xs = Xs_fil
Us = Us_fil
EXs = EXs_fil
EXs_ee = EXs_ee_fil
Fs = Fs_fil
XUs = np.concatenate((Xs, Us), axis=2)
XUs_train = XUs[:n_train, :, :]
exp_data['XUs_train'] = XUs_train
XUs_t_train = XUs_train[:,:-1,:]
exp_data['XUs_t_train'] = XUs_t_train
XU_t_train_avg = np.mean(XUs_t_train, axis=0)
exp_data['XU_t_train_avg'] = XU_t_train_avg
Xs_t_train = XUs_train[:,:-1,:dX]
exp_data['Xs_t_train'] = Xs_t_train
Xs_t1_train = XUs_train[:,1:,:dX]
exp_data['Xs_t1_train'] = Xs_t1_train
Xs_t1_train_avg = np.mean(Xs_t1_train, axis=0)
exp_data['Xs_t1_train_avg'] = Xs_t1_train_avg
Us_t_train = XUs_train[:,:-1,dX:dX+dU]
exp_data['Us_t_train'] = Us_t_train

params = {
            'kp': exp_params_rob['Kp'],
            'kd': exp_params_rob['Kd'],
            'dX': dX,
            'dP': dP,
            'dV': dV,
            'dU': dU,
            'dt': dt,
}
x_init = exp_params_rob['x0']
Xrs_t_train = obtian_joint_space_policy(params, XUs_t_train, x_init)
exp_data['Xrs_t_train'] = Xrs_t_train

# plt.figure()
# tm = np.linspace(0, T * dt, T)
# # jPos
# for j in range(7):
#     plt.subplot(2, 7, 1 + j)
#     plt.title('j%dPos' % (j + 1))
#     plt.plot(tm, Xs[:,:, j].T, color='g', alpha=0.3)
#     plt.plot(tm, np.mean(Xs[:, :, j], axis=0), color='g', linestyle='--', label='p mean')
#     # plt.plot(tm[:-1], np.mean(Xrs_t_train[:, :, j], axis=0), color='g', linestyle='-.', label='pref mean')
# plt.legend()
#
# # jVel
# for j in range(7):
#     plt.subplot(2, 7, 8 + j)
#     plt.title('j%dVel' % (j + 1))
#     plt.plot(tm, Xs[:, :, dP+j].T, color='b', alpha=0.3)
#     plt.plot(tm, np.mean(Xs[:, :, dP + j], axis=0), color='b', linestyle='--', label='v mean')
#     # plt.plot(tm[:-1], np.mean(Xrs_t_train[:, :, dP + j], axis=0), color='b', linestyle='-.', label='vref mean')
# plt.legend()

plt.show()

EXFs = np.concatenate((EXs, Fs), axis=2)
EXFs_train = EXFs[:n_train, :, :]
exp_data['EXFs_train'] = EXFs_train
EXFs_t_train = EXFs_train[:,:-1,:]
exp_data['EXFs_t_train'] = EXFs_t_train
EXs_t_train = EXFs_train[:,:-1,:12]
exp_data['EXs_t_train'] = EXs_t_train
EXs_t1_train = EXFs_train[:,1:,:12]
exp_data['EXs_t1_train'] = EXs_t1_train
Fs_t_train = EXFs_train[:,:-1,12:]
exp_data['Fs_t_train'] = Fs_t_train
EXs_ee_t_train = EXs_ee[:n_train, :-1, :]
exp_data['EXs_ee_t_train'] = EXs_ee_t_train
EXs_ee_t1_train = EXs_ee[:n_train, 1:, :]
exp_data['EXs_ee_t1_train'] = EXs_ee_t1_train

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
