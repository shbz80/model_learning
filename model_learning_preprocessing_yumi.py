import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle
from sklearn.preprocessing import StandardScaler
from model_leraning_utils import YumiKinematics
from pykdl_utils.kdl_kinematics import *
from model_leraning_utils import obtian_joint_space_policy
import copy

logfile_ip = "./Results/yumi_peg_exp_new_raw_data_train.p"
logfile_op = "./Results/yumi_peg_exp_new_preprocessed_data_train_1.p"

# f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')
f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_gps_generated.urdf', 'r')

euler_from_matrix = trans.euler_from_matrix
J_G_to_A = YumiKinematics.jacobian_geometric_to_analytic

#pykdl stuff
# robot = Robot.from_xml_string(f.read())
# base_link = robot.get_root()
# end_link = 'left_contact_point'
robot = Robot.from_xml_string(f.read())
base_link = 'yumi_base_link'
end_link = 'gripper_l_base'
kdl_kin = KDLKinematics(robot, base_link, end_link)

# Overwrite wrist data with no action
overwrite_j7_data = False
p_7 = -2.1547
v_7 = 0.0
u_7 = 0.0

reject = [0]

exp_data = pickle.load(open(logfile_ip, "rb"))
Xs = exp_data['X']  # state
Us = exp_data['U']
EXs = exp_data['EX']  # state
Fs = exp_data['F']

if overwrite_j7_data:
    Xs[:, :, 6:7] = p_7
    Xs[:, :, 13:14] = v_7
    Us[:, :, 6:7] = u_7
    exp_data['X'] = Xs
    exp_data['U'] = Us
    dP = 7
    dV = 7
    dU = 7
    N, T, dX = Xs.shape
    Qts = Xs[:, :, :dP]
    Qts_d = Xs[:, :, dP:dP+dV]
    Uts = Us
    Ets = np.zeros((N, T, 6))
    Ets_d = np.zeros((N, T, 6))
    Fts = np.zeros((N, T, 6))
    for n in range(N):
        for i in range(T):
            Tr = kdl_kin.forward(Qts[n,i], end_link=end_link, base_link=base_link)
            epos = np.array(Tr[:3, 3])
            epos = epos.reshape(-1)
            erot = np.array(Tr[:3, :3])
            tmp = euler_from_matrix(erot, 'sxyz')
            erot = copy.copy(tmp[::-1])
            Ets[n,i] = np.append(epos, erot)

            J_G = np.array(kdl_kin.jacobian(Qts[n,i]))
            J_G = J_G.reshape((6, 7))
            J_A = J_G_to_A(J_G, Ets[n,i][3:])
            Ets_d[n,i] = J_A.dot(Qts_d[n,i])

            Fts[n,i] = np.linalg.pinv(J_A.T).dot(Uts[n,i])

    EXs = np.concatenate((Ets,Ets_d),axis=2)
    Fs = Fts
    exp_data['EX'] = EXs
    exp_data['F'] = Fs

exp_params = exp_data['exp_params']
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
#     plt.plot(tm, Xs[:,:, j].T, color='g', alpha=0.3)
#     plt.plot(tm, np.mean(Xs[:, :, j], axis=0), color='g', linestyle='--')
#
# # jVel
# for j in range(7):
#     plt.subplot(3, 7, 8 + j)
#     plt.title('j%dVel' % (j + 1))
#     plt.plot(tm, Xs[:, :, dP+j].T, color='b', alpha=0.3)
#     plt.plot(tm, np.mean(Xs[:, :, dP + j], axis=0), color='b', linestyle='--')
#
# # jTrq
# for j in range(7):
#     plt.subplot(3, 7, 15 + j)
#     plt.title('j%dTrq' % (j + 1))
#     plt.plot(tm, Us[:, :, j].T, color='r', alpha=0.3)
#     plt.plot(tm, np.mean(Us[:, :, j], axis=0), color='r', linestyle='--')
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

n_trials = n_trials - len(reject)           # adjusted after filtering
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
            'kp': np.array([0.22, 0.22, 0.18, 0.15, 0.05, 0.05, 0.025])*100.0*0.5,
            'kd': np.array([0.07, 0.07, 0.06, 0.05, 0.015, 0.015, 0.01])*10.0*0.5,
            'dX': dX,
            'dP': dP,
            'dV': dV,
            'dU': dU,
            'dt': dt,
}
x_init = np.concatenate((np.array([-1.3033, -1.3531, 0.9471, 0.3177, 2.0745, 1.4900, -2.1547]),
                          np.zeros(7)))
Xrs_t_train = obtian_joint_space_policy(params, XUs_t_train, x_init)
exp_data['Xrs_t_train'] = Xrs_t_train

plt.figure()
tm = np.linspace(0, T * dt, T)
# jPos
for j in range(7):
    plt.subplot(2, 7, 1 + j)
    plt.title('j%dPos' % (j + 1))
    plt.plot(tm, Xs[:,:, j].T, color='g', alpha=0.3)
    plt.plot(tm, np.mean(Xs[:, :, j], axis=0), color='g', linestyle='--', label='p mean')
    plt.plot(tm[:-1], np.mean(Xrs_t_train[:, :, j], axis=0), color='g', linestyle='-.', label='pref mean')
plt.legend()

# jVel
for j in range(7):
    plt.subplot(2, 7, 8 + j)
    plt.title('j%dVel' % (j + 1))
    plt.plot(tm, Xs[:, :, dP+j].T, color='b', alpha=0.3)
    plt.plot(tm, np.mean(Xs[:, :, dP + j], axis=0), color='b', linestyle='--', label='v mean')
    plt.plot(tm[:-1], np.mean(Xrs_t_train[:, :, dP + j], axis=0), color='b', linestyle='-.', label='vref mean')
plt.legend()

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
