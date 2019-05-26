import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
from model_leraning_utils import traj_with_globalgp
from YumiKinematics import YumiKinematics
from mjc_exp_policy import kin_params

blocks_logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_mm.p"
yumi_logfile = "./Results/yumi_peg_exp_new_preprocessed_data_train_4.p"

blocks_moe_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_moe.p"
blocks_bnn_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_bnn.p"
blocks_mgp_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_mgp.p"
blocks_gp_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_gp.p"
yumi_moe_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_moe.p"
yumi_bnn_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn.p"
yumi_mgp_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_mgp.p"
yumi_gp_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_gp.p"

blocks_exp_data = pickle.load( open(blocks_logfile, "rb" ))
yumi_exp_data = pickle.load( open(yumi_logfile, "rb" ))
blocks_moe_results = pickle.load( open(blocks_moe_result_file, "rb" ))
blocks_bnn_results = pickle.load( open(blocks_bnn_result_file, "rb" ))
blocks_mgp_results = pickle.load( open(blocks_mgp_result_file, "rb" ))
blocks_gp_results = pickle.load( open(blocks_gp_result_file, "rb" ))

yumi_moe_results = pickle.load( open(yumi_moe_result_file, "rb" ))
yumi_bnn_results = pickle.load( open(yumi_bnn_result_file, "rb" ))
yumi_mgp_results = pickle.load( open(yumi_mgp_result_file, "rb" ))
yumi_gp_results = pickle.load( open(yumi_gp_result_file, "rb" ))

# delta_model = True
# traj_with_bnn_ = traj_with_globalgp(None, None, None, None, dlt_mdl=delta_model)
# traj_with_mgp_ = traj_with_globalgp(None, None, None, None, dlt_mdl=delta_model)
#
# Xs_t_test = blocks_exp_data['Xs_t_test']
# traj_with_bnn = blocks_bnn_results['density_est']
# traj_with_bnn_.params = traj_with_bnn.params
# traj_with_bnn_.sample_trajs = traj_with_bnn.sample_trajs
# traj_with_bnn_.traj_density = traj_with_bnn.traj_density
# traj_with_bnn_.plot_gmm_traj(Xs_t_test)
#
# traj_with_mgp = blocks_mgp_results['density_est']
# traj_with_mgp_.params = traj_with_mgp.params
# traj_with_mgp_.sample_trajs = traj_with_mgp.sample_trajs
# traj_with_mgp_.traj_density = traj_with_mgp.traj_density
# traj_with_mgp_.plot_gmm_traj(Xs_t_test)

# yumi_exp_params = yumi_exp_data['exp_params']
# dP = yumi_exp_params['dP']
# dV = yumi_exp_params['dV']
# dU = yumi_exp_params['dU']
# dX = dP+dV
# dEP = 6
# dEV = 6
# dEX = 12
# dF = 6
# T = yumi_exp_params['T'] - 1
# dt = yumi_exp_params['dt']
# H=T
#
#
# # plot only mode of multimodal dist
# sim_data_tree = yumi_moe_results['track_data']
# tm = np.array(range(H))
# P_mu = np.zeros((H, dP))
# V_mu = np.zeros((H, dV))
# Xs_mu_prop = []
# for t in range(H):
#     tracks = sim_data_tree[t]
#     xp_pairs = [[track[2], track[6]] for track in tracks]
#     xs = [track[2] for track in tracks]
#     Xs_mu_prop.append(xs)
#     xp_max = max(xp_pairs, key=lambda x: x[1])
#     P_mu[t] = xp_max[0][:dP]
#     V_mu[t] = xp_max[0][dP:dP+dV]
#
# # plot the predicted trajectory in cartesian space
# yumiKin = YumiKinematics(kin_params)
# XU_t_train_avg = yumi_exp_data['XU_t_train_avg']
# X_mu_gp = np.mean(yumi_gp_results['traj_samples'], axis=0)
# X_mu_bnn = np.mean(yumi_bnn_results['traj_samples'], axis=0)
# X_mu_mgp = np.mean(yumi_mgp_results['traj_samples'], axis=0)
# ep_prop = np.zeros((H, 3))
# ep_train = np.zeros((H, 3))
# ep_gp = np.zeros((H, 3))
# ep_bnn = np.zeros((H, 3))
# ep_mgp = np.zeros((H, 3))
# P_mu_gp = X_mu_gp[:, :dP]
# P_mu_bnn = X_mu_bnn[:, :dP]
# P_mu_mgp = X_mu_mgp[:, :dP]
# for t in range(H):
#     q_prop = P_mu[t]
#     x_prop = yumiKin.fwd_pose(q_prop)
#     ep_prop[t] = x_prop[:3]
#
#     q_train = XU_t_train_avg[t, :dP]
#     x_train = yumiKin.fwd_pose(q_train)
#     ep_train[t] = x_train[:3]
#
#     q_gp = P_mu_gp[t]
#     x_gp = yumiKin.fwd_pose(q_gp)
#     ep_gp[t] = x_gp[:3]
#
#     q_bnn = P_mu_bnn[t]
#     x_bnn = yumiKin.fwd_pose(q_bnn)
#     ep_bnn[t] = x_bnn[:3]
#
#     q_mgp = P_mu_mgp[t]
#     x_mgp = yumiKin.fwd_pose(q_mgp)
#     ep_mgp[t] = x_mgp[:3]
# fig = plt.figure()
# plt.rcParams.update({'font.size': 10})
# markersize = 2
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot3D(ep_train[:, 0], ep_train[:, 1], ep_train[:, 2], marker='s', markersize=markersize, label='mean train')
# ax.plot3D(ep_gp[:, 0], ep_gp[:, 1], ep_gp[:, 2], marker='s', markersize=markersize, label='GP')
# ax.plot3D(ep_prop[:,0], ep_prop[:,1], ep_prop[:,2], marker='s', markersize=markersize, label='proposed')
# ax.plot3D(ep_bnn[:,0], ep_bnn[:,1], ep_bnn[:,2], marker='s', markersize=markersize, label='BNN')
# ax.plot3D(ep_mgp[:,0], ep_mgp[:,1], ep_mgp[:,2], marker='s', markersize=markersize, label='mGP')
# # ax.scatter3D(ep_prop[:, 0], ep_prop[:, 1], ep_prop[:, 2], marker='s', c=col_mode)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # ax.set_title('Comparison of predicted mean trajectories')
# ax.legend()
# plt.savefig('contact_motion_comparison.pdf')
# plt.show(block=False)

blocks_exp_params = blocks_exp_data['exp_params']
dP = blocks_exp_params['dP']
dV = blocks_exp_params['dV']
dU = blocks_exp_params['dU']
dX = dP+dV
T = blocks_exp_params['T'] - 1
dt = blocks_exp_params['dt']
XUs_t_test = blocks_exp_data['XUs_t_test']
Xs_t_test = XUs_t_test[:, :, :dX]
H = T
# plot long term prediction results of UGP
path_dict = blocks_moe_results['path_data']
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
    # plt.fill_between(time, pos - pos_std * 1.96, pos + pos_std * 1.96, alpha=0.2, color=path['col'])
    plt.subplot(122)
    plt.scatter(time, vel, c=rbga_col, marker='s', label='M'+str(path_key[0])+' mean')
    # plt.fill_between(time, vel - vel_std * 1.96, vel + vel_std * 1.96, alpha=0.2, color=path['col'])



tm = np.array(range(H))*dt
# plot training data
x = Xs_t_test[0]
# x = Xs_t_train[0]
plt.subplot(121)
plt.plot(tm, x[:H, :dP], ls='--', color='k', alpha=0.2, label='Training data')
plt.subplot(122)
plt.plot(tm, x[:H, dP:dP + dV], ls='--', color='k', alpha=0.2, label='Training data')
for x in Xs_t_test[1:]:
# for x in Xs_t_train[1:]:
    plt.subplot(121)
    plt.plot(tm, x[:H, :dP], ls='--', color='k', alpha=0.2)
    # plt.legend()
    plt.subplot(122)
    plt.plot(tm, x[:H, dP:dP+dV], ls='--', color='k', alpha=0.2)
    # plt.legend()
plt.show()

None