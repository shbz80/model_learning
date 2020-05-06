import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import pickle
from YumiKinematics import YumiKinematics
from mjc_exp_policy import kin_params
from model_leraning_utils import UGP, plotEllipsiodError, get_N_HexCol
import operator
from collections import Counter

# yumi_logfile = "/home/shahbaz/Research/Software/model_learning/Results/yumi_peg_exp_new_preprocessed_data_train_big_data_wom10.p"
yumi_logfile = "/home/shahbaz/Research/Software/model_learning/Results/Final/yumi_peg_exp_new_preprocessed_data_train_big_data_wom10_fixed.p"
yumi_data = pickle.load( open(yumi_logfile, "rb" ))

# yumi_bnn_uc_bd_logfile = "./Results/results_yumi_bnn_bigdata.p"
yumi_bnn_uc_bd_logfile = "./Results/Final/results_yumi_bnn_d40.p"
# yumi_moe_bd_logfile = "./Results/results_yumi_moe_bigdata.p"
yumi_moe_bd_logfile = "./Results/Final/results_yumi_moe_d40.p"
yumi_moe_wo_init_bd_logfile = "./Results/Final/results_wo_init_yumi_moe_d40.p"
yumi_moe_bd_basic_me_logfile = "./Results/Final/results_yumi_moe_d40_basic_me.p"
yumi_bnn_uc_bd_results = pickle.load( open(yumi_bnn_uc_bd_logfile, "rb" ))
yumi_moe_bd_results = pickle.load( open(yumi_moe_bd_logfile, "rb" ))
yumi_moe_wo_init_bd_results = pickle.load( open(yumi_moe_wo_init_bd_logfile, "rb" ))
yumi_moe_bd_basic_me_results = pickle.load( open(yumi_moe_bd_basic_me_logfile, "rb" ))

# yumi_gp_sd_logfile = "./Results/results_yumi_gp_smalldata.p"
yumi_gp_sd_logfile = "./Results/Final/results_yumi_gp_d15.p"
# yumi_mgp_uc_sd_logfile = "./Results/results_yumi_mgp_smalldata.p"
# yumi_mgp_sd_logfile = "./Results/results_yumi_mgp_smalldata_dx_limit.p"
yumi_mgp_sd_logfile = "./Results/Final/results_yumi_mgp_d15.p"
# yumi_mgp_sd_logfile_1 = "./Results/Final/results_yumi_mgp_d15_2.p"
# yumi_bnn_sd_logfile = "./Results/results_yumi_bnn_smalldata.p"
yumi_bnn_sd_logfile = "./Results/Final/results_yumi_bnn_d15.p"
# yumi_bnn_uc_sd_logfile = "./Results/results_yumi_bnn_smalldata_unstable.p"
# yumi_moe_sd_logfile = "./Results/results_yumi_moe_smalldata.p"
yumi_moe_sd_logfile = "./Results/Final/results_yumi_moe_d15.p"
yumi_gp_sd_results = pickle.load( open(yumi_gp_sd_logfile, "rb" ))
yumi_mgp_sd_results = pickle.load( open(yumi_mgp_sd_logfile, "rb" ))
# yumi_mgp_sd_results_1 = pickle.load( open(yumi_mgp_sd_logfile_1, "rb" ))
# yumi_mgp_uc_sd_results = pickle.load( open(yumi_mgp_uc_sd_logfile, "rb" ))
yumi_bnn_sd_results = pickle.load( open(yumi_bnn_sd_logfile, "rb" ))
# yumi_bnn_uc_sd_results = pickle.load( open(yumi_bnn_uc_sd_logfile, "rb" ))
yumi_moe_sd_results = pickle.load( open(yumi_moe_sd_logfile, "rb" ))

yumi_sd_params = yumi_data['exp_params']
dP = yumi_sd_params['dP']
dV = yumi_sd_params['dV']
dU = yumi_sd_params['dU']
dX = dP+dV
T = yumi_sd_params['T'] - 1
dt = yumi_sd_params['dt']
XUs_t_test = yumi_data['XUs_t_test']
EXs_t_test = yumi_data['EXs_t_test']
Xs_t_test = XUs_t_test[:, :, :dX]
n_test, H, _ = Xs_t_test.shape

traj_samples_gp = yumi_gp_sd_results['traj_samples']
traj_samples_mgp = yumi_mgp_sd_results['traj_samples']
# pred_means_mgp = yumi_mgp_sd_results_1['pred_mean']
traj_samples_bnn = yumi_bnn_sd_results['traj_samples']
# traj_samples_mgp_uc = yumi_mgp_uc_sd_results['traj_samples']
# traj_samples_bnn_uc = yumi_bnn_uc_sd_results['traj_samples']
traj_samples_bnn_uc_bd = yumi_bnn_uc_bd_results['traj_samples']
track_data_moe_sd = yumi_moe_sd_results['track_data']
track_data_moe_bd = yumi_moe_bd_results['track_data']
track_data_moe_wo_init_bd = yumi_moe_wo_init_bd_results['track_data']
del yumi_moe_wo_init_bd_results
path_dict_bd = yumi_moe_bd_results['path_data']
P_mu_gp = np.mean(traj_samples_gp, axis=0)[:, :dP]
V_mu_gp = np.mean(traj_samples_gp, axis=0)[:, dP:dP+dV]
P_mu_mgp = np.mean(traj_samples_mgp, axis=0)[:, :dP]
P_mu_bnn = np.mean(traj_samples_bnn, axis=0)[:, :dP]
# P_mu_mgp_uc = np.mean(traj_samples_mgp_uc, axis=0)[:, :dP]
# P_mu_bnn_uc = np.mean(traj_samples_bnn_uc, axis=0)[:, :dP]
P_mu_bnn_uc_bd = np.mean(traj_samples_bnn_uc_bd, axis=0)[:, :dP]
P_mu_moe = np.zeros((H, dP))
# V_mu_moe = np.zeros((H, dV))
P_mu_moe_bd = np.zeros((H, dP))
P_var_moe = np.zeros((H, dP, dP))
P_var_moe_bd = np.zeros((H, dP, dP))
V_mu_moe_bd = np.zeros((H, dP))
V_mu_moe_wo_init_bd = np.zeros((H, dP))
V_var_moe_bd = np.zeros((H, dV, dV))
V_var_moe_wo_init_bd = np.zeros((H, dV, dV))
L_moe = np.zeros(H, dtype=int)
L_moe_bd = np.zeros(H, dtype=int)
L_moe_wo_init_bd = np.zeros(H, dtype=int)
for t in range(H):
    tracks = track_data_moe_sd[t]
    xp_pairs = [[track[0], track[2], track[3], track[6]] for track in tracks]
    xp_max = max(xp_pairs, key=lambda x: x[3])
    P_mu_moe[t] = xp_max[1][:dP]
    P_var_moe[t] = xp_max[2][:dP, :dP]
    L_moe[t] = xp_max[0]

    tracks = track_data_moe_bd[t]
    xp_pairs = [[track[0], track[2], track[3], track[6]] for track in tracks]
    xp_max = max(xp_pairs, key=lambda x: x[3])
    P_mu_moe_bd[t] = xp_max[1][:dP]
    P_var_moe_bd[t] = xp_max[2][:dP, :dP]
    V_mu_moe_bd[t] = xp_max[1][dP:dP+dV]
    V_var_moe_bd[t] = xp_max[2][dP:dP+dV, dP:dP+dV]
    L_moe_bd[t] = xp_max[0]

    tracks = track_data_moe_wo_init_bd[t]
    xp_pairs = [[track[0], track[2], track[3], track[6]] for track in tracks]
    xp_max = max(xp_pairs, key=lambda x: x[3])
    V_mu_moe_wo_init_bd[t] = xp_max[1][dP:dP + dV]
    V_var_moe_wo_init_bd[t] = xp_max[2][dP:dP + dV, dP:dP + dV]
    L_moe_wo_init_bd[t] = xp_max[0]

tm =  np.array(range(T))*dt

# plot the predicted trajectory in cartesian space
yumiKin = YumiKinematics(kin_params)
XU_t_train_avg = yumi_data['XU_t_train_avg']
del yumi_data
XU_t_test_avg = np.mean(XUs_t_test, axis=0)
ep_moe = np.zeros((H, 3))
ev_moe = np.zeros((H, 3))
ep_moe_bd = np.zeros((H, 3))
ep_train = np.zeros((H, 3))
ep_test = np.zeros((H, 3))
ev_test = np.zeros((H, 3))
ep_gp = np.zeros((H, 3))
ev_gp = np.zeros((H, 3))
ep_bnn = np.zeros((H, 3))
# ep_bnn_uc = np.zeros((H, 3))
ep_bnn_uc_bd = np.zeros((H, 3))
ep_mgp = np.zeros((H, 3))
# ep_mgp_uc = np.zeros((H, 3))

for t in range(H):
    q_moe = P_mu_moe[t]
    x_moe = yumiKin.fwd_pose(q_moe)
    ep_moe[t] = x_moe[:3]

    q_moe_bd = P_mu_moe_bd[t]
    x_moe_bd = yumiKin.fwd_pose(q_moe_bd)
    ep_moe_bd[t] = x_moe_bd[:3]

    q_train = XU_t_train_avg[t, :dP]
    x_train = yumiKin.fwd_pose(q_train)
    ep_train[t] = x_train[:3]

    q_test = XU_t_test_avg[t, :dP]
    x_test = yumiKin.fwd_pose(q_test)
    ep_test[t] = x_test[:3]

    q_gp = P_mu_gp[t]
    x_gp = yumiKin.fwd_pose(q_gp)
    ep_gp[t] = x_gp[:3]

    q_bnn = P_mu_bnn[t]
    x_bnn = yumiKin.fwd_pose(q_bnn)
    ep_bnn[t] = x_bnn[:3]

    # q_bnn_uc = P_mu_bnn_uc[t]
    # x_bnn_uc = yumiKin.fwd_pose(q_bnn_uc)
    # ep_bnn_uc[t] = x_bnn_uc[:3]

    q_bnn_uc_bd = P_mu_bnn_uc_bd[t]
    x_bnn_uc_bd = yumiKin.fwd_pose(q_bnn_uc_bd)
    ep_bnn_uc_bd[t] = x_bnn_uc_bd[:3]
    #
    q_mgp = P_mu_mgp[t]
    x_mgp = yumiKin.fwd_pose(q_mgp)
    ep_mgp[t] = x_mgp[:3]

    # q_mgp_uc = P_mu_mgp_uc[t]
    # x_mgp_uc = yumiKin.fwd_pose(q_mgp_uc)
    # ep_mgp_uc[t] = x_mgp_uc[:3]
# i=0
# for pred_mean in pred_means_mgp:
#     ep = np.zeros((H, 3))
#     for t in range(H):
#         q = pred_mean[t][:dP]
#         x = yumiKin.fwd_pose(q)
#         ep[t] = x[:3]
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     ax.plot3D(ep[:, 0], ep[:, 1], ep[:, 2], marker='s', markersize=3, label=str(i))
#     ax.plot3D(ep_test[:, 0], ep_test[:, 1], ep_test[:, 2], marker='s', markersize=3,
#               label='$\mathbb{D}^{test}$')
#     plt.show(block=False)
#     plt.legend()
#     i += 1

ugp_params = {
    'alpha': 1.,
    'kappa': 2.,
    'beta': 0.,
}
ugp_fk = UGP(dP, **ugp_params)

e_mu_moe_bd = np.zeros((H, 3))
e_var_moe_bd = np.zeros((H, 3, 3))
for t in range(H):
    q_mu = P_mu_moe_bd[t]
    q_var = P_var_moe_bd[t]
    e_mu, e_var, _, _, _ = ugp_fk.get_posterior(yumiKin, q_mu, q_var)
    e_mu_moe_bd[t] = e_mu[:3]
    e_var_moe_bd[t] = e_var[:3, :3]

fig = plt.figure()
plt.rcParams.update({'font.size': 22})
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'
markersize = 7
markevery =4
cl_gp = 'g'
cl_bnn = 'c'
cl_moe = 'm'
cl_mgp = 'b'

ax = fig.add_subplot(1, 1, 1, projection='3d')
# fig.tight_layout()
fig.subplots_adjust(left=0.00, bottom=0.00, right=1., top=1.)
EX_t_test = EXs_t_test[0]
ax.plot3D(EX_t_test[:, 0], EX_t_test[:, 1], EX_t_test[:, 2], color='k', linewidth=3, linestyle='--',alpha=0.4, label='$\mathbb{D}_{15}^{test}$')
for EX_t_test in EXs_t_test:
    ax.plot3D(EX_t_test[:, 0], EX_t_test[:, 1], EX_t_test[:, 2], color='k', linewidth=3, linestyle='--', alpha=0.4)
# ax.plot3D(ep_train[:, 0], ep_train[:, 1], ep_train[:, 2], marker='s', markersize=markersize, label='$\mathbb{D}^{train}$')
# ax.plot3D(ep_test[:, 0], ep_test[:, 1], ep_test[:, 2], marker='s', markersize=markersize, label='$\mathbb{D}^{test}$')
ax.plot3D(ep_moe[:,0], ep_moe[:,1], ep_moe[:,2], marker='s', markersize=markersize, markevery=markevery, color=cl_moe, label='Ours')
ax.plot3D(ep_mgp[:,0], ep_mgp[:,1], ep_mgp[:,2], marker='s', markersize=markersize, markevery=markevery, color=cl_mgp, label='mGP')
# ax.plot3D(ep_mgp_uc[:,0], ep_mgp_uc[:,1], ep_mgp_uc[:,2], marker='s', markersize=markersize, label='mGP(uc)')
ax.plot3D(ep_gp[:, 0], ep_gp[:, 1], ep_gp[:, 2], marker='s', markersize=markersize, markevery=markevery, color=cl_gp, label='GP')
ax.plot3D(ep_bnn[:,0], ep_bnn[:,1], ep_bnn[:,2], marker='s', markersize=markersize, markevery=markevery, color=cl_bnn, label='uANN')
# ax.plot3D(ep_bnn_uc[:,0], ep_bnn_uc[:,1], ep_bnn_uc[:,2], marker='s', markersize=markersize, color=cl_bnn, label='uANN')
# ax.plot3D(ep_bnn_uc_bd[:,0], ep_bnn_uc_bd[:,1], ep_bnn_uc_bd[:,2], marker='s', markersize=markersize, label='uANN-40')

# ax.plot3D(ep_moe_bd[:,0], ep_moe_bd[:,1], ep_moe_bd[:,2], marker='s', markersize=markersize, label='Our method-40')
# ax.scatter3D(ep_prop[:, 0], ep_prop[:, 1], ep_prop[:, 2], marker='s', c=col_mode)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_title('Comparison of predicted mean trajectories')
ax.legend(loc='upper left', frameon=False)
# ax.legend()
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
# plt.savefig('contact_motion_comparison.pdf')
plt.show(block=False)

# plot yumi lt pred for big data
labels, counts = zip(*sorted(Counter(L_moe_bd).items(), key=operator.itemgetter(0)))
K = len(labels)
# colors = np.zeros((K,4))
# colors = get_N_HexCol(K)
# colors = np.asarray(colors) / 255.
red = [230, 25, 75]
green = [60, 180, 75]
blue = [0, 130, 200]
cyan = [70, 240, 240]
magenta = [240, 50, 230]
teal = [0, 128, 128]
yellow = [255, 225, 25]
orange = [245, 130, 48]
# list_col = [red, green, blue, cyan, magenta, teal, yellow, orange]
list_col = [orange, cyan, blue, yellow, magenta, green, teal, red]
colors = np.array(list_col)/255.0

label_col_dict = dict(zip(labels, colors))
col = np.zeros((T, 3))
markersize =10
list_legend = []
for label in labels:
    col[(L_moe_bd==label)] = label_col_dict[label]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
# fig.tight_layout()
fig.subplots_adjust(left=0.00, bottom=0.00, right=1., top=1.)
EX_t_test = EXs_t_test[0]
list_legend.append(ax.plot3D(EX_t_test[:, 0], EX_t_test[:, 1], EX_t_test[:, 2], color='k', linewidth=3, linestyle='--',alpha=0.4, label='$\mathbb{D}_{40}^{test}$')[0])
for EX_t_test in EXs_t_test:
    ax.plot3D(EX_t_test[:, 0], EX_t_test[:, 1], EX_t_test[:, 2], color='k', linestyle='--', alpha=0.4, linewidth=3)
# ax.scatter3D(e_mu_moe[:, 0], e_mu_moe[:, 1], e_mu_moe[:, 2], marker='s', s=markersize, color=col)
# legends = ['Free', 'Free', 'Free', 'Wood', 'Wood', 'Wood', 'Wood & Steel', 'Wood & Steel']
legends = None
i = 0
for label in labels:
    e_mu_moe_i = e_mu_moe_bd[(L_moe_bd == label)]
    cl = label_col_dict[label]
    # cl = list_col[i]
    list_legend.append(ax.scatter3D(e_mu_moe_i[:, 0], e_mu_moe_i[:, 1], e_mu_moe_i[:, 2], marker='s', s=markersize, color=cl))
    i = i+1
plotEllipsiodError(e_mu_moe_bd, e_var_moe_bd, col, ax=ax, alpha=0.05)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
# ax.set_title('Comparison of predicted mean trajectories')
# ax.legend(list_legend, ['$\mathbb{D}_{40}^{test}$', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'Mode means'], loc='upper left', frameon=False, ncol=9)
ax.legend( loc='upper left', frameon=False)
# ax.legend()
plt.show(block=False)

plt.figure()
# plt.subplot(121)
plt.xlabel('$t\ (s)$')
plt.ylabel('$\dot{q}_4(t)\ (rad/s)$')
tm = np.array(range(T))*dt
i=0
for label in labels:
    v_mu_moe_i = V_mu_moe_bd[(L_moe_bd == label)]
    v_var_moe_i = V_var_moe_bd[(L_moe_bd == label)]
    t = tm[(L_moe_bd == label)]
    cl = label_col_dict[label]
    plt.plot(t, v_mu_moe_i[:,3], color=colors[i], linewidth=3)
    plt.fill_between(t, v_mu_moe_i[:,3] - np.sqrt(v_var_moe_i[:,3,3]) * 1.96, v_mu_moe_i[:,3] + np.sqrt(v_var_moe_i[:,3,3]) * 1.96,
                     alpha=0.2, color=colors[i])
    i = i + 1
plt.plot(tm, np.average(Xs_t_test[:, :, dP+3], axis=0), linewidth=3, alpha=0.4, color='k', linestyle='--', label='Avg. $\mathbb{D}^{test}_{40}$')
plt.legend( loc='upper left', frameon=False)
plt.show(block=False)

plt.figure()
# plt.subplot(122)
plt.xlabel('$t\ (s)$')
plt.ylabel('$\dot{q}_4(t)\ (rad/s)$')
tm = np.array(range(T))*dt
i=0
for label in labels:
    v_mu_moe_i = V_mu_moe_wo_init_bd[(L_moe_wo_init_bd == label)]
    v_var_moe_i = V_var_moe_wo_init_bd[(L_moe_wo_init_bd == label)]
    t = tm[(L_moe_wo_init_bd == label)]
    cl = label_col_dict[label]
    plt.plot(t, v_mu_moe_i[:,3], color=colors[i], linewidth=3, marker='s', markevery=5, markersize=10)
    plt.fill_between(t, v_mu_moe_i[:,3] - np.sqrt(v_var_moe_i[:,3,3]) * 1.96, v_mu_moe_i[:,3] + np.sqrt(v_var_moe_i[:,3,3]) * 1.96,
                     alpha=0.2, color=colors[i])
    i = i + 1
X_mu_pred = yumi_moe_bd_basic_me_results['x_mu']
X_var_pred = yumi_moe_bd_basic_me_results['x_var']
mode_seq = yumi_moe_bd_basic_me_results['mode_seq']
del yumi_moe_bd_basic_me_results
for label in labels:
    cl = label_col_dict[label]
    t = tm[(mode_seq == label)]
    X_mu = X_mu_pred[:, dP+3][(mode_seq == label)]
    X_var = X_var_pred[:, dP+3, dP+3][(mode_seq == label)]
    plt.plot(t, X_mu, color=cl, linewidth=3, marker='o', markevery=5, markersize=10)
    plt.fill_between(t, X_mu - np.sqrt(X_var) * 1.96, X_mu + np.sqrt(X_var) * 1.96,
                     alpha=0.2, color=cl)
plt.plot(tm, np.average(Xs_t_test[:, :, dP+3], axis=0), linewidth=3, alpha=0.4, color='k', linestyle='--', label='Avg. $\mathbb{D}^{test}_{40}$')
plt.legend( loc='upper left', frameon=False)
plt.show()


# # list_col = [orange, cyan, blue, yellow, magenta, green, teal, red]
# list_col = [yellow, blue, teal, green, cyan, red, magenta, orange]
# colors = np.array(list_col)/255.0
# plt.figure()
# i=0
# for path_key in path_dict_bd:
#     path = path_dict_bd[path_key]
#     time = np.array(path['time'])
#     qd_2 = np.array(path['X'])[:, dP+2]
#     qd_2_std = np.array(path['X_std'])[:, dP+2]
#     plt.xlabel('$t$ (s)')
#     plt.ylabel('$\dot{q}(t)_2$ (rad/s)')
#     plt.plot(time, qd_2, color=colors[i])
#     plt.fill_between(time, qd_2 - qd_2_std * 1.96, qd_2 + qd_2_std * 1.96,
#                      alpha=0.2, color=colors[i])
#     # for i in range(n_test):
#     #     x = Xs_t_test[i, :, dP+2]
#     #     plt.plot(tm, x, alpha=0.1, color='k', linestyle='--')
#     plt.plot(tm, np.average(Xs_t_test[:, :, dP+2], axis=0), alpha=0.05, color='k', linestyle='--')
#     i = i+1
# plt.show()

def aggregrate_statistics(nll_list, n):
    nll_list = np.array(nll_list)
    s0 = np.full((nll_list.shape[0]), n)
    mu = nll_list[:,0].reshape(-1)
    sigma = nll_list[:, 1].reshape(-1)
    sigma_sq = np.square(sigma)
    s1 = np.multiply(s0,mu)
    s2 = np.multiply(s0,(np.square(mu)+sigma_sq))
    S0 = np.sum(s0)
    S1 = np.sum(s1)
    S2 = np.sum(s2)
    Sigma_sq = np.divide(S2,S0) - np.divide(S1,S0)**2
    Sigma = np.sqrt(Sigma_sq)
    Mu = np.mean(mu)
    return Mu, Sigma

print('NLL-15:')
print('GP:', yumi_gp_sd_results['nll'])
mu, sig = aggregrate_statistics(yumi_mgp_sd_results['nll'], 500)
print('mGP:', (-mu, sig))
# mu, sig = aggregrate_statistics(yumi_mgp_uc_sd_results['nll'], 500)
# print('mGP(uc):', (-mu, sig))
mu, sig = aggregrate_statistics(yumi_bnn_sd_results['nll'], 500)
print('uANN:', (-mu, sig))
# mu, sig = aggregrate_statistics(yumi_bnn_uc_sd_results['nll'], 500)
# print('uANN:', (-mu, sig))
# print('MOE:', yumi_moe_sd_results['nll'])
mu, sig = aggregrate_statistics(yumi_moe_sd_results['nll'], 500)
print('MOE:', (-mu, sig))

print('RMSE-15:')
print('GP:', yumi_gp_sd_results['rmse'])
print('mGP:', np.mean(yumi_mgp_sd_results['rmse']))
del yumi_mgp_sd_results
# print('mGP(uc):', np.mean(yumi_mgp_uc_sd_results['rmse']))
print('uANN:', np.mean(yumi_bnn_sd_results['rmse']))
# print('uANN:', np.mean(yumi_bnn_uc_sd_results['rmse']))
print('MOE:', np.mean(yumi_moe_sd_results['rmse']))

print('Pred-time-15:')
print('GP:', yumi_gp_sd_results['gp_pred_time'])
print('uANN:', np.mean(yumi_bnn_sd_results['pred_time']))
print('MOE:', yumi_moe_sd_results['moe_pred_time'])

print('Train-time-15:')
print('GP:', yumi_gp_sd_results['gp_training_time'])
del yumi_gp_sd_results
print('uANN:', np.mean(yumi_bnn_sd_results['train_time']))
del yumi_bnn_sd_results
moe_train_time = yumi_moe_sd_results['expert_train_time'] + yumi_moe_sd_results['cluster_time']\
+ yumi_moe_sd_results['trans_gp_time'] + yumi_moe_sd_results['svm_train_time']
del yumi_moe_sd_results
print('MOE:', moe_train_time)

print('NLL-40:')
mu, sig = aggregrate_statistics(yumi_bnn_uc_bd_results['nll'], 500)
print('uANN:', (-mu, sig))
mu, sig = aggregrate_statistics(yumi_moe_bd_results['nll'], 500)
print('MOE:', (-mu, sig))

print('RMSE-40:')
print('uANN:', np.mean(yumi_bnn_uc_bd_results['rmse']))
print('uANN:', np.mean(yumi_moe_bd_results['rmse']))

print('Pred-time-40:')
print('uANN:', np.mean(yumi_bnn_uc_bd_results['pred_time']))
print('MOE:', yumi_moe_bd_results['moe_pred_time'])

print('Train-time-40:')
print('uANN:', np.mean(yumi_bnn_uc_bd_results['train_time']))
del yumi_bnn_uc_bd_results
# clustering_time = 264.99
clustering_time = yumi_moe_bd_results['cluster_time']
BNN_pred_time_dens = 3.083568811416626
expert_train_time = yumi_moe_bd_results['expert_train_time']
init_train_time = yumi_moe_bd_results['trans_gp_time']
mode_predic_train_time = yumi_moe_bd_results['svm_train_time']
del yumi_moe_bd_results
print('clust clustering_time', clustering_time)
print('expert_train_time', expert_train_time)
print('init_train_time', init_train_time)
print('mode_predic_train_time', mode_predic_train_time)

moe_train_time = expert_train_time + clustering_time\
+ init_train_time + mode_predic_train_time
print('MOE:', moe_train_time)
None
