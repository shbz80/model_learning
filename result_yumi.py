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

yumi_logfile = "/home/shahbaz/Research/Software/model_learning/Results/yumi_peg_exp_new_preprocessed_data_train_big_data_wom10.p"
yumi_data = pickle.load( open(yumi_logfile, "rb" ))

yumi_bnn_uc_bd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn_bigdata.p"
yumi_moe_bd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_moe_bigdata.p"
yumi_bnn_uc_bd_results = pickle.load( open(yumi_bnn_uc_bd_logfile, "rb" ))
yumi_moe_bd_results = pickle.load( open(yumi_moe_bd_logfile, "rb" ))

yumi_gp_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_gp_smalldata.p"
yumi_mgp_uc_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_mgp_smalldata.p"
yumi_mgp_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_mgp_smalldata_dx_limit.p"
yumi_bnn_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn_smalldata.p"
yumi_bnn_uc_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn_smalldata_unstable.p"
yumi_moe_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_moe_smalldata.p"
yumi_gp_sd_results = pickle.load( open(yumi_gp_sd_logfile, "rb" ))
yumi_mgp_sd_results = pickle.load( open(yumi_mgp_sd_logfile, "rb" ))
yumi_mgp_uc_sd_results = pickle.load( open(yumi_mgp_uc_sd_logfile, "rb" ))
yumi_bnn_sd_results = pickle.load( open(yumi_bnn_sd_logfile, "rb" ))
yumi_bnn_uc_sd_results = pickle.load( open(yumi_bnn_uc_sd_logfile, "rb" ))
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
traj_samples_bnn = yumi_bnn_sd_results['traj_samples']
traj_samples_mgp_uc = yumi_mgp_uc_sd_results['traj_samples']
traj_samples_bnn_uc = yumi_bnn_uc_sd_results['traj_samples']
traj_samples_bnn_uc_bd = yumi_bnn_uc_bd_results['traj_samples']
track_data_moe_sd = yumi_moe_sd_results['track_data']
track_data_moe_bd = yumi_moe_bd_results['track_data']
P_mu_gp = np.mean(traj_samples_gp, axis=0)[:, :dP]
P_mu_mgp = np.mean(traj_samples_mgp, axis=0)[:, :dP]
P_mu_bnn = np.mean(traj_samples_bnn, axis=0)[:, :dP]
P_mu_mgp_uc = np.mean(traj_samples_mgp_uc, axis=0)[:, :dP]
P_mu_bnn_uc = np.mean(traj_samples_bnn_uc, axis=0)[:, :dP]
P_mu_bnn_uc_bd = np.mean(traj_samples_bnn_uc_bd, axis=0)[:, :dP]
P_mu_moe = np.zeros((H, dP))
P_mu_moe_bd = np.zeros((H, dP))
P_var_moe = np.zeros((H, dP, dP))
L_moe = np.zeros(H, dtype=int)
for t in range(H):
    # tracks = track_data_moe_sd[t]
    tracks = track_data_moe_bd[t]
    xp_pairs = [[track[0], track[2], track[3], track[6]] for track in tracks]
    xp_max = max(xp_pairs, key=lambda x: x[3])
    P_mu_moe[t] = xp_max[1][:dP]
    P_var_moe[t] = xp_max[2][:dP, :dP]
    L_moe[t] = xp_max[0]

    tracks_bd = track_data_moe_bd[t]
    xp_pairs_bd = [[track_bd[2], track_bd[6]] for track_bd in tracks_bd]
    xp_max_bd = max(xp_pairs_bd, key=lambda x: x[1])
    P_mu_moe_bd[t] = xp_max_bd[0][:dP]

# plot the predicted trajectory in cartesian space
yumiKin = YumiKinematics(kin_params)
XU_t_train_avg = yumi_data['XU_t_train_avg']
ep_moe = np.zeros((H, 3))
ep_moe_bd = np.zeros((H, 3))
ep_train = np.zeros((H, 3))
ep_gp = np.zeros((H, 3))
ep_bnn = np.zeros((H, 3))
ep_bnn_uc = np.zeros((H, 3))
ep_bnn_uc_bd = np.zeros((H, 3))
ep_mgp = np.zeros((H, 3))
ep_mgp_uc = np.zeros((H, 3))

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

    q_gp = P_mu_gp[t]
    x_gp = yumiKin.fwd_pose(q_gp)
    ep_gp[t] = x_gp[:3]

    q_bnn = P_mu_bnn[t]
    x_bnn = yumiKin.fwd_pose(q_bnn)
    ep_bnn[t] = x_bnn[:3]

    q_bnn_uc = P_mu_bnn_uc[t]
    x_bnn_uc = yumiKin.fwd_pose(q_bnn_uc)
    ep_bnn_uc[t] = x_bnn_uc[:3]

    q_bnn_uc_bd = P_mu_bnn_uc_bd[t]
    x_bnn_uc_bd = yumiKin.fwd_pose(q_bnn_uc_bd)
    ep_bnn_uc_bd[t] = x_bnn_uc_bd[:3]
    #
    q_mgp = P_mu_mgp[t]
    x_mgp = yumiKin.fwd_pose(q_mgp)
    ep_mgp[t] = x_mgp[:3]

    q_mgp_uc = P_mu_mgp_uc[t]
    x_mgp_uc = yumiKin.fwd_pose(q_mgp_uc)
    ep_mgp_uc[t] = x_mgp_uc[:3]

ugp_params = {
    'alpha': 1.,
    'kappa': 2.,
    'beta': 0.,
}
ugp_fk = UGP(dP, **ugp_params)

e_mu_moe = np.zeros((H, 3))
e_var_moe = np.zeros((H, 3, 3))
for t in range(H):
    q_mu = P_mu_moe[t]
    q_var = P_var_moe[t]
    e_mu, e_var, _, _, _ = ugp_fk.get_posterior(yumiKin, q_mu, q_var)
    e_mu_moe[t] = e_mu[:3]
    e_var_moe[t] = e_var[:3, :3]

fig = plt.figure()
plt.rcParams.update({'font.size': 12})
markersize = 3
cl_gp = 'g'
cl_bnn = 'c'
cl_moe = 'm'
cl_mgp = 'b'

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot3D(ep_train[:, 0], ep_train[:, 1], ep_train[:, 2], marker='s', markersize=markersize, label='$\mathbb{D}^{test}$')
ax.plot3D(ep_gp[:, 0], ep_gp[:, 1], ep_gp[:, 2], marker='s', markersize=markersize, color=cl_gp, label='GP')
ax.plot3D(ep_mgp[:,0], ep_mgp[:,1], ep_mgp[:,2], marker='s', markersize=markersize, color=cl_mgp, label='mGP(c)')
# ax.plot3D(ep_mgp_uc[:,0], ep_mgp_uc[:,1], ep_mgp_uc[:,2], marker='s', markersize=markersize, color=cl_mgp, label='mGP')
ax.plot3D(ep_bnn[:,0], ep_bnn[:,1], ep_bnn[:,2], marker='s', markersize=markersize, color=cl_bnn, label='BNN(c)')
# ax.plot3D(ep_bnn_uc[:,0], ep_bnn_uc[:,1], ep_bnn_uc[:,2], marker='s', markersize=markersize, color=cl_bnn, label='BNN')
# ax.plot3D(ep_bnn_uc_bd[:,0], ep_bnn_uc_bd[:,1], ep_bnn_uc_bd[:,2], marker='s', markersize=markersize, label='BNN-40')
ax.plot3D(ep_moe[:,0], ep_moe[:,1], ep_moe[:,2], marker='s', markersize=markersize, color=cl_moe, label='Our method')
# ax.plot3D(ep_moe_bd[:,0], ep_moe_bd[:,1], ep_moe_bd[:,2], marker='s', markersize=markersize, label='Our method-40')
# ax.scatter3D(ep_prop[:, 0], ep_prop[:, 1], ep_prop[:, 2], marker='s', c=col_mode)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_title('Comparison of predicted mean trajectories')
ax.legend()
# plt.savefig('contact_motion_comparison.pdf')
plt.show(block=False)

labels, counts = zip(*sorted(Counter(L_moe).items(), key=operator.itemgetter(0)))
K = len(labels)
colors = np.zeros((K,4))
colors = get_N_HexCol(K)
colors = np.asarray(colors) / 255.

# colors_itr = iter(cm.rainbow(np.linspace(0, 1, K)))
# for i in range(K):
#     colors[i] = next(colors_itr)
# colors=colors[:,:3]


label_col_dict = dict(zip(labels, colors))
col = np.zeros((T, 3))
markersize =10

for label in labels:
    col[(L_moe==label)] = label_col_dict[label]
ax = fig.add_subplot(1, 1, 1, projection='3d')
EX_t_test = EXs_t_test[0]
ax.plot3D(EX_t_test[:, 0], EX_t_test[:, 1], EX_t_test[:, 2], color='k', linestyle='--',alpha=0.2, label='$\mathbb{D}_{40}^{test}$')
for EX_t_test in EXs_t_test:
    ax.plot3D(EX_t_test[:, 0], EX_t_test[:, 1], EX_t_test[:, 2], color='k', linestyle='--', alpha=0.2)
# ax.scatter3D(e_mu_moe[:, 0], e_mu_moe[:, 1], e_mu_moe[:, 2], marker='s', s=markersize, color=col)
# legends = ['Free', 'Free', 'Free', 'Wood', 'Wood', 'Wood', 'Wood & Steel', 'Wood & Steel']
legends = None
i = 0
for label in labels:
    e_mu_moe_i = e_mu_moe[(L_moe == label)]
    cl = label_col_dict[label]
    ax.scatter3D(e_mu_moe_i[:, 0], e_mu_moe_i[:, 1], e_mu_moe_i[:, 2], marker='s', s=markersize, color=cl)
    i = i+1
plotEllipsiodError(e_mu_moe, e_var_moe, col, ax=ax, alpha=0.05)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_title('Comparison of predicted mean trajectories')
ax.legend()

def aggregrate_statistics(nll_list, n):
    nll_list = np.array(nll_list)
    s0 = np.full((nll_list.shape[0]), n)
    mu = nll_list[:,0].reshape(-1)
    sigma = nll_list[:, 1].reshape(-1)
    sigma_sq = np.square(sigma)
    s1 = s0*mu
    s2 = s0*(np.square(mu)*sigma_sq)
    S0 = np.sum(s0)
    S1 = np.sum(s1)
    S2 = np.sum(s2)
    Sigma_sq = S2/S0 - (S1/S0)**2
    Sigma = np.sqrt(Sigma_sq)
    Mu = np.mean(mu)
    return Mu, Sigma

print('NLL-15:')
print('GP:', yumi_gp_sd_results['nll'])
mu, sig = aggregrate_statistics(yumi_mgp_sd_results['nll'], 500)
print('mGP(c):', (mu, sig))
mu, sig = aggregrate_statistics(yumi_mgp_uc_sd_results['nll'], 500)
print('mGP:', (mu, sig))
mu, sig = aggregrate_statistics(yumi_bnn_sd_results['nll'], 500)
print('BNN(c):', (mu, sig))
mu, sig = aggregrate_statistics(yumi_bnn_uc_sd_results['nll'], 500)
print('BNN:', (mu, sig))
print('MOE:', yumi_moe_sd_results['nll'])

print('RMSE-15:')
print('GP:', yumi_gp_sd_results['rmse'])
print('mGP(c):', np.mean(yumi_mgp_sd_results['rmse']))
print('mGP:', np.mean(yumi_mgp_uc_sd_results['rmse']))
print('BNN(c):', np.mean(yumi_bnn_sd_results['rmse']))
print('BNN:', np.mean(yumi_bnn_uc_sd_results['rmse']))
print('MOE:', yumi_moe_sd_results['rmse'])

print('Pred-time-15:')
print('GP:', yumi_gp_sd_results['gp_pred_time'])
print('BNN:', np.mean(yumi_bnn_sd_results['pred_time']))
print('MOE:', yumi_moe_sd_results['moe_pred_time'])

print('Train-time-15:')
print('GP:', yumi_gp_sd_results['gp_training_time'])
print('BNN:', np.mean(yumi_bnn_sd_results['train_time']))
moe_train_time = yumi_moe_sd_results['expert_train_time'] + yumi_moe_sd_results['cluster_time']\
+ yumi_moe_sd_results['trans_gp_time'] + yumi_moe_sd_results['svm_train_time']
print('MOE:', moe_train_time)

print('NLL-40:')
mu, sig = aggregrate_statistics(yumi_bnn_uc_bd_results['nll'], 500)
print('BNN:', (mu, sig))
print('MOE:', yumi_moe_bd_results['nll'])

print('RMSE-40:')
print('BNN:', np.mean(yumi_bnn_uc_bd_results['rmse']))
print('MOE:', yumi_moe_bd_results['rmse'])

print('Pred-time-40:')
print('BNN:', np.mean(yumi_bnn_uc_bd_results['pred_time']))
print('MOE:', yumi_moe_bd_results['moe_pred_time'])

print('Train-time-40:')
print('BNN:', np.mean(yumi_bnn_uc_bd_results['train_time']))
clustering_time = 264.99
BNN_pred_time_dens =  3.083568811416626
expert_train_time = yumi_moe_bd_results['expert_train_time']
init_train_time = yumi_moe_bd_results['trans_gp_time']
mode_predic_train_time = yumi_moe_bd_results['svm_train_time']
print('clust clustering_time', clustering_time)
print('expert_train_time', expert_train_time)
print('init_train_time', init_train_time)
print('mode_predic_train_time', mode_predic_train_time)

moe_train_time = expert_train_time + clustering_time\
+ init_train_time + mode_predic_train_time
print('MOE:', moe_train_time)
None
