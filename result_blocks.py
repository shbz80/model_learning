import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import pickle
from model_leraning_utils import traj_with_globalgp
from YumiKinematics import YumiKinematics
from mjc_exp_policy import kin_params


############# small data analysis #####################

blocks_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_mm_smalldata.p"
blocks_gp_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_gp_smalldata.p"
blocks_mgp_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_mgp_smalldata.p"
blocks_bnn_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_bnn_smalldata.p"
blocks_moe_sd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_moe_smalldata.p"

blocks_gp_d15_logfile = "./Results/Final/results_blocks_gp_d15.p"
blocks_moe_d15_logfile = "./Results/Final/results_blocks_moe_d15.p"
blocks_moe_d40_logfile = "./Results/Final/results_blocks_moe_d40.p"

blocks_sd_data = pickle.load( open(blocks_sd_logfile, "rb" ))
blocks_gp_sd_results = pickle.load( open(blocks_gp_sd_logfile, "rb" ))
blocks_mgp_sd_results = pickle.load( open(blocks_mgp_sd_logfile, "rb" ))
blocks_bnn_sd_results = pickle.load( open(blocks_bnn_sd_logfile, "rb" ))
blocks_moe_sd_results = pickle.load( open(blocks_moe_sd_logfile, "rb" ))

blocks_gp_d15_results = pickle.load( open(blocks_gp_d15_logfile, "rb" ))
blocks_moe_d15_results = pickle.load( open(blocks_moe_d15_logfile, "rb" ))
blocks_moe_d40_results = pickle.load( open(blocks_moe_d40_logfile, "rb" ))

# blocks_moe_sd_results = blocks_moe_d15_results

blocks_sd_params = blocks_sd_data['exp_params']
dP = blocks_sd_params['dP']
dV = blocks_sd_params['dV']
dU = blocks_sd_params['dU']
dX = dP+dV
T = blocks_sd_params['T'] - 1
dt = blocks_sd_params['dt']
XUs_t_test = blocks_sd_data['XUs_t_test']
Xs_t_test = XUs_t_test[:, :, :dX]
X_t_test = Xs_t_test.reshape(-1,Xs_t_test.shape[-1])
n_test, H, _ = Xs_t_test.shape

blocks_gp_density = blocks_gp_sd_results['density_est'].traj_density
blocks_mgp_density = blocks_mgp_sd_results['density_est'].traj_density
blocks_bnn_density = blocks_bnn_sd_results['density_est'].traj_density
K = 2
gp_traj_means = np.zeros((H, K, dX + 1))
gp_traj_stds = np.zeros((H, K, dX + 1))
mgp_traj_means = np.zeros((H, K, dX + 1))
mgp_traj_stds = np.zeros((H, K, dX + 1))
bnn_traj_means = np.zeros((H, K, dX + 1))
bnn_traj_stds = np.zeros((H, K, dX + 1))
for t in range(H):
    for k in range(K):
        gp_traj_means[t, k, :dX] = blocks_gp_density[t][1][k]
        gp_traj_means[t, k, dX:] = blocks_gp_density[t][0][k]
        mgp_traj_means[t, k, :dX] = blocks_mgp_density[t][1][k]
        mgp_traj_means[t, k, dX:] = blocks_mgp_density[t][0][k]
        bnn_traj_means[t, k, :dX] = blocks_bnn_density[t][1][k]
        bnn_traj_means[t, k, dX:] = blocks_bnn_density[t][0][k]
        gp_traj_stds[t, k, :dX] = np.sqrt(np.diag(blocks_gp_density[t][2][k]))
        gp_traj_stds[t, k, dX:] = blocks_gp_density[t][0][k]
        mgp_traj_stds[t, k, :dX] = np.sqrt(np.diag(blocks_mgp_density[t][2][k]))
        mgp_traj_stds[t, k, dX:] = blocks_mgp_density[t][0][k]
        bnn_traj_stds[t, k, :dX] = np.sqrt(np.diag(blocks_bnn_density[t][2][k]))
        bnn_traj_stds[t, k, dX:] = blocks_bnn_density[t][0][k]

#########plot d15 comparison################################
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 22})
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['text.usetex'] = True
tm = np.array(range(H)) * dt
plt.figure()
plt.subplot(121)
# plt.title('Position')
plt.subplot(122)
# plt.title('Velocity')
marker = 'o'
marker_size = 20
path_dict = blocks_moe_sd_results['path_data']
cl_moe = 'm'
rgb_moe = plt_colors.to_rgba(cl_moe)[:3]
plots = []

for k in range(K):
    plt.subplot(121)
    # gp
    cl_gp = 'g'
    rgb_gp = plt_colors.to_rgba(cl_gp)
    rgba_gp = np.tile(rgb_gp, (H, 1))
    prob_gp = gp_traj_means[:, k, 2]
    rgba_gp[:, 3] = prob_gp.reshape(-1)
    if k == 0:
        label = 'GP'
        plots.append(plt.scatter(tm, gp_traj_means[:, k, 0], color=rgba_gp, marker=marker, s=marker_size, label=label))
    else:
        label = None
        plt.scatter(tm, gp_traj_means[:, k, 0], color=rgba_gp, marker=marker, s=marker_size, label=label)
    # plt.scatter(tm, gp_traj_means[:, k, 0] + 1.96 * gp_traj_stds[:, k, 0], color=rgba_gp, marker='_')
    # plt.scatter(tm, gp_traj_means[:, k, 0] - 1.96 * gp_traj_stds[:, k, 0], color=rgba_gp, marker='_')

    # mgp
    cl_mgp = 'b'
    rgb_mgp = plt_colors.to_rgba(cl_mgp)
    rgba_mgp = np.tile(rgb_mgp, (H, 1))
    prob_mgp = mgp_traj_means[:, k, 2]
    rgba_mgp[:, 3] = prob_mgp.reshape(-1)
    if k == 0:
        label = 'mGP'
        plots.append(
            plt.scatter(tm, mgp_traj_means[:, k, 0], color=rgba_mgp, marker=marker, s=marker_size, label=label))
    else:
        label = None
        plt.scatter(tm, mgp_traj_means[:, k, 0], color=rgba_mgp, marker=marker, s=marker_size, label=label)
    # plt.scatter(tm, mgp_traj_means[:, k, 0] + 1.96 * mgp_traj_stds[:, k, 0], color=rgba_mgp, marker='_')
    # plt.scatter(tm, mgp_traj_means[:, k, 0] - 1.96 * mgp_traj_stds[:, k, 0], color=rgba_mgp, marker='_')

    # bnn
    cl_bnn = 'c'
    rgb_bnn = plt_colors.to_rgba(cl_bnn)
    rgba_bnn = np.tile(rgb_bnn, (H, 1))
    prob_bnn = bnn_traj_means[:, k, 2]
    rgba_bnn[:, 3] = prob_bnn.reshape(-1)
    if k == 0:
        label = 'uANN'
        plots.append(
            plt.scatter(tm, bnn_traj_means[:, k, 0], color=rgba_bnn, marker=marker, s=marker_size, label=label))
    else:
        label = None
        plt.scatter(tm, bnn_traj_means[:, k, 0], color=rgba_bnn, marker=marker, s=marker_size, label=label)
    # plt.scatter(tm, bnn_traj_means[:, k, 0] + 1.96 * bnn_traj_stds[:, k, 0], color=rgba_bnn, marker='_')
    # plt.scatter(tm, bnn_traj_means[:, k, 0] - 1.96 * bnn_traj_stds[:, k, 0], color=rgba_bnn, marker='_')

    plt.subplot(122)
    # plt.title('Velocity')
    # gp
    plt.scatter(tm, gp_traj_means[:, k, 1], color=rgba_gp, marker=marker, s=marker_size)
    # plt.scatter(tm, gp_traj_means[:, k, 1] + 1.96 * gp_traj_stds[:, k, 1], color=rgba_gp, marker='_')
    # plt.scatter(tm, gp_traj_means[:, k, 1] - 1.96 * gp_traj_stds[:, k, 1], color=rgba_gp, marker='_')

    # mgp
    plt.scatter(tm, mgp_traj_means[:, k, 1], color=rgba_mgp, marker=marker, s=marker_size)
    # plt.scatter(tm, mgp_traj_means[:, k, 1] + 1.96 * mgp_traj_stds[:, k, 1], color=rgba_mgp, marker='_')
    # plt.scatter(tm, mgp_traj_means[:, k, 1] - 1.96 * mgp_traj_stds[:, k, 1], color=rgba_mgp, marker='_')

    # bnn
    plt.scatter(tm, bnn_traj_means[:, k, 1], color=rgba_bnn, marker=marker, s=marker_size)
    # plt.scatter(tm, bnn_traj_means[:, k, 1] + 1.96 * bnn_traj_stds[:, k, 1], color=rgba_bnn, marker='_')
    # plt.scatter(tm, bnn_traj_means[:, k, 1] - 1.96 * bnn_traj_stds[:, k, 1], color=rgba_bnn, marker='_')

j=0
for path_key in path_dict:
    path = path_dict[path_key]
    time = np.array(path['time'])
    pos = np.array(path['X'])[:,:dP].reshape(-1)
    pos_std = np.sqrt(np.array(path['X_var'])[:, :dP, :dP]).reshape(time.shape[0])
    vel = np.array(path['X'])[:, dP:dX].reshape(-1)
    vel_std = np.sqrt(np.array(path['X_var'])[:, dP:dX, dP:dX]).reshape(time.shape[0])
    prob = np.array(path['prob']).reshape(-1,1)
    prob = np.clip(prob, 0., 1.)
    col = np.tile(rgb_moe, (time.shape[0],1))
    rbga_col = np.concatenate((col, prob), axis=1)
    if j == 2:
        label = "our method"
        # label = "train"
        plt.subplot(121)
        plots.append(plt.scatter(time, pos, c=rbga_col, marker=marker, s=marker_size, label=label))
        # plt.fill_between(time, pos - pos_std * 1.96, pos + pos_std * 1.96, alpha=0.2, color=path['col'])
    else:
        label = None
        plt.subplot(121)
        plt.scatter(time, pos, c=rbga_col, marker=marker, s=marker_size, label=label)
        # plt.fill_between(time, pos - pos_std * 1.96, pos + pos_std * 1.96, alpha=0.2, color=path['col'])
    plt.subplot(122)
    plt.scatter(time, vel, c=rbga_col, marker=marker, s=marker_size)
    # plt.fill_between(time, vel - vel_std * 1.96, vel + vel_std * 1.96, alpha=0.2, color=path['col'])
    j += 1

plt.subplot(121)
for i in range(Xs_t_test.shape[0]):
    if i == 0:
        label = "$\mathbb{D}^{test}_{15}$"
        # label = "train"
        x = Xs_t_test[i]
        plots.append(plt.plot(tm, x[:H, :dP], ls='--', color='k', alpha=0.4, linewidth=2, label=label)[0])
    else:
        label = None
        x = Xs_t_test[i]
        plt.plot(tm, x[:H, :dP], ls='--', color='k', alpha=0.4, linewidth=2, label=label)

plt.legend(plots, ['GP','mGP','uANN','Ours','$\mathbb{D}^{test}_{15}$'], loc='upper left', frameon=False)
plt.subplot(122)
for i in range(Xs_t_test.shape[0]):
    x = Xs_t_test[i]
    plt.plot(tm, x[:H, dP:dP+dV], ls='--', color='k', linewidth=2, alpha=0.4)



plt.subplot(121)
plt.xlabel('Time '+'$(s)$')
plt.ylabel('Position '+'$(m)$')
plt.subplot(122)
plt.xlabel('Time '+'$(s)$')
plt.ylabel('Velocity '+'$(m/s)$')
# plt.savefig('/home/shahbaz/Dropbox/Workspace/Model Learning/Figures/blocks_comparison.pdf')
plt.show(block=False)
#########plot d15 comparison################################

#########plot d15 lt prediction################################
# plot lt-prediction blocks
plt.rcParams.update({'font.size': 20})
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['pdf.fonttype'] = 42
marker = 'o'
marker_size = 20

plt.figure()
plt.subplot(121)
plt.xlabel('Time '+'$(t)$')
plt.ylabel('Position '+'$(m)$')
plt.subplot(122)
plt.xlabel('Time '+'$(t)$')
plt.ylabel('Velocity '+'$(m/s)$')
j=0
list_l = []
t_for_legend = [10, 10, 10, 42]
for path_key in path_dict:
    path = path_dict[path_key]
    time = np.array(path['time'])
    pos = np.array(path['X'])[:, :dP].reshape(-1)
    pos_std = np.sqrt(np.array(path['X_var'])[:, :dP, :dP]).reshape(time.shape[0])
    vel = np.array(path['X'])[:, dP:dX].reshape(-1)
    vel_std = np.sqrt(np.array(path['X_var'])[:, dP:dX, dP:dX]).reshape(time.shape[0])
    prob = np.array(path['prob']).reshape(-1, 1)
    prob = np.clip(prob, 0., 1.)
    col = np.tile(path['col'], (time.shape[0], 1))
    rbga_col = np.concatenate((col, prob), axis=1)
    plt.subplot(121)
    plt.fill_between(time, pos - pos_std * 1.96, pos + pos_std * 1.96, alpha=0.2, color=path['col'])
    plt.scatter(time, pos, c=rbga_col, marker=marker, s=marker_size)
    t = t_for_legend[j]
    list_l.append([time[t], pos[t], rbga_col[t]])
    plt.subplot(122)
    plt.fill_between(time, vel - vel_std * 1.96, vel + vel_std * 1.96, alpha=0.2, color=path['col'])
    plt.scatter(time, vel, c=rbga_col, marker=marker, s=marker_size)
    j += 1


labels=['free', 'stick', 'free', 'slip']

plt.subplot(121)
time, pos, col = list_l[2]
plt.scatter(time, pos, c=col, marker=marker, s=marker_size, label=labels[2])
time, pos, col = list_l[0]
plt.scatter(time, pos, c=col, marker=marker, s=marker_size, label=labels[0])
time, pos, col = list_l[1]
plt.scatter(time, pos, c=col, marker=marker, s=marker_size, label=labels[1])
time, pos, col = list_l[3]
plt.scatter(time, pos, c=col, marker=marker, s=marker_size, label=labels[3])

# plot training data
plt.subplot(121)
for i in range(Xs_t_test.shape[0]):
    if i == 0:
        label = "$\mathbb{D}^{test}_{15}$"
        # label = "train"
    else:
        label = None
    x = Xs_t_test[i]
    plt.plot(tm, x[:H, :dP], ls='--', color='k', alpha=0.4, linewidth=2, label=label)
plt.legend(loc='upper left', frameon=False)

plt.subplot(122)
for i in range(Xs_t_test.shape[0]):
    x = Xs_t_test[i]
    plt.plot(tm, x[:H, dP:dP+dV], ls='--', color='k', linewidth=2,  alpha=0.4)
# plt.savefig('/home/shahbaz/Dropbox/Workspace/Model Learning/Figures/blocks_prediction.pdf')
plt.show(block=False)
#########plot d15 lt prediction################################

def aggregrate_statistics(nll_list, n):
    nll_list = np.array(nll_list)
    s0 = np.full((nll_list.shape[0]), n)
    mu = nll_list[:,0].reshape(-1)
    sigma = nll_list[:, 1].reshape(-1)
    sigma_sq = np.square(sigma)
    s1 = s0*mu
    s2 = s0*(np.square(mu)+sigma_sq)
    S0 = np.sum(s0)
    S1 = np.sum(s1)
    S2 = np.sum(s2)
    Sigma_sq = S2/S0 - (S1/S0)**2
    Sigma = np.sqrt(Sigma_sq)
    Mu = np.mean(mu)
    return Mu, Sigma

print('NLL-15:')
# print('GP:', blocks_gp_sd_results['nll'])
mu, sig = aggregrate_statistics(blocks_gp_d15_results['nll'], 750)
print('GP:', (-mu, sig))
mu, sig = aggregrate_statistics(blocks_mgp_sd_results['nll'], 750)
print('mGP:', (-mu, sig))
mu, sig = aggregrate_statistics(blocks_bnn_sd_results['nll'], 750)
print('uANN:', (-mu, sig))
# print('MOE:', blocks_moe_sd_results['nll'])
nll_l = blocks_moe_d15_results['nll']
del nll_l[1]
mu, sig = aggregrate_statistics(nll_l, 750)
print('MOE:', (-mu, sig))

print('RMSE-15:')
# print('GP:', blocks_gp_sd_results['rmse'])
print('GP:', np.mean(blocks_gp_d15_results['rmse']))
print('mGP:', np.mean(blocks_mgp_sd_results['rmse']))
print('uANN:', np.mean(blocks_bnn_sd_results['rmse']))
# print('MOE:', blocks_moe_sd_results['rmse'])
print('MOE:', np.mean(blocks_moe_d15_results['rmse']))

print('Pred-time-15:')
print('GP:', blocks_gp_sd_results['gp_pred_time'])
print('uANN:', np.mean(blocks_bnn_sd_results['pred_time']))
print('MOE:', blocks_moe_sd_results['moe_pred_time'])

print('Train-time-15:')
print('GP:', blocks_gp_sd_results['gp_training_time'])
print('uANN:', np.mean(blocks_bnn_sd_results['train_time']))
moe_train_time = blocks_moe_sd_results['expert_train_time'] + blocks_moe_sd_results['cluster_time']\
+ blocks_moe_sd_results['trans_gp_time'] + blocks_moe_sd_results['svm_train_time']
print('MOE:', moe_train_time)

# ############# big data analysis #####################
blocks_bd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_mm_bigdata.p"
blocks_gp_bd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_gp_bigdata.p"
blocks_bnn_bd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_bnn_bigdata.p"
blocks_moe_bd_logfile = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_moe_bigdata.p"

blocks_bd_data = pickle.load( open(blocks_bd_logfile, "rb" ))
blocks_gp_bd_results = pickle.load( open(blocks_gp_bd_logfile, "rb" ))
blocks_bnn_bd_results = pickle.load( open(blocks_bnn_bd_logfile, "rb" ))
blocks_moe_bd_results = pickle.load( open(blocks_moe_bd_logfile, "rb" ))

blocks_bd_params = blocks_bd_data['exp_params']
dP = blocks_bd_params['dP']
dV = blocks_bd_params['dV']
dU = blocks_bd_params['dU']
dX = dP+dV
T = blocks_bd_params['T'] - 1
dt = blocks_bd_params['dt']
XUs_t_test = blocks_bd_data['XUs_t_test']
Xs_t_test = XUs_t_test[:5, :, :dX]
X_t_test = Xs_t_test.reshape(-1,Xs_t_test.shape[-1])
n_test, H, _ = Xs_t_test.shape

blocks_gp_density = blocks_gp_bd_results['density_est'].traj_density
blocks_bnn_density = blocks_bnn_bd_results['density_est'].traj_density
K = 2
gp_traj_means = np.zeros((H, K, dX + 1))
gp_traj_stds = np.zeros((H, K, dX + 1))
bnn_traj_means = np.zeros((H, K, dX + 1))
bnn_traj_stds = np.zeros((H, K, dX + 1))
for t in range(H):
    for k in range(K):
        gp_traj_means[t, k, :dX] = blocks_gp_density[t][1][k]
        gp_traj_means[t, k, dX:] = blocks_gp_density[t][0][k]
        bnn_traj_means[t, k, :dX] = blocks_bnn_density[t][1][k]
        bnn_traj_means[t, k, dX:] = blocks_bnn_density[t][0][k]
        gp_traj_stds[t, k, :dX] = np.sqrt(np.diag(blocks_gp_density[t][2][k]))
        gp_traj_stds[t, k, dX:] = blocks_gp_density[t][0][k]
        bnn_traj_stds[t, k, :dX] = np.sqrt(np.diag(blocks_bnn_density[t][2][k]))
        bnn_traj_stds[t, k, dX:] = blocks_bnn_density[t][0][k]

#########plot d40 comparison################################
plt.rcParams.update({'font.size': 22})
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['pdf.fonttype'] = 42
tm = np.array(range(H)) * dt
plt.figure()
plt.subplot(121)
# plt.title('Position')
plt.subplot(122)
# plt.title('Velocity')
marker = 'o'
marker_size = 20
for k in range(K):
    plt.subplot(121)
    # gp
    cl_gp = 'g'
    rgb_gp = plt_colors.to_rgba(cl_gp)
    rgba_gp = np.tile(rgb_gp, (H, 1))
    prob_gp = gp_traj_means[:, k, 2]
    rgba_gp[:, 3] = prob_gp.reshape(-1)
    if k == 0:
        label = 'GP'
    else:
        label = None
    plt.scatter(tm, gp_traj_means[:, k, 0], color=rgba_gp, marker=marker, s=marker_size, label=label)
    # plt.scatter(tm, gp_traj_means[:, k, 0] + 1.96 * gp_traj_stds[:, k, 0], color=rgba_gp, marker='_')
    # plt.scatter(tm, gp_traj_means[:, k, 0] - 1.96 * gp_traj_stds[:, k, 0], color=rgba_gp, marker='_')


    # bnn
    cl_bnn = 'c'
    rgb_bnn = plt_colors.to_rgba(cl_bnn)
    rgba_bnn = np.tile(rgb_bnn, (H, 1))
    prob_bnn = bnn_traj_means[:, k, 2]
    rgba_bnn[:, 3] = prob_bnn.reshape(-1)
    if k == 0:
        label = 'uANN'
    else:
        label = None
    plt.scatter(tm, bnn_traj_means[:, k, 0], color=rgba_bnn, marker=marker, s=marker_size, label=label)
    # plt.scatter(tm, bnn_traj_means[:, k, 0] + 1.96 * bnn_traj_stds[:, k, 0], color=rgba_bnn, marker='_')
    # plt.scatter(tm, bnn_traj_means[:, k, 0] - 1.96 * bnn_traj_stds[:, k, 0], color=rgba_bnn, marker='_')

    plt.subplot(122)
    # plt.title('Velocity')
    # gp
    plt.scatter(tm, gp_traj_means[:, k, 1], color=rgba_gp, marker=marker, s=marker_size)
    # plt.scatter(tm, gp_traj_means[:, k, 1] + 1.96 * gp_traj_stds[:, k, 1], color=rgba_gp, marker='_')
    # plt.scatter(tm, gp_traj_means[:, k, 1] - 1.96 * gp_traj_stds[:, k, 1], color=rgba_gp, marker='_')


    # bnn
    plt.scatter(tm, bnn_traj_means[:, k, 1], color=rgba_bnn, marker=marker, s=marker_size)
    # plt.scatter(tm, bnn_traj_means[:, k, 1] + 1.96 * bnn_traj_stds[:, k, 1], color=rgba_bnn, marker='_')
    # plt.scatter(tm, bnn_traj_means[:, k, 1] - 1.96 * bnn_traj_stds[:, k, 1], color=rgba_bnn, marker='_')

# path_dict = blocks_moe_bd_results['path_data']
path_dict = blocks_moe_d40_results['path_data']

cl_moe = 'm'
rgb_moe = plt_colors.to_rgba(cl_moe)[:3]

j=0
for path_key in path_dict:
    path = path_dict[path_key]
    time = np.array(path['time'])
    pos = np.array(path['X'])[:,:dP].reshape(-1)
    pos_std = np.sqrt(np.array(path['X_var'])[:, :dP, :dP]).reshape(time.shape[0])
    vel = np.array(path['X'])[:, dP:dX].reshape(-1)
    vel_std = np.sqrt(np.array(path['X_var'])[:, dP:dX, dP:dX]).reshape(time.shape[0])
    prob = np.array(path['prob']).reshape(-1,1)
    prob = np.clip(prob, 0., 1.)
    col = np.tile(rgb_moe, (time.shape[0],1))
    rbga_col = np.concatenate((col, prob), axis=1)
    if j == 2:
        label = "our method"
        # label = "train"
    else:
        label = None
    plt.subplot(121)
    plt.scatter(time, pos, c=rbga_col, marker=marker, s=marker_size, label=label)
    # plt.fill_between(time, pos - pos_std * 1.96, pos + pos_std * 1.96, alpha=0.2, color=path['col'])
    plt.subplot(122)
    plt.scatter(time, vel, c=rbga_col, marker=marker, s=marker_size)
    # plt.fill_between(time, vel - vel_std * 1.96, vel + vel_std * 1.96, alpha=0.2, color=path['col'])
    j += 1

plt.subplot(121)
for i in range(Xs_t_test.shape[0]):
    if i == 0:
        label = "$\mathbb{D}^{test}_{40}$"
        # label = "train"
    else:
        label = None
    x = Xs_t_test[i]
    plt.plot(tm, x[:H, :dP], ls='--', color='k', linewidth=2, alpha=0.4, label=label)
plt.legend()

plt.subplot(122)
for i in range(Xs_t_test.shape[0]):
    x = Xs_t_test[i]
    plt.plot(tm, x[:H, dP:dP+dV], ls='--', color='k', linewidth=2, alpha=0.4)

plt.subplot(121)
plt.xlabel('Time '+'$(t)$')
plt.ylabel('Position '+'$(m)$')
plt.subplot(122)
plt.xlabel('Time '+'$(t)$')
plt.ylabel('Velocity '+'$(m/s)$')
plt.show(block=False)
# plt.savefig('/home/shahbaz/Dropbox/Workspace/Model Learning/Figures/blocks_comparison.pdf')
#########plot d40 comparison################################

#########plot d40 lt-prediction################################
# plot lt-prediction blocks
plt.rcParams.update({'font.size': 20})
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['pdf.fonttype'] = 42
marker = 'o'
marker_size = 20

fig = plt.figure()
# fig.subplots_adjust(left=0.00, bottom=0.00, right=1., top=1.)
plt.subplot(121)
plt.xlabel('Time '+'$(s)$')
plt.ylabel('Position '+'$(m)$')
plt.subplot(122)
plt.xlabel('Time '+'$(s)$')
plt.ylabel('Velocity '+'$(m/s)$')
j=0
list_l = []
t_for_legend = [10, 10, 30, 10]
list_col = ['b','g','m','c']
for path_key in path_dict:
    path = path_dict[path_key]
    time = np.array(path['time'])
    pos = np.array(path['X'])[:, :dP].reshape(-1)
    pos_std = np.sqrt(np.array(path['X_var'])[:, :dP, :dP]).reshape(time.shape[0])
    vel = np.array(path['X'])[:, dP:dX].reshape(-1)
    vel_std = np.sqrt(np.array(path['X_var'])[:, dP:dX, dP:dX]).reshape(time.shape[0])
    prob = np.array(path['prob']).reshape(-1, 1)
    prob = np.clip(prob, 0., 1.)
    # path_col = path['col']
    path_col = plt_colors.to_rgb(list_col[j])
    col = np.tile(path_col, (time.shape[0], 1))
    rbga_col = np.concatenate((col, prob), axis=1)
    plt.subplot(121)
    plt.fill_between(time, pos - pos_std * 1.96, pos + pos_std * 1.96, alpha=0.2, color=path['col'])
    plt.scatter(time, pos, c=rbga_col, marker=marker, s=marker_size)
    t = t_for_legend[j]
    list_l.append([time[t], pos[t], rbga_col[t]])
    plt.subplot(122)
    plt.fill_between(time, vel - vel_std * 1.96, vel + vel_std * 1.96, alpha=0.2, color=path['col'])
    plt.scatter(time, vel, c=rbga_col, marker=marker, s=marker_size)
    j += 1


labels=['free', 'stick', 'slip', 'free']

plt.subplot(121)
time, pos, col = list_l[0]
plt.scatter(time, pos, c=col, marker=marker, s=marker_size, label=labels[0])
time, pos, col = list_l[3]
plt.scatter(time, pos, c=col, marker=marker, s=marker_size, label=labels[3])
time, pos, col = list_l[1]
plt.scatter(time, pos, c=col, marker=marker, s=marker_size, label=labels[1])
time, pos, col = list_l[2]
plt.scatter(time, pos, c=col, marker=marker, s=marker_size, label=labels[2])

# plot training data
plt.subplot(121)
for i in range(Xs_t_test.shape[0]):
    if i == 0:
        label = "$\mathbb{D}^{test}_{40}$"
        # label = "train"
    else:
        label = None
    x = Xs_t_test[i]
    plt.plot(tm, x[:H, :dP], ls='--', color='k', linewidth=2, alpha=0.4, label=label)
plt.legend(loc='upper left', frameon=False)

plt.subplot(122)
for i in range(Xs_t_test.shape[0]):
    x = Xs_t_test[i]
    plt.plot(tm, x[:H, dP:dP+dV], ls='--', color='k', linewidth=2, alpha=0.4)
    # plt.show(block=False)
# plt.savefig('/home/shahbaz/Dropbox/Workspace/Model Learning/Figures/blocks_prediction.pdf')
plt.show(block=False)
#########plot d40 lt-prediction################################


print('NLL-40:')
print('GP:', (-blocks_gp_bd_results['nll'][0], blocks_gp_bd_results['nll'][1]))
mu, sig = aggregrate_statistics(blocks_bnn_bd_results['nll'], 750)
print('uANN:', (-mu, sig))
# print('MOE:', blocks_moe_bd_results['nll'])
mu, sig = aggregrate_statistics(blocks_moe_d40_results['nll'], 750)
print('MOE:', (-mu, sig))

print('RMSE-40:')
print('GP:', blocks_gp_bd_results['rmse'])
print('uANN:', np.mean(blocks_bnn_bd_results['rmse']))
print('MOE:', np.mean(blocks_moe_d40_results['rmse']))

print('Pred-time-40:')
print('GP:', blocks_gp_bd_results['gp_pred_time'])
print('uANN:', blocks_bnn_bd_results['pred_time'])
print('MOE:', blocks_moe_bd_results['moe_pred_time'])

print('Train-time-40:')
print('GP:', blocks_gp_bd_results['gp_training_time'])
print('uANN:', np.mean(blocks_bnn_bd_results['train_time']))
moe_train_time = blocks_moe_bd_results['expert_train_time'] + blocks_moe_bd_results['cluster_time']\
+ blocks_moe_bd_results['trans_gp_time'] + blocks_moe_bd_results['svm_train_time']
print('MOE:', moe_train_time)

None