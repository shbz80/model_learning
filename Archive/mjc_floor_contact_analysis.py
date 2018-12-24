import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mat_col
from mpl_toolkits.mplot3d import axes3d
import pickle
from mixture_model_gibbs_sampling import ACF
from collections import Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from collections import Counter
from utilities import *
from copy import deepcopy
from collapsed_Gibbs_sampler import predictive_ll_cluster
import operator
from gmm import GMM

# import PyKDL as kdl
# import pykdl_utils
# import hrl_geom.transformations as trans
# from hrl_geom.pose_converter import PoseConv
# from urdf_parser_py.urdf import Robot
# from pykdl_utils.kdl_kinematics import *
# import pydart2 as pydart



f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')
# euler_from_matrix = pydart.utils.transformations.euler_from_matrix
# J_G_to_A = jacobian_geometric_to_analytic
train_data_id = 1
# task = 'raw_data'
# task = 'GMM'
# task = 'visual'
task = 'compute'
# raw exp data visualization

# exp_params = sample_data['exp_params']

if task=='GMM':
    sample_data = pickle.load( open( "mjc_floor_contact_raw.p", "rb" ) )
    #pykdl stuff
    # robot = Robot.from_xml_string(f.read())
    # base_link = robot.get_root()
    # end_link = 'left_tool0'
    # kdl_kin = KDLKinematics(robot, base_link, end_link)
    #
    # X = sample_data['X'] # N X T X dX
    # U = sample_data['U'] # N X T X dU
    #
    mjc_exp_params = sample_data['exp_params']
    dP = mjc_exp_params['dP']
    dV = mjc_exp_params['dV']
    dU = mjc_exp_params['dU']
    # N, T, dX = X.shape
    #
    # assert(X.shape[2]==(dP+dV))
    # assert(dP==dV)
    #
    # XU = np.zeros((N,T,dX+dU))
    # for n in range(N):
    #     XU[n] = np.concatenate((X[n,:,:],U[n,:,:]),axis=1)
    # XU_t = XU[:,:-1,:]
    # X_t1 = XU[:,1:,:dX]
    # X_t = XU[:,:-1,:dX]
    # delX = X_t1 - X_t
    # dynamics_data = np.concatenate((XU_t,X_t1),axis=2)
    # train_data = dynamics_data[0:train_data_id,:,:]
    # test_data = dynamics_data[train_data_id:,:,:]
    #
    # train_data_flattened = train_data.reshape((-1,train_data.shape[-1]))
    # X_train = train_data_flattened[:,0:dP+dV]
    #
    # Qt = X_train[:,0:dP].reshape((-1,dP))
    # Qt_d = X_train[:,dP:dP+dV].reshape((-1,dV))
    # Qt = Qt.reshape((-1,Qt.shape[-1]))
    # Qt_d = Qt_d.reshape((-1,Qt_d.shape[-1]))
    # Xt = np.concatenate((Qt,Qt_d),axis=1)
    # Et = np.zeros((Qt.shape[0],6))
    # Et_d = np.zeros((Qt.shape[0],6))
    # for i in range(Qt.shape[0]):
    #     Tr = kdl_kin.forward(Qt[i], end_link=end_link, base_link=base_link)
    #     epos = np.array(Tr[:3,3])
    #     epos = epos.reshape(-1)
    #     erot = np.array(Tr[:3,:3])
    #     erot = euler_from_matrix(erot)
    #     Et[i] = np.append(epos,erot)
    #
    #     J_G = np.array(kdl_kin.jacobian(Qt[i]))
    #     J_G = J_G.reshape((6,7))
    #     J_A = J_G_to_A(J_G, Et[i][3:])
    #     Et_d[i] = J_A.dot(Qt_d[i])
    #
    # # XU = np.zeros((N,T,dX+dU))
    # # for n in range(N):
    # #     XU[n] = np.concatenate((X[n,:,:],U[n,:,:]),axis=1)
    # # XU_t = XU[:,:-1,:]
    # # X_t1 = XU[:,1:,:dX]
    # # X_t = XU[:,:-1,:dX]
    # # delX = X_t1 - X_t
    #
    #
    #
    # # data = np.concatenate((XU_t,X_t1),axis=2)
    # # data = np.concatenate((XU_t,delX),axis=2)
    # data = np.concatenate((Et,Et_d),axis=1)
    # # data = data[0:5,:,:]
    # # data = data.reshape((-1,data.shape[-1]))
    #
    # # data_c = X_t
    # # train_data = data_c[0:5,:,:] # half-half training and test data, move this to exp_params
    # # train_data = train_data.reshape((-1,train_data.shape[-1]))
    # # data = data[::4] # down sample data
    data = sample_data['E_train']
    restarts = 20
    K = 2
    Gmm = []
    ll = np.zeros(restarts)
    for it in range(restarts):
        gmm = GMM(dxu=6,dxux=12)
        gmm.update(data,K)
        Gmm.append(deepcopy(gmm))
        ll[it] = gmm.ll
        print('GMM log likelihood:',ll[it])
        del gmm
    best_gmm_id = np.argmax(ll)
    best_gmm = Gmm[best_gmm_id]
    w = best_gmm.w
    idx = np.argmax(w,axis=1)

    colors = get_N_HexCol(K)
    colors = np.asarray(colors)
    col = np.zeros([data.shape[0],3])
    # idx = np.array(assignment)
    for k in range(K):
        col[(idx==k)] = colors[k]

    # p = data[:,:dP]
    # x_gripper = []
    # for i in range(train_data.shape[0]):
    #     q = p[i]
    #     print q
    #     gripper_T = kdl_kin.forward(q, end_link=end_link, base_link=base_link)
    #     epos = np.array(gripper_T[:3,3])
    #     epos = epos.reshape(-1)
    #     erot = np.array(gripper_T[:3,:3])
    #     erot = euler_from_matrix(erot)
    #     x = np.append(epos,erot)
    #     x_gripper.append(x)
    # x_gripper = np.array(x_gripper)

    # px = x_gripper[:,0]
    # py = x_gripper[:,1]
    # pz = x_gripper[:,2]
    Et = data[:,:dP]

    px = Et[:,0]
    py = Et[:,1]
    pz = Et[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter3D(px, py, pz, c = col/255.0)

    plt.show()

if task=='visual':
    # exp_result = pickle.load( open( "mjc_1d_finite_gauss_clustered.p", "rb" ) )
    exp_result = pickle.load( open( "floor_contact_DPM_clustered.p", "rb" ) )
    exp_stats = exp_result['exp_stats']
    params = exp_result['params']
    print 'alpha:', params['init_state']['hyperparameters_']['alpha_']
    # raw_input()
    # plots log likelihood
    # plt.figure()
    num_chains = len(exp_stats)
    for ic in range(num_chains):
        chain_len = len(exp_stats[ic])
        ll_stat = [exp_stats[ic][itr]['ll'] for itr in range(chain_len)]
        plt.figure()
        plt.title('Marginal likelihood, %d' %(ic))
        plt.plot(ll_stat)
    # plt.show()
    #
    # autocorrelation plot for cluster numbers
    # plt.figure()
    for ic in range(num_chains):
        chain_len = len(exp_stats[ic])
        kstat = [exp_stats[ic][itr]["num_clusters_"] for itr in range(chain_len)]
        kstat = np.array(kstat)
        log_k = np.log(kstat)
        Xt = log_k
        N = Xt.shape[0]
        auto_cor_plot = [np.array([t,ACF(Xt,t)]) for t in range(N/2)]
        auto_cor_plot = np.array(auto_cor_plot)
        plt.figure()
        plt.title('Auto-correlation of cluster count, %d' %(ic))
        plt.plot(auto_cor_plot[:,0],auto_cor_plot[:,1])
    plt.show()


if task=='compute':
    # exp_result = pickle.load( open( "mjc_1d_finite_gauss_clustered.p", "rb" ) )
    exp_result = pickle.load( open( "floor_contact_DPM_clustered.p", "rb" ) )
    sample_data = pickle.load( open( "mjc_floor_contact_raw.p", "rb" ) )
    exp_stats = exp_result['exp_stats']
    params = exp_result['params']

    # chose best chain from the above two plots
    # # exp_1
    # best_chain_id = 2
    # mix_itr = 200
    # mult = 5

    # exp_2
    best_chain_id = 3
    mix_itr = 170
    mult = 3

    best_chain = exp_stats[best_chain_id]
    chain_len = len(best_chain)
    burn_in_itr = mix_itr * mult
    convergence_itr = chain_len - burn_in_itr
    num_posterior_samples = convergence_itr/mix_itr
    post_samples_idx = [burn_in_itr + i*mix_itr for i in range(num_posterior_samples)]
    all_post_idx = range(burn_in_itr,chain_len)

    # plot posterior cluster counts
    # post_k_samples = [best_chain[itr]['num_clusters_'] for itr in all_post_idx]
    # cids, counts = zip(*Counter(post_k_samples).items())
    # plt.figure()
    # plt.bar(cids,counts)
    # plt.title('Posterior cluster counts from chain %d and all samples' %(best_chain_id))
    post_k_samples = [best_chain[itr]['num_clusters_'] for itr in all_post_idx]
    cids, counts = zip(*Counter(post_k_samples).items())
    plt.figure()
    plt.bar(cids,counts)
    plt.title('Posterior cluster counts from chain %d and %d independent samples' %(best_chain_id, num_posterior_samples))

    count_list = zip(*Counter(post_k_samples).items())
    count_list = np.array(count_list)
    inferred_k = count_list[0][np.argmax(count_list[1])]
    print 'Inferred number of Gaussian experts:', inferred_k

    # get the latest itr in the best chain for k=inferred_k
    # we use this for getting all parameters instead of mote carlo average TODO
    # for itr in range(chain_len-1,burn_in_itr,-1):
    #     if (best_chain[itr]['num_clusters_']==inferred_k):
    #         best_itr = itr
    #         break;

    # get the max ll itr in the best chain for k=inferred_k
    # we use this for getting all parameters instead of mote carlo average TODO
    post_ll = [(itr,best_chain[itr]['ll']) for itr in all_post_idx if best_chain[itr]['num_clusters_']==inferred_k]
    index, value = max(post_ll, key=operator.itemgetter(1))
    best_itr = index

    # plot latest best cluster sizes
    K = inferred_k
    data = sample_data['E_train']
    # data = exp_result['E_train']
    cids = best_chain[best_itr]['cluster_ids_']
    assignment = best_chain[best_itr]['assignment']

    colors = get_N_HexCol(K)
    colors = np.asarray(colors)
    col = np.zeros([data.shape[0],3])
    idx = np.array(assignment)
    i=0
    for cid in cids:
        col[(idx==cid)] = colors[i]
        i += 1

    cluster_list = [[cluster_id, best_chain[best_itr]['suffstats'][cluster_id].N_k] for cluster_id in best_chain[best_itr]['cluster_ids_']]
    idx, counts = zip(*cluster_list)
    plt.figure()
    plt.title('Cluster sizes in the best itr with k=%d' %inferred_k)
    plt.bar(idx,counts,color=colors/255.)


    # #pykdl stuff
    # robot = Robot.from_xml_string(f.read())
    # base_link = robot.get_root()
    # end_link = 'left_tool0'
    # kdl_kin = KDLKinematics(robot, base_link, end_link)

    dP = params['mjc_exp_params']['dP']
    dV = params['mjc_exp_params']['dV']
    dU = params['mjc_exp_params']['dU']

    # p = data[:,:dP]
    # # v = data[:,dP:]
    # # u = data[:,2]
    #
    # x_gripper = []
    # for i in range(data.shape[0]):
    #     q = p[i]
    #     gripper_T = kdl_kin.forward(q, end_link=end_link, base_link=base_link)
    #     epos = np.array(gripper_T[:3,3])
    #     epos = epos.reshape(-1)
    #     erot = np.array(gripper_T[:3,:3])
    #     erot = euler_from_matrix(erot)
    #     x = np.append(epos,erot)
    #     x_gripper.append(x)
    #
    # x_gripper = np.array(x_gripper)
    #
    # px = x_gripper[:,0]
    # py = x_gripper[:,1]
    # pz = x_gripper[:,2]


    # XU = np.zeros((N,T,dX+dU))
    # for n in range(N):
    #     XU[n] = np.concatenate((X[n,:,:],U[n,:,:]),axis=1)
    # XU_t = XU[:,:-1,:]
    # X_t1 = XU[:,1:,:dX]
    # X_t = XU[:,:-1,:dX]
    # delX = X_t1 - X_t

    px = data[:,0]
    py = data[:,1]
    pz = data[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter3D(px, py, pz, c = col/255.0)

plt.show()
