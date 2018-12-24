import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as color_map
from mpl_toolkits.mplot3d import axes3d
import pickle
from mixture_model_gibbs_sampling import ACF
from collections import Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn import mixture
from collections import Counter
from utilities import *
from copy import deepcopy
from collapsed_Gibbs_sampler import predictive_ll_cluster
import operator
from gmm import GMM
from collections import namedtuple, Counter

f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')
train_data_id = 1

task = 'GMM'
# task = 'visual'
# task = 'compute'

if task=='GMM':
    sample_data = pickle.load( open( "mjc_floor_contact_processed.p", "rb" ) )
    mjc_exp_params = sample_data['exp_params']
    dP = mjc_exp_params['dP']
    dV = mjc_exp_params['dV']
    dU = mjc_exp_params['dU']

    data = sample_data['clust_data']
    EXt = sample_data['EXt']
    px = EXt[:,0]
    py = EXt[:,1]
    pz = EXt[:,2]
    restarts = 50
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
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter3D(px, py, pz, c = col/255.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # gmm = mixture.GaussianMixture(n_components=K,
                                    # covariance_type='full',
                                    # tol=1e-6, n_init=restarts,
                                    # warm_start=False)
    # gmm.fit(data)
    # idx = gmm.predict(data)

    # lowest_bic = np.infty
    # bic = []
    # n_components_range = range(1, 7)
    # for n_components in n_components_range:
    #     # Fit a Gaussian mixture with EM
    #     gmm = mixture.GaussianMixture(  n_components=K,
    #                                     covariance_type='full',
    #                                     tol=1e-6, n_init=restarts,
    #                                     warm_start=False,
    #                                     max_iter=1000)
    #     gmm.fit(data)
    #     bic.append(gmm.bic(data))
    #     if bic[-1] < lowest_bic:
    #         lowest_bic = bic[-1]
    #         best_gmm = gmm
    # idx = best_gmm.predict(data)
    # K = best_gmm.weights_.shape[0]
    # print 'Best BIC K', K
    #
    # colors = get_N_HexCol(K)
    # colors = np.asarray(colors)
    # col = np.zeros([data.shape[0],3])
    # # idx = np.array(assignment)
    # for k in range(K):
    #     col[(idx==k)] = colors[k]
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.scatter3D(px, py, pz, c = col/255.0)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

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
    # sample_data = pickle.load( open( "mjc_floor_contact_processed.p", "rb" ) )
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

    # get the max ll itr in the best chain for k=inferred_k
    # we use this for getting all parameters instead of mote carlo average TODO
    post_ll = [(itr,best_chain[itr]['ll']) for itr in all_post_idx if best_chain[itr]['num_clusters_']==inferred_k]
    index, value = max(post_ll, key=operator.itemgetter(1))
    best_itr = index

    # plot latest best cluster sizes
    K = inferred_k
    # data = sample_data['clust_data']
    data = exp_result['clust_data']
    EXt = exp_result['EXt']
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

    # dP = params['mjc_exp_params']['dP']
    # dV = params['mjc_exp_params']['dV']
    # dU = params['mjc_exp_params']['dU']

    px = EXt[:,0]
    py = EXt[:,1]
    pz = EXt[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter3D(px, py, pz, c = col/255.0)

plt.show()
