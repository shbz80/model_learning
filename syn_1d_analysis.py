import numpy as np
import matplotlib.pyplot as plt
from utilities import plot_ellipse
from utilities import get_N_HexCol
import pickle
from collections import Counter
from mixture_model_gibbs_sampling import ACF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from collections import Counter
from utilities import conditional_Gaussian_mixture
from utilities import estep
from copy import deepcopy
from collapsed_Gibbs_sampler import predictive_ll_cluster
import operator

# exp_result = pickle.load( open( "syn_1d_finite_gauss_clustered.p", "rb" ) )
# alpha=4 and k=2
exp_result = pickle.load( open( "syn_1d_infinite_gauss_clustered_8.p", "rb" ) )
exp_stats = exp_result['exp_stats']
params = exp_result['params']

xr = exp_result['xr']
yr = exp_result['yr']
data = exp_result['data']
X, Y = data[:,0], data[:,1]
X = np.reshape(X,(-1,1))
Y = np.reshape(Y,(-1,1))

# plots
fig, ax = plt.subplots()
plt.title('Data')
plt.scatter(X,Y)
plt.plot(xr,yr,color='r')

num_chains = len(exp_stats)

task = 'visual'
# task = 'compute'

if task=='visual':
    # plots log likelihood
    # plt.figure()
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
    # chose best chain from the above two plots
    # # original 4 cluster data
    # best_chain_id = 0 # syn_1d_infinite_gauss_clustered_2.p
    # mix_itr = 20 # syn_1d_infinite_gauss_clustered_2.p
    #
    ## with alpha 1
    # # 5 clusters
    # best_chain_id = 4 # syn_1d_infinite_gauss_clustered_3.p
    # mix_itr = 80 # syn_1d_infinite_gauss_clustered_3.p

    # 3 clusters
    # best_chain_id = 2 # syn_1d_infinite_gauss_clustered_4.p
    # mix_itr = 30 # syn_1d_infinite_gauss_clustered_4.p

    # # with alpha 4
    # # 2 clusters, syn_1d_infinite_gauss_clustered_6.p
    # best_chain_id = 1
    # mix_itr = 20

    # # 5 clusters, syn_1d_infinite_gauss_clustered_7.p
    # best_chain_id = 2
    # mix_itr = 10

    # 8 clusters, syn_1d_infinite_gauss_clustered_8.p
    # best_chain_id = 3
    # mix_itr = 900

    best_chain_id = 0
    mix_itr = 20

    best_chain = exp_stats[best_chain_id]
    chain_len = len(best_chain)
    burn_in_itr = mix_itr * 10 # for syn_1d_infinite_gauss_clustered_8.p
    # burn_in_itr = mix_itr * 10
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
    post_k_samples = [best_chain[itr]['num_clusters_'] for itr in post_samples_idx]
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
    post_ll = [(itr,best_chain[itr]['ll']) for itr in post_samples_idx if best_chain[itr]['num_clusters_']==inferred_k]
    index, value = max(post_ll, key=operator.itemgetter(1))
    best_itr = index

    # plot latest best cluster sizes
    K = inferred_k
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
    plt.title('Cluster sizes in the latest itr with k=%d' %inferred_k)
    plt.bar(idx,counts,color=colors/255.)

    # plot clustering result

    cluster_means = [best_chain[best_itr]['theta'][cid].mean for cid in best_chain[best_itr]['cluster_ids_']]
    cluster_means = np.asarray(cluster_means)
    cluster_means = cluster_means.reshape(-1,2)
    cluster_vars = [best_chain[best_itr]['theta'][cid].var for cid in best_chain[best_itr]['cluster_ids_']]
    cluster_vars = np.asarray(cluster_vars)
    cluster_weights = best_chain[best_itr]['pi']
    cluster_weights = cluster_weights.reshape(-1,1)
    fig, ax = plt.subplots()
    plt.title('Clustered with DPMM collapsed Gibbs')
    plt.scatter(X,Y,c=col/255.0)
    plt.scatter(cluster_means[:,0],cluster_means[:,1])      # Cluster means
    for k in range(K):
        plot_ellipse(ax, cluster_means[k], cluster_vars[k], color=colors[k]/255.0) # cluster ellipses
    # plt.plot(xr,yr,color='r')

    # gp prediction
    gp_params = {
                    'alpha': 0., # use this when using white kernal
                    'K_C': C(1.0, (1e-2, 1e2)),
                    'K_RBF': RBF(1, (1e-3, 1e3)),
                    # 'K_RBF': RBF(1, (1e-3, 1e0)),
                    'K_W': W(noise_level=1., noise_level_bounds=(1e-1, 1e1)),
                    # 'K_W': W(noise_level=1., noise_level_bounds=(1e0, 5e0)),
                    'normalize_y': True,
                    'restarts': 10,
                    }

    alpha = gp_params['alpha']
    K_C = gp_params['K_C']
    K_RBF = gp_params['K_RBF']
    K_W = gp_params['K_W']
    normalize_y = gp_params['normalize_y']
    restarts = gp_params['restarts']
    # kernel = K_C + K_RBF + K_W
    kernel = K_RBF + K_W

    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, Y)
    Nsec = params['discon_params']['Nsec']
    T = Nsec * 1. # assumimg each segment is 1 sec
    N = X.shape[0]
    xt = np.linspace(0.,T,N)
    xt = xt.reshape(-1,1)
    yt_mean, yt_cov = gp.predict(xt, return_cov=True)
    xt = np.reshape(xt,(-1))
    yt_mean = np.reshape(yt_mean,(-1))
    yt_std = np.reshape(np.sqrt(np.diag(yt_cov)),(-1))
    plt.figure()
    plt.plot(xt,yt_mean,color='b')
    plt.fill_between(xt, yt_mean - yt_std, yt_mean + yt_std, alpha=0.2, color='b')
    plt.plot(xr,yr,color='r')
    plt.title('GPR without clustering')

    mu = cluster_means
    sigma = cluster_vars
    xt = xt.reshape(-1,1)
    dx = 1
    dxy = 2
    mu_y_x = conditional_Gaussian_mixture(mu, sigma, xt, dx,dxy)
    y_x = np.zeros((N,1))
    clsidx = np.zeros(N,dtype=int)
    clsidx_dp = np.zeros(N,dtype=int)
    for n in range(N):
        x = np.tile(xt[n],(K,1))
        y = mu_y_x[n].reshape(K,1)
        xy = np.c_[x,y]
        # Compute probability of each point under each cluster.
        logobs = estep(xy,mu,sigma,cluster_weights)
        pairs = list(zip(best_chain[best_itr]['cluster_ids_'], xy))
        scores = predictive_ll_cluster(best_chain[best_itr],pairs)
        scores = np.array(scores)
        clsidx_dp[n] = scores[np.argmax(scores[:,1])][0]
        # Renormalize to get cluster weights.
        idx = np.argmax(logobs)
        r = idx//K
        c = idx%K
        clsidx[n] = best_chain[best_itr]['cluster_ids_'][c]
        y_x[n] = mu_y_x[n][c]

    # mixture of GP experts - independent gating and expert learning
    idx = np.array(best_chain[best_itr]['assignment'])
    MGP = {}
    for cid in best_chain[best_itr]['cluster_ids_']:
        xk = X[(idx==cid)]
        yk = Y[(idx==cid)]
        alpha = gp_params['alpha']
        K_C = gp_params['K_C']
        K_RBF = gp_params['K_RBF']
        K_W = gp_params['K_W']
        normalize_y = gp_params['normalize_y']
        restarts = gp_params['restarts']
        # kernel = K_C + K_RBF + K_W
        kernel = K_RBF + K_W
        # gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(xk, yk)
        # MGP.append(deepcopy(gp))
        MGP[cid] = deepcopy(gp)
        del gp

    N = X.shape[0]
    yn_mean = np.zeros(N)
    yn_cov = np.zeros(N)
    for n in range(N):
        # z = clsidx[n]
        z = clsidx_dp[n]
        yn_mean[n], yn_cov[n] = MGP[z].predict(X[n].reshape(1,1),return_cov=True)
    assert(yn_cov.shape==(N,)) # make sure it is single point gp prediction
    yn_std = np.sqrt(yn_cov)

    plt.figure()
    xt = X.reshape(N)
    plt.plot(xt,yn_mean,color='b')        # predictions from mixture model
    plt.fill_between(xt, yn_mean - yn_std, yn_mean + yn_std, alpha=0.2, color='b')
    # plt.plot(xn,y_mean)       # predictions from GP
    plt.plot(xr,yr,color='r')   # Ground truth function
    plt.title('Prediction with DPMM model')
    plt.show()
