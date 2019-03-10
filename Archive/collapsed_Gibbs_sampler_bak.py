import pandas as pd
import numpy as np
from collections import namedtuple
from numpy import random
from Archive.utilities import multivariate_t_distribution as mstd
import matplotlib.pyplot as plt


SuffStat = namedtuple('SuffStat', 'x_bar sigma_bar N_k m k v S N', )
Theta = namedtuple('Theta', 'mean var', )


def initial_state(init_state, data, rand_clust_size, seq_init):
    # alpha = init_state['hyperparameters_']['alpha_']
    state = {}
    state.update(init_state)
    X = np.array(data)
    N = X.shape[0]
    X = X.reshape(N,-1)
    D = X.shape[1]
    m0 = x_bar = np.reshape(np.mean(X, axis=0),(D,1)) # dataset mean
    v0 = D+2
    Xc = X.T - x_bar
    # print 'Xc',Xc
    S_x_bar = Xc.dot(Xc.T)
    # print 'S_x_bar',S_x_bar
    S0 = np.diag(np.diag(S_x_bar)/N) # it says S0 = diag(S_x_bar)/N, assuming it means a diagonal matrix
    # print 'S0',S0
    state['hyperparameters_']['m0'] = m0
    # state['hyperparameters_']['k0'] = k0
    state['hyperparameters_']['v0'] = v0
    state['hyperparameters_']['S0'] = S0
    if seq_init:
        state['num_clusters_'] = 1
        state['assignment'] = [-1 for _ in data] # no initial assignment, use initial K =1
        state['cluster_ids_'] = cluster_ids = range(state['num_clusters_'])
    else:
        if rand_clust_size:
            state['num_clusters_'] = random.randint(1,10)
        state['cluster_ids_'] = cluster_ids = range(state['num_clusters_'])
        state['assignment'] = [random.choice(cluster_ids) for _ in data] # initial random assignments
    state['suffstats'] = {cid: None for cid in cluster_ids} # sufficient statistics clusters
    state['data_'] = data
    state['N'] = N # size of data
    state['D'] = D # dimension of observation
    state['pi'] = None
    state['theta'] = {cid: None for cid in cluster_ids}

    # for cluster_id, N in Counter(state['assignment']).iteritems():
    #     update_suffstats(state,cluster_id)
    for cluster_id in state['cluster_ids_']:
        init_suffstats(state,cluster_id)
    return state

def init_suffstats(state,cluster_id):
    m0 = state['hyperparameters_']['m0']
    k0 = state['hyperparameters_']['k0']
    v0 = state['hyperparameters_']['v0']
    S0 = state['hyperparameters_']['S0']
    N0 = state['N']
    # print 'hello'
    x_k = [x
        for x, cid in zip(state['data_'], state['assignment'])
        if cid == cluster_id
    ]
    x_k = np.array(x_k)
    # print 'x_k',x_k.shape
    N_k = x_k.shape[0]
    assert(N_k>=0)
    if (N_k > 0):
        D_k = x_k.shape[1]
        x_bar_k = np.reshape(x_k.mean(axis=0),(D_k,1))
        x_kc = x_k - x_bar_k.T
        sigma_bar_k = (1./N_k)*x_kc.T.dot(x_kc)
        kN_k = k0 + N_k
        vN_k = v0 + N_k
        mN_k = (k0*m0 + N_k*x_bar_k)/kN_k
        S_k = x_k.T.dot(x_k)
        SN_k = S0 + S_k + k0*np.outer(m0,m0) - kN_k*np.outer(mN_k,mN_k)
        # print 'SN_k',SN_k
        # SN_k_1 = S0 + x_kc.T.dot(x_kc) + k0*N_k/(k0+N_k)*np.outer((x_bar_k-m0),(x_bar_k-m0))
        # print 'SN_k_1',SN_k_1
        state['suffstats'][cluster_id] = SuffStat(x_bar_k, sigma_bar_k, N_k, mN_k, kN_k, vN_k, SN_k, N0)
    else:
        state['suffstats'][cluster_id] = SuffStat(0, 0, 0, m0, k0, v0, S0, N0)


# def update_suffstats(state,cluster_id,data_id):
#     m0 = state['hyperparameters_']['m0']
#     k0 = state['hyperparameters_']['k0']
#     v0 = state['hyperparameters_']['v0']
#     S0 = state['hyperparameters_']['S0']
#     N0 = state['N']
#     # print 'hello'
#
#     x_bar_k_p
#
#     x_k = np.array(x_k)
#     # print 'x_k',x_k.shape
#     N_k = x_k.shape[0]
#     assert(N_k>=0)
#     if (N_k > 0):
#         D_k = x_k.shape[1]
#         x_bar_k = np.reshape(x_k.mean(axis=0),(D_k,1))
#         x_kc = x_k - x_bar_k.T
#         sigma_bar_k = (1./N_k)*x_kc.T.dot(x_kc)
#         kN_k = k0 + N_k
#         vN_k = v0 + N_k
#         mN_k = (k0*m0 + N_k*x_bar_k)/kN_k
#         S_k = x_k.T.dot(x_k)
#         SN_k = S0 + S_k + k0*np.outer(m0,m0) - kN_k*np.outer(mN_k,mN_k)
#         # print 'SN_k',SN_k
#         # SN_k_1 = S0 + x_kc.T.dot(x_kc) + k0*N_k/(k0+N_k)*np.outer((x_bar_k-m0),(x_bar_k-m0))
#         # print 'SN_k_1',SN_k_1
#         state['suffstats'][cluster_id] = SuffStat(x_bar_k, sigma_bar_k, N_k, mN_k, kN_k, vN_k, SN_k, N0)
#     else:
#         state['suffstats'][cluster_id] = SuffStat(0, 0, 0, m0, k0, v0, S0, N0)

def log_predictive_likelihood(data_id, cluster_id, state):
    """Predictive likelihood of the data at data_id is generated
    by cluster_id given the currenbt state.
    """
    D = state['D']
    x = np.array(state['data_'][data_id])
    x = x.reshape(D,1)
    ss = state['suffstats'][cluster_id]
    return _log_predictive_likelihood(ss, x, D)

def _log_predictive_likelihood(ss, x, D):
    mu = ss.m
    Sigma = (ss.k+1)/(ss.k*(ss.v-D+1))*ss.S
    v = ss.v-D+1
    log_p = np.log(mstd(x,mu,Sigma,v,D))
    return log_p

def log_predictive_likelihood_dp(data_id, cluster_id, state):
    """Predictive likelihood of the data at data_id is generated
    by cluster_id given the currenbt state.
    """
    x = state['data_'][data_id]
    D = state['D']
    x = x.reshape(D,1)
    if cluster_id == "new":
        m0 = state['hyperparameters_']['m0']
        k0 = state['hyperparameters_']['k0']
        v0 = state['hyperparameters_']['v0']
        S0 = state['hyperparameters_']['S0']
        N0 = state['data_'].shape[0]
        ss = SuffStat(0, 0, 0, m0, k0, v0, S0, N0)
    else:
        ss = state['suffstats'][cluster_id]
    return _log_predictive_likelihood(ss, x, D)

def log_cluster_assign_score(cluster_id, state):
    """Log-likelihood that a new point generated will
    be assigned to cluster_id given the current state.
    """
    current_cluster_size = state['suffstats'][cluster_id].N_k
    num_clusters = state['num_clusters_']
    alpha = state['hyperparameters_']['alpha_']
    # denomenator missing. This is okay because it is taken care of in the
    # subsequent normalization step
    return np.log(current_cluster_size + alpha * 1. / num_clusters)

def log_cluster_assign_score_dp(cluster_id, state):
    """Log-likelihood that a new point generated will
    be assigned to cluster_id given the current state.
    """
    # denomenator missing. This is okay because it is taken care of in the
    # subsequent normalization step
    if (cluster_id == "new") or (state['suffstats'][cluster_id].N_k == 0):
        return np.log(state['hyperparameters_']["alpha_"])
    else:
        return np.log(state['suffstats'][cluster_id].N_k)

def cluster_assignment_distribution(data_id, state):
    """Compute the marginal distribution of cluster assignment
    for each cluster.
    """
    scores = {}
    for cid in state['suffstats'].keys():
        log_pred_ll = log_predictive_likelihood(data_id, cid, state)
        # print log_pred_ll
        scores[cid] = log_pred_ll
        log_clust_ll = log_cluster_assign_score(cid, state)
        # print log_clust_ll
        scores[cid] += log_clust_ll
        # raw_input()
    scores = {cid: np.exp(score) for cid, score in scores.iteritems()}
    normalization = 1.0/sum(scores.values())
    scores = {cid: score*normalization for cid, score in scores.iteritems()}
    return scores

def cluster_assignment_distribution_dp(data_id, state):
    """Compute the marginal distribution of cluster assignment
    for each cluster.
    """
    scores = {}
    cluster_ids = state['suffstats'].keys() + ['new']
    for cid in cluster_ids:
        score_ll = log_predictive_likelihood_dp(data_id, cid, state)
        # print 'score_ll',score_ll
        scores[cid] = score_ll
        score_clust = log_cluster_assign_score_dp(cid, state)
        # print 'score_clust',score_clust
        scores[cid] += score_clust
        # raw_input()
    scores = {cid: np.exp(score) for cid, score in scores.iteritems()}
    normalization = 1.0/sum(scores.values())
    scores = {cid: score*normalization for cid, score in scores.iteritems()}
    return scores

def add_datapoint_to_suffstats(data_id, cluster_id, state):
    """Add datapoint to cluster
    """
    assert(state['assignment'][data_id] != cluster_id)
    state['assignment'][data_id] = cluster_id
    init_suffstats(state,cluster_id)

SuffStat = namedtuple('SuffStat', 'x_bar sigma_bar N_k m k v S N', )

def remove_datapoint_from_suffstats(data_id, cluster_id, state):
    """Remove datapoint from the cluster
    """
    assert(state['assignment'][data_id] == cluster_id)
    # ss = state['suffstats'][cluster_id]
    # assert(ss.N_k > 0)
    state['assignment'][data_id] = -1 # -1 to ingnore the data
    init_suffstats(state,cluster_id)

    # N0 = state['N']
    # D = state['D']
    # m0 = state['hyperparameters_']['m0']
    # k0 = state['hyperparameters_']['k0']
    # v0 = state['hyperparameters_']['v0']
    # S0 = state['hyperparameters_']['S0']
    #
    # x_bar_k_o = ss.
    # xi = np.asarray(state['data_'][data_id],(D,1))
    #
    # # print 'hello'
    #
    # x_bar_k_o = ss.x_bar
    # sigma_bar_k_o = ss.sigma_bar
    # N_k_o = ss.N_k
    # N_k_n = N_k_o - 1
    # assert(x_bar_k_o.shape == xi.shape)
    # x_bar_k_n = (x_bar_k_o * N_k_o - xi)/(1.0 * N_k_n)
    # xixi = np.outer(xi-x_bar_k_o,xi-x_bar_k_o)
    # sigma_bar_k_n = (sigma_bar_k_o * N_k_o - xixi)/(1.0 * N_k_n)
    #
    #
    # ss.mN_k
    # ss.kN_k
    # ss.vN_k
    # ss.SN_k
    #
    # if (N_k > 0):
    #     D_k = D
    #     N_k = N_k_n
    #     x_bar_k = x_bar_k_n
    #     sigma_bar_k = sigma_bar_k_n
    #     kN_k = k0 + N_k_n
    #     vN_k = v0 + N_k_n
    #     mN_k = (k0*m0 + N_k*x_bar_k)/kN_k
    #     S_k = x_k.T.dot(x_k)
    #     SN_k = S0 + S_k + k0*np.outer(m0,m0) - kN_k*np.outer(mN_k,mN_k)
    #     # print 'SN_k',SN_k
    #     # SN_k_1 = S0 + x_kc.T.dot(x_kc) + k0*N_k/(k0+N_k)*np.outer((x_bar_k-m0),(x_bar_k-m0))
    #     # print 'SN_k_1',SN_k_1
    #     state['suffstats'][cluster_id] = SuffStat(x_bar_k, sigma_bar_k, N_k, mN_k, kN_k, vN_k, SN_k, N0)
    # else:
    #     state['suffstats'][cluster_id] = SuffStat(0, 0, 0, m0, k0, v0, S0, N0)

def create_cluster(state):
    state["num_clusters_"] += 1
    cluster_id = max(state['suffstats'].keys()) + 1
    state['cluster_ids_'].append(cluster_id)
    init_suffstats(state,cluster_id)
    # print "Created cluster:", cluster_id
    return cluster_id

def destroy_cluster(state, cluster_id):
    state["num_clusters_"] -= 1 # Bug! changed = to -= TODO
    del state['suffstats'][cluster_id]
    del state['theta'][cluster_id]
    state['cluster_ids_'].remove(cluster_id)
    # print "Destroyed cluster:", cluster_id

def prune_clusters(state):
    for cid in state['cluster_ids_']:
        if state['suffstats'][cid].N_k == 0:
            destroy_cluster(state, cid)

def sample_cluster_thetas(state):
    D = state['D']
    for cid in state['cluster_ids_']:
        ss = state['suffstats'][cid]
        mu = ss.m
        Sigma = ss.S/(ss.k*(ss.v-D+1))
        v = ss.v-D+1
        # # actual sample from the conditional posterior
        # # we don't need this for collapsed Gibbs sampling
        # mean = gen_T_dist(mu, Sigma, df=v, n=1)
        # var = stats.invwishart.rvs(ss.v,ss.S,size=1)

        # mean of cluster mean
        mean = mu
        # mean of cluster variance
        var = ss.S/(ss.v - D - 1)

        state['theta'][cid] = Theta(mean,var)

def sample_cluster_weights(state):
    ss = state['suffstats']
    alpha_k = [ss[cid].N_k + state['hyperparameters_']['alpha_'] / state['num_clusters_'] for cid in state['cluster_ids_']]
    # # sample form the conditional posterior
    # # no need for this in collapsed Gibbs
    # state['pi'] = stats.dirichlet.rvs(alpha_k,size=1).flatten()

    # expected cluster weights
    alpha_0 = np.sum(alpha_k)
    alpha = alpha_k/alpha_0
    state['pi'] = alpha

def gibbs_step(state):
    # sample indicator variables
    data_id = range(state['N'])
    pairs = zip(data_id, state['data_'], state['assignment'])
    random.shuffle(pairs)
    for _, (data_id, datapoint, cid) in enumerate(pairs):
        remove_datapoint_from_suffstats(data_id,cid,state)
        scores = cluster_assignment_distribution(data_id, state).items()
        labels, scores = zip(*scores)
        cid = random.choice(labels, p=scores)
        add_datapoint_to_suffstats(data_id,cid,state)

    # sample cluster weights
    sample_cluster_weights(state)

    # sample cluster thetas (means and covariances)
    sample_cluster_thetas(state)

def sample_assignment(data_id, state):
    """Sample new assignment from marginal distribution.
    If cluster is "`new`", create a new cluster.
    """
    scores = cluster_assignment_distribution_dp(data_id, state).items()
    labels, scores = zip(*scores)
    cid = random.choice(labels, p=scores)
    if cid == "new":
        return create_cluster(state)
    else:
        return int(cid)

def init_gibbs_step_dp(state):
    """Collapsed Gibbs sampler for Dirichlet Process Mixture Model
    """
    # sample indicator variables
    pairs = zip(state['data_'], state['assignment'])
    for data_id, (datapoint, cid) in enumerate(pairs):
        cid = sample_assignment(data_id, state)
        add_datapoint_to_suffstats(data_id,cid,state)
    # sample cluster weights
    sample_cluster_weights(state)

    # sample cluster thetas (means and covariances)
    sample_cluster_thetas(state)

def gibbs_step_dp(state):
    """Collapsed Gibbs sampler for Dirichlet Process Mixture Model
    """
    # sample indicator variables
    data_id = range(state['N'])
    pairs = zip(data_id, state['data_'], state['assignment'])
    random.shuffle(pairs)
    for _, (data_id, datapoint, cid) in enumerate(pairs):
        remove_datapoint_from_suffstats(data_id,cid,state)
        prune_clusters(state)
        cid = sample_assignment(data_id, state)
        # state['assignment'][data_id] = cid
        add_datapoint_to_suffstats(data_id,cid,state)

    # sample cluster weights
    sample_cluster_weights(state)

    # sample cluster thetas (means and covariances)
    sample_cluster_thetas(state)

def plot_clusters(state):
    gby = pd.DataFrame({
            'data': state['data_'],
            'assignment': state['assignment']}
        ).groupby(by='assignment')['data']
    hist_data = [gby.get_group(cid).tolist()
                 for cid in gby.groups.keys()]
    plt.hist(hist_data,
             bins=20,
             histtype='stepfilled', alpha=.5 )
