import pandas as pd
import numpy as np
from collections import namedtuple, Counter
# from scipy import stats
from numpy import random
# from utilities import multivariate_t_distribution as mstd
import matplotlib.pyplot as plt


SuffStat = namedtuple('SuffStat', 'x_bar sigma_bar N_k m k v S N', )


def initial_state(init_state, data):
    # alpha = init_state['hyperparameters_']['alpha_']
    state = {}
    state.update(init_state)
    X = np.array(data)
    N = X.shape[0]
    X = X.reshape(N,-1)
    D = X.shape[1]
    m0 = x_bar = np.mean(X, axis=0).T # dataset mean
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
    state['cluster_ids_'] = cluster_ids = range(init_state['num_clusters_'])
    state['suffstats'] = {cid: None for cid in cluster_ids} # sufficient statistics clusters
    state['assignment'] = [random.choice(cluster_ids) for _ in data] # initial random assignments
    # state['pi'] = {cid: alpha / num_clusters for cid in cluster_ids}, # alpha_k for symmetric DP or Dirichlet
    state['data_'] = data
    state['N'] = N # size of data
    state['D'] = D # dimension of observation

    for cluster_id, N in Counter(state['assignment']).iteritems():
        update_suffstats(state,cluster_id)
    return state

def update_suffstats(state,cluster_id):
    m0 = state['hyperparameters_']['m0']
    k0 = state['hyperparameters_']['k0']
    v0 = state['hyperparameters_']['v0']
    S0 = state['hyperparameters_']['S0']
    N0 = state['N']

    x_k = [x
        for x, cid in zip(state['data_'], state['assignment'])
        if cid == cluster_id
    ]
    x_k = np.array(x_k)
    N_k = x_k.shape[0]
    assert(N_k>=0)
    if (N_k > 0):
        x_bar_k = x_k.mean(axis=0).T
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

def log_predictive_likelihood(data_id, cluster_id, state):
    """Predictive likelihood of the data at data_id is generated
    by cluster_id given the currenbt state.
    """
    D = state['D']
    x = np.array(state['data_'][data_id])
    ss = state['suffstats'][cluster_id]
    return _log_predictive_likelihood(ss, x, D)

def _log_predictive_likelihood(ss, x, D):
    mu = ss.m
    Sigma = (ss.k+1)/(ss.k*(ss.v-D+1))*ss.S
    v = ss.v-D+1
    log_p = np.log(mstd(x,mu,Sigma,v,D))
    # print mu, Sigma, v, D, log_p
    return log_p

def log_predictive_likelihood_dp(data_id, cluster_id, state):
    """Predictive likelihood of the data at data_id is generated
    by cluster_id given the currenbt state.
    """
    x = state['data_'][data_id]
    D = state['D']
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
    if cluster_id == "new":
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
        scores[cid] = log_predictive_likelihood_dp(data_id, cid, state)
        scores[cid] += log_cluster_assign_score_dp(cid, state)
    scores = {cid: np.exp(score) for cid, score in scores.iteritems()}
    normalization = 1.0/sum(scores.values())
    scores = {cid: score*normalization for cid, score in scores.iteritems()}
    return scores

def add_datapoint_to_suffstats(data_id, cluster_id, state):
    """Add datapoint to cluster
    """
    state['assignment'][data_id] = cluster_id
    update_suffstats(state,cluster_id)


def remove_datapoint_from_suffstats(data_id, cluster_id, state):
    """Remove datapoint from the cluster
    """
    state['assignment'][data_id] = -1 # -1 to ingnore the data
    update_suffstats(state,cluster_id)

def create_cluster(state):
    state["num_clusters_"] += 1
    cluster_id = max(state['suffstats'].keys()) + 1
    state['cluster_ids_'].append(cluster_id)
    update_suffstats(state,cluster_id)
    print "Created cluster:", cluster_id
    return cluster_id

def destroy_cluster(state, cluster_id):
    state["num_clusters_"] -= 1 # Bug! changed = to -= TODO
    del state['suffstats'][cluster_id]
    state['cluster_ids_'].remove(cluster_id)
    print "Destroyed cluster:", cluster_id

def prune_clusters(state):
    for cid in state['cluster_ids_']:
        if state['suffstats'][cid].N_k == 0:
            destroy_cluster(state, cid)

def gibbs_step(state):
    pairs = zip(state['data_'], state['assignment'])
    for data_id, (datapoint, cid) in enumerate(pairs):
        remove_datapoint_from_suffstats(data_id,cid,state)
        scores = cluster_assignment_distribution(data_id, state).items()
        # print scores
        # raw_input()
        labels, scores = zip(*scores)
        cid = random.choice(labels, p=scores)
        # print cid
        # raw_input()
        # state['assignment'][data_id] = cid
        add_datapoint_to_suffstats(data_id,cid,state)

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

def gibbs_step_dp(state):
    """Collapsed Gibbs sampler for Dirichlet Process Mixture Model
    """
    pairs = zip(state['data_'], state['assignment'])
    for data_id, (datapoint, cid) in enumerate(pairs):
        remove_datapoint_from_suffstats(data_id,cid,state)
        prune_clusters(state)
        cid = sample_assignment(data_id, state)
        # state['assignment'][data_id] = cid
        add_datapoint_to_suffstats(data_id,cid,state)

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
