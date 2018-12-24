import pandas as pd
import numpy as np
from collections import namedtuple, Counter
from scipy import stats
from numpy import random
# from utilities import log_multivariate_t_distribution as log_mstd
# from utilities import multivariate_t_rvs as gen_T_dist
import copy
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.linalg import cholesky, cho_solve, solve_triangular

def initial_state_gp(init_state, data, rand_clust_size, seq_init):
    # alpha = init_state['hyperparameters_']['alpha_']
    state = {}
    state.update(init_state)
    X = np.array(data)
    N = X.shape[0]
    X = X.reshape(N,-1)
    D = X.shape[1]

    if seq_init:
        state['num_clusters_'] = 1
        state['assignment'] = [-1 for _ in data] # no initial assignment, use initial K =1
        state['cluster_ids_'] = cluster_ids = range(state['num_clusters_'])
    else:
        if rand_clust_size:
            state['num_clusters_'] = random.randint(1,10)
        state['cluster_ids_'] = cluster_ids = range(state['num_clusters_'])
        state['assignment'] = [random.choice(cluster_ids) for _ in data] # initial random assignments
    state['gpstats'] = {cid: {} for cid in cluster_ids}
    state['data_'] = np.array(data)
    state['N'] = N # size of data
    state['D'] = D # dimension of observation
    state['ll'] = np.inf

    # GP stuff
    dX = state['dX']
    dY = state['dY']
    Xd = data[:,:dX]
    Yd = data[:,dX:]
    state['hyperparameters_']['gp_params']['signal_variance'] = np.var(Yd,axis=0)
    # for cluster_id, N in Counter(state['assignment']).iteritems():
    #     update_suffstats(state,cluster_id)
    for cluster_id in state['cluster_ids_']:
        init_gpstats(state,cluster_id)
    return state

def init_gpstats(state,cluster_id):
    N = state['N']
    dX = state['dX']
    dY = state['dY']
    gp_params = state['hyperparameters_']['gp_params']
    gp_params['K_RBF'] = RBF(np.ones(dX), (1e-3, 1e3)) # to use ARD

    data_id = range(state['N'])
    pairs = zip(data_id, state['data_'], state['assignment'])

    pairs_k = [(did, x)
        for did, x, cid in pairs
        if cid == cluster_id
    ]
    if (pairs_k):
        dids, data_k = zip(*pairs_k)
        dids = list(dids)
        data_k = np.array(data_k)
        N_k = data_k.shape[0]
        state['gpstats'][cluster_id]['dids'] = dids
        state['gpstats'][cluster_id]['gps'] = []
        state['gpstats'][cluster_id]['N_k'] = N_k
        update_nonempty_cluster_gpstat(state, cluster_id)
    else:
        state['gpstats'][cluster_id]['gps'] = []
        state['gpstats'][cluster_id]['dids'] = []
        state['gpstats'][cluster_id]['N_k'] = 0
    return

def update_nonempty_cluster_gpstat(state, cluster_id):
    N = state['N']
    dX = state['dX']
    dY = state['dY']

    dids = state['gpstats'][cluster_id]['dids']
    assert(dids)
    data_k = state['data_'][dids]
    X = data_k[:,:dX].reshape(-1,dX)
    Y = data_k[:,dX:].reshape(-1,dY)
    if(state['gpstats'][cluster_id]['gps']):
        del state['gpstats'][cluster_id]['gps']

    gp_params = state['hyperparameters_']['gp_params']
    K_RBF = gp_params['K_RBF']
    K_W = gp_params['K_W']
    normalize_y = gp_params['normalize_y']
    restarts = gp_params['restarts']
    kernel = K_RBF + K_W
    gps = []
    for n in range(dY):
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=restarts, normalize_y=normalize_y)
        gp.fit(X, Y[:,n])
        # print 'cid',cluster_id, 'gp %d fitted' %(n)
        gps.append(copy.deepcopy(gp))
        del gp
    state['gpstats'][cluster_id]['gps'] = gps
    return

def update_all_gpstats(state):
    for cluster_id in state['cluster_ids_']:
        update_nonempty_cluster_gpstat(state, cluster_id)
    return

def log_predictive_likelihood_gp(data_id, cluster_id, state):
    dX = state['dX']
    dY = state['dY']
    xy = state['data_'][data_id]
    D = state['D']
    x = xy[:dX]
    y = xy[dX:]

    dids = state['gpstats'][cluster_id]['dids']
    assert(data_id not in dids)
    data_k = state['data_'][dids]
    X = data_k[:,:dX].reshape(-1,dX)
    Y = data_k[:,dX:].reshape(-1,dY)
    assert(state['gpstats'][cluster_id]['gps'])
    gps = state['gpstats'][cluster_id]['gps']
    mu = np.zeros(dY)
    sigma = np.zeros(dY)
    for n in range(len(gps)): # for a GP in each output dim
        gp = gps[n]
        Yd = Y[:,n]
        Yd_mean = np.mean(Yd, axis=0)
        Y_d = Yd - Yd_mean
        kernel = gp.kernel_
        K = kernel(X)
        K[np.diag_indices_from(K)] += gp.alpha
        print K.shape
        L = cholesky(K, lower=True)
        alpha_ = cho_solve((L, True), Yd)
        K_trans = gp.kernel_(x, X)
        y_mean = K_trans.dot(alpha_)  # Line 4 (y_mean = f_star)
        mu[n] = Yd_mean + y_mean  # undo normal.
        v = cho_solve((L, True), K_trans.T)  # Line 5
        sigma[n] = kernel(x) - K_trans.dot(v)  # Line 6
    log_predictive_likelihood = _log_predictive_likelihood_gp(y, mu, sigma)
    return log_predictive_likelihood

def _log_predictive_likelihood_gp(x, mu, sigma):
    log_p = 0.
    for n in range(mu.shape[0]):
        log_p += stats.norm.logpdf(x, mu[n], sigma[n])
    return log_p

def log_predictive_likelihood_dp_gp(data_id, cluster_id, state):
    """Predictive likelihood of the data at data_id is generated
    by cluster_id given the currenbt state.
    """
    dX = state['dX']
    dY = state['dY']
    xy = state['data_'][data_id]
    D = state['D']
    x = xy[:dX]
    y = xy[dX:]

    if (cluster_id == "new"):
        mu = np.zeros(dY)
        v0 = state['hyperparameters_']['gp_params']['signal_variance']
        v1 = state['hyperparameters_']['gp_params']['noise_variance']
        sigma = v0 + v1
        assert(sigma.shape==(dY,))
    elif (state['gpstats'][cluster_id]['N_k']==0):
        mu = np.zeros(dY)
        v0 = state['hyperparameters_']['gp_params']['signal_variance']
        v1 = state['hyperparameters_']['gp_params']['noise_variance']
        sigma = v0 + v1
        assert(sigma.shape==(dY,))
    else:
        dids = state['gpstats'][cluster_id]['dids']
        assert(dids)
        assert(data_id not in dids)
        data_k = state['data_'][dids]
        X = data_k[:,:dX].reshape(-1,dX)
        Y = data_k[:,dX:].reshape(-1,dY)
        assert(state['gpstats'][cluster_id]['gps'])
        gps = state['gpstats'][cluster_id]['gps']
        mu = np.zeros(dY)
        sigma = np.zeros(dY)
        for n in range(len(gps)): # for a GP in each output dim
            gp = gps[n]
            Yd = Y[:,n]
            Yd_mean = np.mean(Yd, axis=0)
            Y_d = Yd - Yd_mean
            kernel = gp.kernel_
            K = kernel(X)
            K[np.diag_indices_from(K)] += gp.alpha
            L = cholesky(K, lower=True)
            alpha_ = cho_solve((L, True), Yd)
            K_trans = gp.kernel_(x, X)
            y_mean = K_trans.dot(alpha_)  # Line 4 (y_mean = f_star)
            mu[n] = Yd_mean + y_mean  # undo normal.
            v = cho_solve((L, True), K_trans.T)  # Line 5
            sigma[n] = kernel(x) - K_trans.dot(v)  # Line 6
    # print 'mu',mu
    # print 'sigma',sigma
    # print 'y',y
    log_predictive_likelihood = _log_predictive_likelihood_gp(y, mu, np.sqrt(sigma))
    return log_predictive_likelihood

def log_cluster_assign_score(cluster_id, state):
    """Log-likelihood that a new point generated will
    be assigned to cluster_id given the current state.
    """
    current_cluster_size = state['gpstats'][cluster_id]['N_k']
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
    if (cluster_id == "new") or (state['gpstats'][cluster_id]['N_k'] == 0):
        return np.log(state['hyperparameters_']["alpha_"])
    else:
        return np.log(state['gpstats'][cluster_id]['N_k'])

def cluster_assignment_distribution_gp(data_id, state):
    """Compute the marginal distribution of cluster assignment
    for each cluster.
    """
    scores = {}
    for cid in state['gpstats'].keys():
        log_pred_ll = log_predictive_likelihood_gp(data_id, cid, state)
        # print log_pred_ll
        scores[cid] = log_pred_ll
        log_clust_ll = log_cluster_assign_score(cid, state)
        # print log_clust_ll
        scores[cid] += log_clust_ll
        # raw_input()
    scores = {cid: np.exp(score) for cid, score in scores.iteritems()}
    normalization = 1.0/sum(scores.values())
    scores = {cid: np.asscalar(score*normalization) for cid, score in scores.iteritems()}
    return scores

def cluster_assignment_distribution_dp_gp(data_id, state):
    """Compute the marginal distribution of cluster assignment
    for each cluster.
    """
    scores = {}
    cluster_ids = state['gpstats'].keys() + ['new']
    # print 'data_id', data_id
    for cid in cluster_ids:
        # print 'cid',cid, 'data_id',data_id
        # raw_input()
        score_ll = log_predictive_likelihood_dp_gp(data_id, cid, state)
        # print 'score_ll',score_ll
        # raw_input()
        scores[cid] = score_ll
        score_clust = log_cluster_assign_score_dp(cid, state)
        # print 'score_clust',score_clust
        # raw_input()
        scores[cid] += score_clust
        # degub
        # print 'cid',cid
        # if cid != 'new':
        #     print 'cid size',state['suffstats'][cid].N_k
        # print 'score_ll',score_ll
        # print 'score_clust',score_clust
        # raw_input()
    scores = {cid: np.exp(score) for cid, score in scores.iteritems()}
    normalization = 1.0/sum(scores.values())
    scores = {cid: np.asscalar(score*normalization) for cid, score in scores.iteritems()}
    # print 'scores', scores
    # raw_input()
    return scores

def add_datapoint_to_gpstats(data_id, cluster_id, state):
    """Add datapoint from the cluster
    """
    assert(state['assignment'][data_id] != cluster_id)
    gp_stat = state['gpstats'][cluster_id]
    assert(gp_stat['N_k'] >= 0)
    assert(data_id not in gp_stat['dids'])
    # assert(gp_stat['gps'])

    state['assignment'][data_id] = cluster_id
    dids = gp_stat['dids']
    dids.append(data_id)
    gp_stat['N_k'] += 1

    if (gp_stat['N_k']==1):
        update_nonempty_cluster_gpstat(state, cluster_id)

def remove_datapoint_from_gpstats(data_id, cluster_id, state):
    """Remove datapoint from the cluster
    """
    assert(state['assignment'][data_id] == cluster_id)
    gp_stat = state['gpstats'][cluster_id]
    assert(gp_stat['N_k'] > 0)
    assert(data_id in gp_stat['dids'])
    assert(gp_stat['gps'])

    state['assignment'][data_id] = -1 # -1 to ingnore the data
    # init_suffstats(state,cluster_id)
    dids = gp_stat['dids']
    data_idx = dids.index(data_id)
    del dids[data_idx]
    gp_stat['N_k'] -= 1

def create_cluster_gp(state):
    state["num_clusters_"] += 1
    cluster_id = max(state['gpstats'].keys()) + 1
    state['cluster_ids_'].append(cluster_id)
    state['gpstats'][cluster_id] = {}
    state['gpstats'][cluster_id]['dids'] = []
    state['gpstats'][cluster_id]['N_k'] = 0
    state['gpstats'][cluster_id]['gps'] = []
    print 'cluster', cluster_id, 'created'
    return cluster_id

def destroy_cluster_gp(state, cluster_id):
    state["num_clusters_"] -= 1 # Bug! changed = to -= TODO
    del state['gpstats'][cluster_id]
    state['cluster_ids_'].remove(cluster_id)
    print 'cluster', cluster_id, 'destroyed'

def prune_clusters_gp(state):
    for cid in state['cluster_ids_']:
        if state['gpstats'][cid]['N_k'] == 0:
            destroy_cluster_gp(state, cid)

def gibbs_step_gp(state):
    # sample indicator variables
    data_id = range(state['N'])
    pairs = zip(data_id, state['data_'], state['assignment'])
    random.shuffle(pairs)
    for _, (data_id, datapoint, cid) in enumerate(pairs):
        remove_datapoint_from_gpstats(data_id,cid,state)
        scores = cluster_assignment_distribution_gp(data_id, state).items()
        labels, scores = zip(*scores)
        if state['sampling_scheme']=='crp':
            cid = random.choice(labels, p=scores)
        else:
            cid = labels[np.argmax(scores)]
        add_datapoint_to_gpstats(data_id,cid,state)
    update_all_gpstats(state)

def sample_assignment_gp(data_id, state):
    """Sample new assignment from marginal distribution.
    If cluster is "`new`", create a new cluster.
    """
    scores = cluster_assignment_distribution_dp_gp(data_id, state).items()
    labels, scores = zip(*scores)
    if state['sampling_scheme']=='crp':
        cid = random.choice(labels, p=scores)
    else:
        cid = labels[np.argmax(scores)]
    # print 'chosen cid',cid
    # raw_input()
    if cid == "new":
        return create_cluster_gp(state)
    else:
        return int(cid)

def init_gibbs_step_dp_gp(state):
    """Collapsed Gibbs sampler for Dirichlet Process Mixture Model
    """
    # sample indicator variables
    pairs = zip(state['data_'], state['assignment'])
    for data_id, (datapoint, cid) in enumerate(pairs):
    # data_id = range(state['N'])
    # pairs = zip(data_id, state['data_'], state['assignment'])
    # random.shuffle(pairs)
    # for _, (data_id, datapoint, cid) in enumerate(pairs):
        cid = sample_assignment_gp(data_id, state)
        add_datapoint_to_gpstats(data_id,cid,state)
    update_all_gpstats(state)

def gibbs_step_dp_gp(state):
    """Collapsed Gibbs sampler for Dirichlet Process Mixture Model
    """
    # sample indicator variables
    data_id = range(state['N'])
    pairs = zip(data_id, state['data_'], state['assignment'])
    random.shuffle(pairs)
    for _, (data_id, datapoint, cid) in enumerate(pairs):
        # print 'data_id',data_id, 'current assignment', cid
        remove_datapoint_from_gpstats(data_id,cid,state)
        # print 'removed data_id', data_id, 'from', 'cluster',cid
        prune_clusters_gp(state)
        cid = sample_assignment_gp(data_id, state)
        # print 'new assignment', cid
        # state['assignment'][data_id] = cid
        add_datapoint_to_gpstats(data_id,cid,state)
        # print 'added data_id', data_id, 'to', 'cluster',cid
    update_all_gpstats(state)

def predictive_ll_cluster(state,pairs):
    # pair: list of tuple of data and cid
    # for each item in pair return predictive ll of data for cluster  cid
    D = state['D']
    scores = []
    for pair in pairs:
        x = pair[1]
        x = x.reshape(D,1)
        cid = pair[0]
        ss = state['suffstats'][cid]
        ll = _log_predictive_likelihood(ss, x, D) + log_cluster_assign_score(cid, state)
        scores.append([cid,ll])
    return scores
