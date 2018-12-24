import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple, Counter
from scipy import stats
from numpy import random
import pprint

np.random.seed(12345)
data = pd.Series.from_csv("clusters.csv")

SuffStat = namedtuple('SuffStat', 'theta N')

def initial_state(num_clusters=3, alpha=1.0):
    cluster_ids = range(num_clusters)

    state = {
        'cluster_ids_': cluster_ids, # from 0 to K-1
        'data_': data, # the data array
        'num_clusters_': num_clusters, # K
        'cluster_variance_': .01, # fixed variance of each Gaussian
        'alpha_': alpha, # DP or Dirichlet parameter
        'hyperparameters_': { # hyper param of the distribution for cluster params
                                # that is shared by all clusters
            "mean": 0,
            "variance": 1,
        },
        'suffstats': {cid: None for cid in cluster_ids}, # sufficient statistics clusters
        'assignment': [random.choice(cluster_ids) for _ in data], # initial random assignments
        'pi': {cid: alpha / num_clusters for cid in cluster_ids}, # alpha_k for symmetric DP or Dirichlet
    }
    update_suffstats(state)
    return state

def update_suffstats(state):
    print state['suffstats']
    print type(state['assignment'])
    print state['assignment']
    for cluster_id, N in Counter(state['assignment']).iteritems():
        points_in_cluster = [x
            for x, cid in zip(state['data_'], state['assignment'])
            if cid == cluster_id
        ]
        mean = np.array(points_in_cluster).mean()
        # update each clusters with their mean and size
        state['suffstats'][cluster_id] = SuffStat(mean, N)

def log_predictive_likelihood(data_id, cluster_id, state):
    """Predictive likelihood of the data at data_id is generated
    by cluster_id given the currenbt state.

    From Section 2.4 of
    http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """
    ss = state['suffstats'][cluster_id]
    hp_mean = state['hyperparameters_']['mean']
    hp_var = state['hyperparameters_']['variance']
    param_var = state['cluster_variance_']
    x = state['data_'][data_id]
    return _log_predictive_likelihood(ss, hp_mean, hp_var, param_var, x)

def log_predictive_likelihood_dp(data_id, cluster_id, state):
    """Predictive likelihood of the data at data_id is generated
    by cluster_id given the currenbt state.

    From Section 2.4 of
    http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """
    if cluster_id == "new":
        ss = SuffStat(0, 0)
    else:
        ss = state['suffstats'][cluster_id]

    hp_mean = state['hyperparameters_']['mean']
    hp_var = state['hyperparameters_']['variance']
    param_var = state['cluster_variance_']
    x = state['data_'][data_id]
    return _log_predictive_likelihood(ss, hp_mean, hp_var, param_var, x)


def _log_predictive_likelihood(ss, hp_mean, hp_var, param_var, x):
    posterior_sigma2 = 1 / (ss.N * 1. / param_var + 1. / hp_var)
    predictive_mu = posterior_sigma2 * (hp_mean * 1. / hp_var + ss.N * ss.theta * 1. / param_var)
    predictive_sigma2 = param_var + posterior_sigma2
    predictive_sd = np.sqrt(predictive_sigma2)
    return stats.norm(predictive_mu, predictive_sd).logpdf(x)


def log_cluster_assign_score(cluster_id, state):
    """Log-likelihood that a new point generated will
    be assigned to cluster_id given the current state.
    """
    current_cluster_size = state['suffstats'][cluster_id].N
    num_clusters = state['num_clusters_']
    alpha = state['alpha_']
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
        return np.log(state["alpha_"])
    else:
        return np.log(state['suffstats'][cluster_id].N)

def cluster_assignment_distribution(data_id, state):
    """Compute the marginal distribution of cluster assignment
    for each cluster.
    """
    scores = {}
    for cid in state['suffstats'].keys():
        scores[cid] = log_predictive_likelihood(data_id, cid, state)
        scores[cid] += log_cluster_assign_score(cid, state)
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

def add_datapoint_to_suffstats(x, ss):
    """Add datapoint to sufficient stats for normal component
    """
    return SuffStat((ss.theta*(ss.N)+x)/(ss.N+1), ss.N+1)


def remove_datapoint_from_suffstats(x, ss):
    """Remove datapoint from sufficient stats for normal component
    """
    assert(ss.N > 0)
    if ss.N == 1:
        return SuffStat(0,0)
    else:
        return SuffStat((ss.theta*(ss.N)-x*1.0)/(ss.N-1), ss.N-1)

def create_cluster(state):
    state["num_clusters_"] += 1
    cluster_id = max(state['suffstats'].keys()) + 1
    state['suffstats'][cluster_id] = SuffStat(0, 0)
    state['cluster_ids_'].append(cluster_id)
    print "Created cluster:", cluster_id
    return cluster_id

def destroy_cluster(state, cluster_id):
    state["num_clusters_"] -= 1 # Bug! changed = to -= TODO
    del state['suffstats'][cluster_id]
    state['cluster_ids_'].remove(cluster_id)
    print "Destroyed cluster:", cluster_id

def prune_clusters(state):
    for cid in state['cluster_ids_']:
        if state['suffstats'][cid].N == 0:
            destroy_cluster(state, cid)

def gibbs_step(state):
    pairs = zip(state['data_'], state['assignment'])
    for data_id, (datapoint, cid) in enumerate(pairs):

        state['suffstats'][cid] = remove_datapoint_from_suffstats(datapoint,
                                                                  state['suffstats'][cid])
        scores = cluster_assignment_distribution(data_id, state).items()
        labels, scores = zip(*scores)
        cid = random.choice(labels, p=scores)
        state['assignment'][data_id] = cid
        state['suffstats'][cid] = add_datapoint_to_suffstats(state['data_'][data_id], state['suffstats'][cid])

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
        state['suffstats'][cid] = remove_datapoint_from_suffstats(datapoint, state['suffstats'][cid])
        prune_clusters(state)
        cid = sample_assignment(data_id, state)
        state['assignment'][data_id] = cid
        state['suffstats'][cid] = add_datapoint_to_suffstats(state['data_'][data_id], state['suffstats'][cid])

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

'''
actual number of clusters 3
convergence criterion: number of cluster for 10 consequtive Gibbs step remains unchanged
with init cluter count 2 and alpha 1 it never really converged, but almost to 4
with alpha 0.1 it converged much earlier to 3 in 36 iterations
with init cluster count 10 and alpha 0.1 it converged to 4 in 67 iterations
'''
state = initial_state(num_clusters=3, alpha=0.1)
nc = state["num_clusters_"]
rcnt = 0
for i in range(100):
    gibbs_step_dp(state)
    print 'Itr:',i,'Number of clusters:', state["num_clusters_"]
    pnc = nc
    nc = state["num_clusters_"]
    if nc == pnc:
        rcnt += 1
    else:
        rcnt = 0
    print 'Repetition counter:',rcnt
    if rcnt == 10: break
print 'Itr:',i,'Converged cluster count:',state["num_clusters_"]

plt.figure()
plot_clusters(state)
plt.show()
