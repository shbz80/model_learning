import scipy.linalg
from collapsed_Gibbs_sampler import *
# from utilities import plot_ellipse
# from utilities import get_N_HexCol
from Archive.utilities import logsum
import copy

def data_ll(state):
    data = np.asarray(state['data_'])
    N, D = data.shape
    K = state["num_clusters_"]
    pi = state['pi']
    pi = np.reshape(pi,(K,1))

    logobs = -0.5*np.ones((N, K))*D*np.log(2*np.pi)
    if np.any(logobs>0):
        print 'first'
        print logobs
        raw_input()
    # for k in range(K):
    #     mu_k, sigma_k = mu[k], sigma[k]
    #     L = scipy.linalg.cholesky(sigma_k, lower=True)
    #     logobs[:, k] -= np.sum(np.log(np.diag(L)))
    #     diff = (data - mu_k).T
    #     soln = scipy.linalg.solve_triangular(L, diff, lower=True)
    #     logobs[:, k] -= 0.5*np.sum(soln**2, axis=0)
    k=0
    for cid in state['cluster_ids_']:
        mu_k = np.array(state['theta'][cid].mean)
        mu_k = np.reshape(mu_k,(1,D))
        sigma_k = np.array(state['theta'][cid].var)
        assert(np.any(scipy.linalg.eigvals(sigma_k)>0))
        L = scipy.linalg.cholesky(sigma_k, lower=True)
        logobs[:, k] -= np.sum(np.log(np.diag(L)))
        diff = (data - mu_k).T
        soln = scipy.linalg.solve_triangular(L, diff, lower=True)
        logobs[:, k] -= 0.5*np.sum(soln**2, axis=0)
        k += 1
    logobs += np.log(pi.T)
    ll = np.sum(logsum(logobs, axis=1))
    # raw_input()
    return ll

def run_Markov_chain(init_state, data, DP_flag,rand_clust_size,seq_init,max_itr):
    state = initial_state(init_state, data, rand_clust_size,seq_init)
    chain_stat = []
    if DP_flag:
        if seq_init:
            init_gibbs_step_dp(state) # use this seq init
        else:
            gibbs_step_dp(state)
    else:
        gibbs_step(state)
    state['ll'] = data_ll(state)

    gibbs_step_stat = copy.deepcopy(state)
    del gibbs_step_stat['data_']
    chain_stat.append(gibbs_step_stat)

    for i in range(1,max_itr):
        if DP_flag:
            gibbs_step_dp(state)
        else:
            gibbs_step(state)
        state['ll'] = data_ll(state)
        if (i%10 == 0):
            print 'Itr:',i,'Number of clusters:', state["num_clusters_"], 'll:',state['ll']
        gibbs_step_stat = copy.deepcopy(state)
        del gibbs_step_stat['data_']
        chain_stat.append(gibbs_step_stat)
    return chain_stat

def ACF(X,t):
    mu = np.mean(X)
    N = X.shape[0]
    assert(N > t)
    num = 0
    den = 0
    for i in range(N-t):
        xd = X[i] - mu
        num += xd*(X[i+t] - mu)
        den += xd*xd
        if xd==0:
            acor = 0
        else:
            acor = num/den
    return acor
