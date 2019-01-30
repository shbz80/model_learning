import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mat_col
from utilities import plot_ellipse
from utilities import get_N_HexCol
import pickle
# from mixture_model_gibbs_sampling import ACF
from collections import Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn import mixture
from scipy.special import digamma
from collections import Counter
from utilities import conditional_Gaussian_mixture
from utilities import estep
from utilities import gp_plot
from copy import deepcopy
# from collapsed_Gibbs_sampler import predictive_ll_cluster
import operator
from gmm import GMM
import datetime
import copy

sample_data = pickle.load( open( "mjc_1d_4mode_raw.p", "rb" ) ) # final data
exp_params = sample_data['exp_params']
# task = 'raw_data'
# task = 'GMM'
# task = 'skgmm'
# task = 'vbgmm'
# task = 'dpgmm' # do not run this. It will rewrite
# task = 'plot'
# task = 'gp_fit'
task = 'gp_predict'
# task = 'visual'
# task = 'compute'
# raw exp data visualization

X = sample_data['X'] # N X T X dX
U = sample_data['U'] # N X T X dU

dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']
N, T, dX = X.shape

assert(X.shape[2]==(dP+dV))
assert(dP==dV)

P = X[:,:,0:dP].reshape((N,T,dP))
V = X[:,:,dP:dP+dV].reshape((N,T,dV))

XU = np.zeros((N,T,dX+dU))
for n in range(N):
    XU[n] = np.concatenate((X[n,:,:],U[n,:,:]),axis=1)
XU_t = XU[:,:-1,:]
X_t1 = XU[:,1:,:dX]
X_t = XU[:,:-1,:dX]
delX = X_t1 - X_t

data = np.concatenate((XU_t,X_t1),axis=2)
# data = np.concatenate((XU_t,delX),axis=2)
train_data = data[0:10,:,:] # half-half training and test data, move this to exp_params
test_data = data[10:,:,:]
data = train_data.reshape((-1,train_data.shape[-1]))
test_data = test_data.reshape((-1,test_data.shape[-1]))

# # ground truth data
# ground_data = pickle.load( open( "mjc_1d_4mode_raw_1_ground.p", "rb" ) )
# X_g = ground_data['X'] # N X T X dX
# U_g = ground_data['U'] # N X T X dU
#
# P_g = X_g[:,:,0:dP].reshape((1,T,dP))
# V_g = X_g[:,:,dP:dP+dV].reshape((1,T,dV))
#
# XU_g = np.zeros((1,T,dX+dU))
# for n in range(1):
#     XU_g[n] = np.concatenate((X_g[n,:,:],U_g[n,:,:]),axis=1)
# XU_t_g = XU_g[:,:-1,:]
# X_t1_g = XU_g[:,1:,:dX]
# X_t_g = XU_g[:,:-1,:dX]
# delX = X_t1_g - X_t_g
#
# data_g = np.concatenate((XU_t_g,X_t1_g),axis=2)
# true_data = data_g.reshape((-1,data_g.shape[-1]))
# XY_tr = true_data
# X_tr = XY_tr[:,:dP+dV+dU]
# Y_tr = XY_tr[:,dP+dV+dU:]

if task=='raw_data':

    i = range(exp_params['T'])
    t = np.array(i)*exp_params['dt']

    pos_samples = P.reshape((N,T)).T
    vel_samples = V.reshape((N,T)).T
    act_samples = U.reshape((N,T)).T

    plt.figure()
    plt.plot(t,pos_samples)

    plt.figure()
    plt.plot(t,vel_samples)

    plt.figure()
    plt.plot(t,act_samples)
    plt.show()


if task=='GMM':

    # prepare reduced data set for clustering
    p_r = data[:,0].reshape(-1,1)
    # u_r = data[:,1].reshape(-1,1)
    x1_r = data[:,4:]
    data_r = np.concatenate((p_r,x1_r),axis=1)

    # prepare data set for state space clustering
    p_r = data[:,0].reshape(-1,1)
    v_r = data[:,1].reshape(-1,1)
    data_r = np.concatenate((p_r,v_r),axis=1)

    restarts = 10
    K = 5
    Gmm = []
    ll = np.zeros(restarts)
    for it in range(restarts):
        # gmm = GMM(dxu=3,dxux=5)
        # gmm.update(data,K)
        gmm = GMM(dxu=1,dxux=2)    # change this when altering the dims to include in clustering
        gmm.update(data_r,K)
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
    col = np.zeros([data_r.shape[0],3])
    # idx = np.array(assignment)
    for k in range(K):
        col[(idx==k)] = colors[k]

    p = data[:,0]
    v = data[:,1]
    u = data[:,2]
    delp = data[:,3]
    delv = data[:,4]

    p1 = delp
    v1 = delv
    # p1 = delp + p
    # v1 = delv + v
    plt.figure()
    plt.subplot(231)
    plt.xlabel('p(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(p,p1,c=col/255.0)
    plt.subplot(232)
    plt.xlabel('v(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(v,p1,c=col/255.0)
    plt.subplot(233)
    plt.xlabel('u(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(u,p1,c=col/255.0)
    plt.subplot(234)
    plt.xlabel('p(t)')
    plt.ylabel('v(t+1)')
    # plt.scatter(p,v1,c=col/255.0)
    plt.scatter(p,v,c=col/255.0) # only for state-space clustering
    plt.subplot(235)
    plt.xlabel('v(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(v,v1,c=col/255.0)
    plt.subplot(236)
    plt.xlabel('u(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(u,v1,c=col/255.0)



    # p = p.reshape(5,-1)[0]
    # v = v.reshape(5,-1)[0]
    p = p[:p.shape[0]/5]
    v = v[:v.shape[0]/5]
    N = p.shape[0]
    i = range(N)
    t = np.array(i)*exp_params['dt']
    col = col[:N]
    plt.figure()
    plt.subplot(121)
    plt.xlabel('t')
    plt.ylabel('p(t)')
    plt.scatter(t,p,color=col/255.0)
    plt.subplot(122)
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.scatter(t,v,color=col/255.0)
    plt.show()

if task=='skgmm':

    mjc_1d_results = {}
    restarts = 20
    K = 30
    n_components = K
    gmm = mixture.GaussianMixture(  n_components=n_components,
                                    covariance_type='full',
                                    tol=1e-6,
                                    n_init=restarts,
                                    warm_start=False,
                                    init_params='random',
                                    max_iter=1000)
    gmm.fit(data)

    idx = gmm.predict(data)
    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))

    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    plt.figure()
    plt.bar(range(K),gmm.weights_,color=colors)
    plt.ylabel('Cluster weights')
    plt.xlabel('Cluster labels')
    # plt.title('GMM cluster weights')

    K = len(labels)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    col = np.zeros([data.shape[0],3])
    k=0
    for label in labels:
        col[(idx==label)] = colors[k]
        k+=1
    plt.figure()
    plt.bar(labels,counts,color=colors)
    plt.title('GMM cluster dist')

    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    col = np.zeros([data.shape[0],3])
    # idx = np.array(assignment)
    for k in range(K):
        col[(idx==k)] = colors[k]

    p = data[:,0]
    v = data[:,1]
    u = data[:,2]
    p1 = data[:,3]
    v1 = data[:,4]

    plt.figure()
    plt.subplot(231)
    plt.xlabel('p(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(p,p1,c=col)
    plt.subplot(232)
    plt.xlabel('v(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(v,p1,c=col)
    plt.subplot(233)
    plt.xlabel('u(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(u,p1,c=col)
    plt.subplot(234)
    plt.xlabel('p(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(p,v1,c=col)
    plt.subplot(235)
    plt.xlabel('v(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(v,v1,c=col)
    plt.subplot(236)
    plt.xlabel('u(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(u,v1,c=col)

    p = p[:p.shape[0]/10]
    v = v[:v.shape[0]/10]
    N = p.shape[0]
    i = range(N)
    t = np.array(i)*exp_params['dt']
    col = col[:N]
    plt.figure()
    plt.subplot(121)
    plt.xlabel('t')
    plt.ylabel('p(t)')
    plt.scatter(t,p,color=col)
    plt.subplot(122)
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.scatter(t,v,color=col)
    plt.show()
    mjc_1d_results['skgmm'] = gmm
    pickle.dump( mjc_1d_results, open( "mjc_1d_4mode_results_1.p", "wb" ) )

if task=='vbgmm':

    N, D = data.shape

    restarts = 50
    K = 50
    alpha = 1e-10
    v0 = D+2
    max_components = K
    vbgmm = mixture.BayesianGaussianMixture(n_components=max_components,
                                            covariance_type='full',
                                            tol=1e-6,
                                            n_init=restarts,
                                            max_iter=1000,
                                            weight_concentration_prior_type='dirichlet_distribution',
                                            weight_concentration_prior=alpha,
                                            mean_precision_prior=None,
                                            mean_prior=None, # None = x_bar
                                            degrees_of_freedom_prior=v0,
                                            covariance_prior=None,
                                            warm_start=False,
                                            init_params='random'
                                            )
    vbgmm.fit(data)

    K = vbgmm.weights_.shape[0]
    labels = range(K)
    # colors = get_colors(K)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    plt.figure()
    plt.bar(labels,vbgmm.weights_,color=colors)
    plt.title('VBGMM cluster weights')

    idx = vbgmm.predict(data)
    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))

    K = len(labels)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    col = np.zeros([data.shape[0],3])
    k=0
    for label in labels:
        col[(idx==label)] = colors[k]
        k+=1

    plt.figure()
    plt.bar(labels,counts,color=colors)
    plt.title('VBGMM cluster dist')

    p = data[:,0]
    v = data[:,1]
    u = data[:,2]
    p1 = data[:,3]
    v1 = data[:,4]

    plt.figure()
    plt.subplot(231)
    plt.xlabel('p(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(p,p1,c=col)
    plt.subplot(232)
    plt.xlabel('v(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(v,p1,c=col)
    plt.subplot(233)
    plt.xlabel('u(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(u,p1,c=col)
    plt.subplot(234)
    plt.xlabel('p(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(p,v1,c=col)
    plt.subplot(235)
    plt.xlabel('v(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(v,v1,c=col)
    plt.subplot(236)
    plt.xlabel('u(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(u,v1,c=col)

    p = p[:p.shape[0]/10]
    v = v[:v.shape[0]/10]
    N = p.shape[0]
    i = range(N)
    t = np.array(i)*exp_params['dt']
    col = col[:N]
    plt.figure()
    plt.subplot(121)
    plt.xlabel('t')
    plt.ylabel('p(t)')
    plt.scatter(t,p,color=col)
    plt.subplot(122)
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.scatter(t,v,color=col)
    plt.show()

if task=='dpgmm':
    mjc_1d_results = pickle.load( open( "mjc_1d_4mode_results_1.p", "rb" ) )
    N, D = data.shape
    restarts = 20
    K = 30
    alpha = 1e0
    v0 = D+2
    max_components = K
    dpgmm = mixture.BayesianGaussianMixture(n_components=max_components,
                                            covariance_type='full',
                                            tol=1e-6,
                                            n_init=restarts,
                                            max_iter=1000,
                                            weight_concentration_prior_type='dirichlet_process',
                                            weight_concentration_prior=alpha,
                                            mean_precision_prior=None,
                                            mean_prior=None, # None = x_bar
                                            degrees_of_freedom_prior=v0,
                                            covariance_prior=None,
                                            warm_start=False,
                                            init_params='random'
                                            )
    dpgmm.fit(data)
    print 'Converged DPGMM',dpgmm.converged_, 'on', dpgmm.n_iter_, 'iterations with lower bound',dpgmm.lower_bound_
    K = dpgmm.weights_.shape[0]

    # colors = get_colors(K)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    plt.figure()
    plt.bar(range(K),dpgmm.weights_,color=colors)
    plt.title('DPGMM cluster weights')

    idx = dpgmm.predict(data)
    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))
    colors_in_labels = colors[list(labels)]
    plt.figure()
    plt.bar(labels,counts,color=colors_in_labels)
    plt.title('DPGMM cluster dist')

    col = np.zeros([data.shape[0],3])
    for label in labels:
        col[(idx==label)] = colors[label]

    p = data[:,0]
    v = data[:,1]
    u = data[:,2]
    p1 = data[:,3]
    v1 = data[:,4]

    plt.figure()
    plt.subplot(231)
    plt.xlabel('p(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(p,p1,c=col)
    plt.subplot(232)
    plt.xlabel('v(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(v,p1,c=col)
    plt.subplot(233)
    plt.xlabel('u(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(u,p1,c=col)
    plt.subplot(234)
    plt.xlabel('p(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(p,v1,c=col)
    plt.subplot(235)
    plt.xlabel('v(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(v,v1,c=col)
    plt.subplot(236)
    plt.xlabel('u(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(u,v1,c=col)

    p = p[:p.shape[0]/10]
    v = v[:v.shape[0]/10]
    N = p.shape[0]
    i = range(N)
    t = np.array(i)*exp_params['dt']
    col = col[:N]
    plt.figure()
    plt.subplot(121)
    plt.xlabel('t')
    plt.ylabel('p(t)')
    plt.scatter(t,p,color=col)
    plt.subplot(122)
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.scatter(t,v,color=col)
    plt.show()



    mjc_1d_results['dpgmm'] = dpgmm
    pickle.dump( mjc_1d_results, open( "mjc_1d_4mode_results_1.p", "wb" ) )

if task == 'plot':
    mjc_1d_results = pickle.load( open( "mjc_1d_4mode_results_1.p", "rb" ) )
    gmm = mjc_1d_results['skgmm']
    K_gmm = gmm.weights_.shape[0]
    dpgmm = mjc_1d_results['dpgmm']
    K_dpgmm = dpgmm.weights_.shape[0]
    assert(K_gmm==K_dpgmm)
    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.bar(range(K_gmm),gmm.weights_)
    # plt.title('GMM-EM')
    plt.ylabel('Cluster weights')
    plt.xlabel('Cluster labels')
    plt.savefig('gmm_em_weights_block_mass.pdf')
    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.bar(range(K_dpgmm),dpgmm.weights_)
    # plt.title('DPGMM')
    plt.ylabel('Cluster weights')
    plt.xlabel('Cluster labels')
    plt.savefig('var_dpgmm_weights_block_mass.pdf')

    p = data[:,0]
    v = data[:,1]
    u = data[:,2]
    p1 = data[:,3]
    v1 = data[:,4]

    idx = dpgmm.predict(data)
    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))

    K = len(labels)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    col = np.zeros([data.shape[0],3])
    k=0
    for label in labels:
        col[(idx==label)] = colors[k]
        k+=1

    plt.figure()
    plt.subplot(231)
    plt.xlabel(r'$q(t)$')
    plt.ylabel(r'$q(t+1)$')
    plt.scatter(p,p1,c=col)
    plt.subplot(232)
    # plt.xlabel(r'$\dot q(t)$')
    # plt.ylabel(r'$q(t+1)$')
    plt.scatter(v,p1,c=col)
    plt.subplot(233)
    # plt.xlabel(r'$f(t)$')
    # plt.ylabel(r'$q(t+1)$')
    plt.scatter(u,p1,c=col)
    plt.subplot(234)
    plt.xlabel(r'$q(t)$')
    plt.ylabel(r'$\dot q(t+1)$')
    plt.scatter(p,v1,c=col)
    plt.subplot(235)
    plt.xlabel(r'$\dot q(t)$')
    # plt.ylabel(r'$\dot q(t+1)$')
    plt.scatter(v,v1,c=col)
    plt.subplot(236)
    plt.xlabel(r'$f(t)$')
    # plt.ylabel(r'$\dot q(t+1)$')
    plt.scatter(u,v1,c=col)
    plt.savefig('block_mass_clust.pdf')

    plt.figure()
    plt.xlabel(r'$q(t)$')
    plt.ylabel(r'$\dot q(t+1)$')
    plt.scatter(p,v1,c=col)
    plt.savefig('x_v1_block_mass.pdf')
    plt.figure()
    plt.xlabel(r'$\dotq (t)$')
    plt.ylabel(r'$\dot q(t+1)$')
    plt.scatter(v,v1,c=col)
    plt.savefig('v_v1_block_mass.pdf')

    plt.figure()
    plt.plot()

    plt.show()

if task=='gp_fit':
    XY = data
    X = XY[:,:dP+dV+dU]
    Y = XY[:,dP+dV+dU:]
    N,D = XY.shape

    XY_test = test_data
    X_test = XY_test[:,:dP+dV+dU]
    Y_test = XY_test[:,dP+dV+dU:]

    mjc_1d_results = pickle.load( open( "mjc_1d_4mode_results_1.p", "rb" ) )
    dpgmm = mjc_1d_results['dpgmm']
    idx = dpgmm.predict(data)
    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))

    K = len(labels)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    col = np.zeros([data.shape[0],3])
    k=0
    for label in labels:
        col[(idx==label)] = colors[k]
        k+=1

    gp_params = {
                    'alpha': 0., # use this when using white kernal
                    'K_C': C(1.0, (1e-3, 1e3)),
                    'K_RBF': RBF(np.ones(dP+dV+dU), (1e-3, 1e3)),
                    # 'K_RBF': RBF(np.ones(3), (1e-3, 1e3)),
                    'K_W': W(noise_level=1., noise_level_bounds=(1e-3, 1e3)),
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

    # fit global GP
    print 'Global GP fit starting...', datetime.datetime.time(datetime.datetime.now())
    global_gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
    global_gp.fit(X, Y)
    print 'Global GP fit ending...', datetime.datetime.time(datetime.datetime.now())

    # fit MOE
    print 'MOE GP fit starting...', datetime.datetime.time(datetime.datetime.now())
    moe_GP = {}
    for k in range(K):
        label = labels[k]
        x = X[(idx==label)]
        y = Y[(idx==label)]
        cluster_gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
        cluster_gp.fit(x, y)
        # gp_plot(x, y,cluster_gp)
        print 'Fitted cluster',k,'GP'
        moe_GP[label] = copy.deepcopy(cluster_gp)
        del cluster_gp
    print 'MOE GP fit ending...', datetime.datetime.time(datetime.datetime.now())

    mjc_1d_results['global_gp'] = global_gp
    mjc_1d_results['moe_GP'] = moe_GP
    pickle.dump( mjc_1d_results, open( "mjc_1d_4mode_results_1.p", "wb" ) )


if task=='gp_predict':

    XY_test = test_data
    # XY_test = XY_tr

    N,D = XY_test.shape
    X_test = XY_test[:,:dP+dV+dU]
    Y_test = XY_test[:,dP+dV+dU:]

    mjc_1d_results = pickle.load( open( "mjc_1d_4mode_results_1.p", "rb" ) )
    dpgmm = mjc_1d_results['dpgmm']
    global_gp = mjc_1d_results['global_gp']
    moe_GP = mjc_1d_results['moe_GP']
    dpgmm_idx = dpgmm.predict(data)
    Y_pred, Ystd = global_gp.predict(X_test, return_std=True)
    global_gp_score = np.linalg.norm(Y_pred - Y_test)
    print 'Global GP score:',global_gp_score

    # prediction with IO data
    idx_test_xy = dpgmm.predict(XY_test)

    # predict with I data_train
    _, XY_n_features = XY_test.shape
    _, X_n_features = X_test.shape
    X_means = dpgmm.means_[:,:X_n_features]
    X_precisions_cholesky = dpgmm.precisions_cholesky_[:,:X_n_features,:X_n_features]
    X_covariance_type = dpgmm.covariance_type
    dof = dpgmm.degrees_of_freedom_ - XY_n_features + X_n_features
    X_mean_precision = dpgmm.mean_precision_
    log_gauss_ = (mixture.gaussian_mixture._estimate_log_gaussian_prob(
        X_test, X_means, X_precisions_cholesky, X_covariance_type) -
        .5 * X_n_features * np.log(dof))

    log_lambda_ = X_n_features * np.log(2.) + np.sum(digamma(
        .5 * (dof - np.arange(0, X_n_features)[:, np.newaxis])), 0)

    log_prob_X_test = log_gauss_ + .5 * (log_lambda_ - X_n_features / X_mean_precision)
    weighted_log_prob = log_prob_X_test + dpgmm._estimate_log_weights()
    idx_test_x = weighted_log_prob.argmax(axis=1)

    gpc_params = {
                    'K_RBF': RBF(1, (1e-3, 1e3)),
                    'restarts': 10,
                    'gpc_enable':True
                    }
    kernel = gpc_params['K_RBF']
    gpc = GaussianProcessClassifier(  kernel=kernel,
                                    n_restarts_optimizer=gpc_params['restarts'],)
    # prediction with IO data
    print 'GPC fit starting...', datetime.datetime.time(datetime.datetime.now())
    gpc.fit(X[::2],dpgmm_idx[::2])
    # gpc.fit(X_train,dpgmm_idx)
    print 'GPC fit ending...', datetime.datetime.time(datetime.datetime.now())
    print 'GPC test starting...', datetime.datetime.time(datetime.datetime.now())
    gpc_test_idx = gpc.predict(X_test)
    print 'GPC test ending...', datetime.datetime.time(datetime.datetime.now())

    # plt.figure()
    # plt.plot(range(len(idx_test_xy)),idx_test_xy,range(len(idx_test_xy)),idx_test_x)
    # idx_test = idx_test_x
    # idx_test = idx_test_xy
    idx_test = gpc_test_idx

    labels, counts = zip(*Counter(idx_test).items())
    labels, counts = zip(*sorted(Counter(idx_test).items(),key=operator.itemgetter(0)))
    print labels, counts
    moe_gp_score = 0
    Y_pred_moe = np.zeros((N,dX))
    for label in labels:
        x_test = X_test[(idx_test==label)]
        y_test = Y_test[(idx_test==label)]
        cluster_gp = moe_GP[label]
        Y_pred_moe[(idx_test==label)] = cluster_gp.predict(x_test)
        score = np.linalg.norm(Y_pred_moe[(idx_test==label)] - y_test)
        print 'score', score

    # print Y_pred.shape, Y_pred_moe.shape, Y_test.shape
    # plt.figure()
    # plt.plot(range(T-1),Y_tr[:T-1,0],label='Y_true')
    # # plt.plot(range(T-1),Y_test[:T-1,0],label='Y_test')
    # plt.plot(range(T-1),Y_pred[:T-1,0],label='Y_pred')
    # plt.plot(range(T-1),Y_pred_moe[:T-1,0],label='Y_pred_moe')
    # plt.legend()
    # plt.figure()
    # plt.plot(range(T-1),Y_tr[:T-1,1],label='Y_true')
    # # plt.plot(range(T-1),Y_test[:T-1,1],label='Y_test')
    # plt.plot(range(T-1),Y_pred[:T-1,1],label='Y_pred')
    # plt.plot(range(T-1),Y_pred_moe[:T-1,1],label='Y_pred_moe')
    # plt.legend()
    moe_gp_score = np.linalg.norm(Y_pred_moe - Y_test)
    print 'MOE GP score:',moe_gp_score
    print 'MoE improvement', (global_gp_score - moe_gp_score)/global_gp_score*100.
    plt.show()

if task=='visual':
    # exp_result = pickle.load( open( "mjc_1d_finite_gauss_clustered.p", "rb" ) )
    exp_result = pickle.load( open( "mjc_1d_infinite_gauss_clustered_3.p", "rb" ) )
    exp_stats = exp_result['exp_stats']
    params = exp_result['params']

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
    exp_result = pickle.load( open( "mjc_1d_infinite_gauss_clustered.p", "rb" ) )
    exp_stats = exp_result['exp_stats']
    params = exp_result['params']

    # chose best chain from the above two plots
    best_chain_id = 2
    mix_itr = 800

    best_chain = exp_stats[best_chain_id]
    chain_len = len(best_chain)
    burn_in_itr = mix_itr * 5
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
    data = exp_result['train_data']
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

    # plot mujoco_1d_block clustered data
    # pc = [best_chain[best_itr]['theta'][cid].mean[0] for cid in cids]
    # pc = np.array(pc)

    p = data[:,0]
    v = data[:,1]
    u = data[:,2]
    delp = data[:,3]
    delv = data[:,4]

    p1 = delp
    v1 = delv
    # p1 = delp + p
    # v1 = delv + v
    plt.figure()
    plt.subplot(231)
    plt.xlabel('p(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(p,p1,c=col/255.0)
    plt.subplot(232)
    plt.xlabel('v(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(v,p1,c=col/255.0)
    plt.subplot(233)
    plt.xlabel('u(t)')
    plt.ylabel('p(t+1)')
    plt.scatter(u,p1,c=col/255.0)
    plt.subplot(234)
    plt.xlabel('p(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(p,v1,c=col/255.0)
    plt.subplot(235)
    plt.xlabel('v(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(v,v1,c=col/255.0)
    plt.subplot(236)
    plt.xlabel('u(t)')
    plt.ylabel('v(t+1)')
    plt.scatter(u,v1,c=col/255.0)



    # p = p.reshape(5,-1)[0]
    # v = v.reshape(5,-1)[0]
    p = p[:p.shape[0]/5]
    v = v[:v.shape[0]/5]
    N = p.shape[0]
    i = range(N)
    t = np.array(i)*exp_params['dt']
    col = col[:N]
    plt.figure()
    plt.subplot(121)
    plt.xlabel('t')
    plt.ylabel('p(t)')
    plt.scatter(t,p,color=col/255.0)
    plt.subplot(122)
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.scatter(t,v,color=col/255.0)
