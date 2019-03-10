# import scipy as sp
import pickle
# from mixture_model_gibbs_sampling import ACF
from sklearn import mixture
from Archive.utilities import *
import copy
# from collapsed_Gibbs_sampler import predictive_ll_cluster
import operator
from collections import Counter
import datetime

f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')

# task = 'gmm'
task = 'dpgmm'
# task = 'dpgmm_plot'
# task = 'matlab_dump'
# task = 'gp_fit'
# task='gp_plot'
ds = 4 # down sample
# sample_data = pickle.load( open( "mjc_blocks_processed_10_4.p", "rb" ) )
sample_data = pickle.load( open( "./Results/yumi_blocks_l1_processed_1.p", "rb" ) )
exp_params = sample_data['exp_params']
dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']
dX = dP+dV+dU
dt = exp_params['dt']
T = exp_params['T']

EXt = sample_data['EXt'][::ds]
EXt_1 = sample_data['EXt_1'][::ds]
Ft = sample_data['Ft'][::ds]
Xt = sample_data['Xt'][::ds]
Xt_1 = sample_data['Xt_1'][::ds]
Ut = sample_data['Ut'][::ds]
print Xt.shape,Xt_1.shape,Ut.shape
# T,_ = data.shape
px = EXt[:,0]
py = EXt[:,1]
pz = EXt[:,2]
rx = EXt[:,3]
ry = EXt[:,4]
rz = EXt[:,5]
vx = EXt[:,6]
vy = EXt[:,7]
vz = EXt[:,8]
wx = EXt[:,9]
wy = EXt[:,10]
wz = EXt[:,11]

# Xt = sample_data['Xt']
q = Xt[:,:dP]
q_d = Xt[:,dP:]

mm_result={}
mm_result['train_data'] = sample_data['train_data']
mm_result['test_data'] = sample_data['test_data']
mm_result['clust_data'] = 'Xt,Ut,Xt_1'
mm_result['EXt'] = sample_data['EXt']
mm_result['EXt_1'] = sample_data['EXt_1']
mm_result['Ft'] = sample_data['Ft']
mm_result['Xt'] = sample_data['Xt']
mm_result['Xt_1'] = sample_data['Xt_1']
mm_result['Ut'] = sample_data['Ut']

test_data = sample_data['test_data']
test_data_flattened = test_data.reshape((-1,test_data.shape[-1]))
test_data = test_data_flattened[::ds]
Qt = test_data[:,0:dP].reshape((-1,dP))
Qt_d = test_data[:,dP:dP+dV].reshape((-1,dV))
Ut_test = test_data[:,dP+dV:dP+dV+dU].reshape((-1,dU))
Qt_1 = test_data[:,dP+dV+dU:dP+dV+dU+dP].reshape((-1,dP))
Qt_1_d = test_data[:,dP+dV+dU+dP:dP+dV+dU+dP+dV].reshape((-1,dV))
Xt_test = np.concatenate((Qt,Qt_d),axis=1)
Xt_1_test = np.concatenate((Qt_1,Qt_1_d),axis=1)


if task=='gmm':
    # # data = np.concatenate((Xt,Ut,Xt_1-Xt),axis=1)
    # data = np.concatenate((Xt,Ut,Xt_1),axis=1)
    # # data = Xt
    # # data = Xt
    # starting_K = 10
    # range_K = 1
    # restarts = 50
    # n_components_range = range(starting_K, starting_K+range_K)
    # lowest_bic = np.infty
    # bic = []
    #
    # for n_components in n_components_range:
    #     # Fit a Gaussian mixture with EM
    #     gmm = mixture.GaussianMixture(  n_components=n_components,
    #                                     covariance_type='full',
    #                                     tol=1e-6,
    #                                     n_init=restarts,
    #                                     warm_start=False,
    #                                     init_params='random',
    #                                     max_iter=1000)
    #     gmm.fit(data)
    #     bic.append(gmm.bic(data))
    #     if bic[-1] < lowest_bic:
    #         lowest_bic = bic[-1]
    #         best_gmm = gmm
    # idx = best_gmm.predict(data)
    # K = best_gmm.weights_.shape[0]
    # print 'Best BIC K', K
    # print 'Converged',best_gmm.converged_, 'on', best_gmm.n_iter_, 'iterations with lower bound',best_gmm.lower_bound_
    # labels, counts = zip(*Counter(idx).items())
    # labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))
    # assert(K == len(labels))
    # colors = get_N_HexCol(K)
    # colors = np.asarray(colors)/255.
    #
    # # plt.figure()
    # # plt.bar(labels,counts,color=colors)
    # # plt.title('GMM cluster dist')
    # col = np.zeros([data.shape[0],3])
    # k=0
    # for label in labels:
    #     col[(idx==label)] = colors[k]
    #     k+=1
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.scatter3D(px, py, pz, c = col)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('GMM with K=%d' %K)

    ####################################################################################
    data = np.concatenate((Xt,Ut,Xt_1),axis=1)
    # data = np.concatenate((Xt,Xt_1),axis=1)
    # data = Xt
    starting_K = 8
    range_K = 1
    restarts = 50
    n_components_range = range(starting_K, starting_K+range_K)
    lowest_bic = np.infty
    bic = []

    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(  n_components=n_components,
                                        covariance_type='full',
                                        tol=1e-6,
                                        n_init=restarts,
                                        warm_start=False,
                                        init_params='random',
                                        max_iter=1000)
        gmm.fit(data)
        bic.append(gmm.bic(data))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    idx = best_gmm.predict(data)
    K = best_gmm.weights_.shape[0]
    print 'Best BIC K', K
    print 'Converged',best_gmm.converged_, 'on', best_gmm.n_iter_, 'iterations with lower bound',best_gmm.lower_bound_
    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))
    assert(K == len(labels))
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.

    plt.figure()
    plt.bar(labels,counts,color=colors)
    plt.title('GMM cluster dist')
    col = np.zeros([data.shape[0],3])
    k=0
    for label in labels:
        col[(idx==label)] = colors[k]
        k+=1

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter3D(px, py, pz, c = col)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('GMM with K=%d' %K)

    # plt.figure()
    # for i in range(dP):
    #     for j in range(dV):
    #         plt.subplot(dP,dV,(dP*i + j + 1))
    #         plt.xlabel('q%d' %(i+1))
    #         plt.ylabel('q_d%d' %(j+1))
    #         plt.scatter(q[:,i],q_d[:,j],color=col, s=1.)
    plt.show()

    mm_result['object'] = best_gmm

    # pickle.dump(mm_result, open( "gmm_result_blocks.p", "wb" ) )

if task=='dpgmm':
    X = np.concatenate((Xt,Ut,Xt_1),axis=1)
    N = X.shape[0]
    X = X.reshape(N,-1)
    D = X.shape[1]
    # x_bar = np.reshape(np.mean(X, axis=0),(D,1)) # dataset mean
    # Xc = X.T - x_bar
    # S_x_bar = Xc.dot(Xc.T)
    # S0 = np.diag(np.diag(S_x_bar)/N)
    v0 = D+2
    alpha = 1e-1 # 1e10 looked good although more number of clusters with mass
    reg_covar = 1e-4
    max_components = 30
    restarts = 10
    verbose = 2

    dpgmm = mixture.BayesianGaussianMixture(n_components=max_components,
                                            covariance_type='full',
                                            tol=1e-6,
                                            reg_covar=reg_covar,
                                            n_init=restarts,
                                            max_iter=1000,
                                            weight_concentration_prior_type='dirichlet_process',
                                            weight_concentration_prior=alpha,
                                            mean_precision_prior=None,
                                            mean_prior=None, # None = x_bar
                                            degrees_of_freedom_prior=v0,
                                            covariance_prior=None,
                                            warm_start=False,
                                            init_params='random',
                                            verbose=verbose,
                                            )
    print 'DPGMM fit starting...', datetime.datetime.time(datetime.datetime.now())
    dpgmm.fit(X)
    print 'DPGMM fit ending...', datetime.datetime.time(datetime.datetime.now())
    print 'Converged',dpgmm.converged_, 'on', dpgmm.n_iter_, 'iterations with lower bound',dpgmm.lower_bound_
    idx = dpgmm.predict(X)

    K = dpgmm.weights_.shape[0]
    labels = range(K)
    # colors = get_colors(K)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    plt.figure()
    plt.bar(labels,dpgmm.weights_,color=colors)
    plt.title('DPGMM cluster weights')

    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))
    K = len(labels)
    # colors = color_map.rainbow(np.random.uniform(0, 1, K))
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    # colors = get_colors(K)
    plt.figure()
    plt.bar(labels,counts,color=colors)
    plt.title('DPGMM cluster dist')

    col = np.zeros([X.shape[0],3])
    k=0
    for label in labels:
        col[(idx==label)] = colors[k]
        k+=1

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter3D(px, py, pz, c = col)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('DPGMM with K=%d' %K)

    pi = copy.copy(dpgmm.weights_)
    # max_pi = np.max(pi)
    mean_pi = np.mean(pi)
    # min_pi = mean_pi*0.5
    min_pi = 0.01

    selected_k_idx = np.where(pi>min_pi)[0]
    K = len(selected_k_idx)
    restarts = 10
    reg_covar = 1e-6
    alpha = 1e-1
    v0 = D+2
    dpgmm_params = dpgmm._get_parameters()

    vbgmm = mixture.BayesianGaussianMixture(n_components=K,
                                            covariance_type='full',
                                            tol=1e-6,
                                            reg_covar=reg_covar,
                                            n_init=restarts,
                                            max_iter=200,
                                            weight_concentration_prior_type='dirichlet_distribution',
                                            weight_concentration_prior=alpha,
                                            mean_precision_prior=None,
                                            mean_prior=None, # None = x_bar
                                            degrees_of_freedom_prior=v0,
                                            covariance_prior=None,
                                            warm_start=True,
                                            verbose=verbose,
                                            )

    vbgmm.converged_ = False
    vbgmm.lower_bound_ = -np.infty
    _, log_resp = dpgmm._e_step(X)
    nk, xk, sk = mixture.gaussian_mixture._estimate_gaussian_parameters(X, np.exp(log_resp), dpgmm.reg_covar,
                                               dpgmm.covariance_type)

    vbgmm_params = (    (dpgmm.weight_concentration_prior_ + nk)[selected_k_idx], # weight_concentration_
                        dpgmm_params[1][selected_k_idx], # mean_precision_
                        dpgmm_params[2][selected_k_idx], # means_
                        dpgmm_params[3][selected_k_idx], # degrees_of_freedom_
                        dpgmm_params[4][selected_k_idx], # covariances_
                        dpgmm_params[5][selected_k_idx]) # precisions_cholesky_

    vbgmm._set_parameters(vbgmm_params)
    vbgmm.covariances_/=(vbgmm.degrees_of_freedom_[:, np.newaxis, np.newaxis])
    print 'VBGMM fit starting...', datetime.datetime.time(datetime.datetime.now())
    vbgmm.fit(X)
    print 'VBGMM fit ending...', datetime.datetime.time(datetime.datetime.now())
    print 'Converged',vbgmm.converged_, 'on', vbgmm.n_iter_, 'iterations with lower bound',vbgmm.lower_bound_
    idx = vbgmm.predict(X)
    labels = range(K)
    # colors = get_colors(K)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    plt.figure()
    plt.bar(labels,vbgmm.weights_,color=colors)
    plt.title('VBGMM cluster weights')

    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))
    K = len(labels)
    # colors = color_map.rainbow(np.random.uniform(0, 1, K))
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    # colors = get_colors(K)
    plt.figure()
    plt.bar(labels,counts,color=colors)
    plt.title('VBGMM cluster dist')

    col = np.zeros([X.shape[0],3])
    k=0
    for label in labels:
        col[(idx==label)] = colors[k]
        k+=1

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter3D(px, py, pz, c = col)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('VBGMM with K=%d' %K)


    plt.show()

    mm_result['dpgmm_object'] = dpgmm
    mm_result['vbgmm_object'] = vbgmm

    # pickle.dump(mm_result, open( "2_stage_result_blocks.p", "wb" ) )
    pickle.dump(mm_result, open( "yumi_l1_results_blocks_1.p", "wb" ) )

if task == 'dpgmm_plot':
    X = np.concatenate((Xt,Ut,Xt_1),axis=1)
    N, D = X.shape
    mm_result = pickle.load( open( "yumi_l3_results_blocks_3.p", "rb" ) )
    # mm_result = pickle.load( open( "2_stage_result_blocks_1.p", "rb" ) )
    gmm = mm_result['vbgmm_object']
    # gmm = mm_result['dpgmm_object']

    # plot cluster weights before cutoff
    K = gmm.weights_.shape[0]
    print 'K:',K
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    plt.figure()
    plt.bar(range(K),gmm.weights_,color=colors)
    plt.title('GMM cluster weights')

    idx = gmm.predict(X)
    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))
    print 'len(labels):',len(labels)
    K = len(labels)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    plt.figure()
    plt.bar(labels,counts,color=colors)
    plt.title('GMM cluster dist')

    col = np.zeros([X.shape[0],3])
    k=0
    for label in labels:
        col[(idx==label)] = colors[k]
        k+=1

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter3D(px, py, pz, c = col)
    # ax = fig.gca(projection='3d')
    # col_quiver = np.zeros([N*3,3])
    # Tj = N
    # # num_traj = 3
    # # Tj = (T-1)*num_traj
    # for i in range(Tj):
    #     col_quiver[i] = col[i]
    #     col_quiver[i*2+Tj] = col[i]
    #     col_quiver[i*2+1+Tj] = col[i]
    # ax.quiver(px[:Tj], py[:Tj], pz[:Tj], vx[:Tj], vy[:Tj], vz[:Tj], length=0.05, colors = col_quiver)
    # ax.scatter3D(px, py, pz, c = col)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('GMM with K=%d' %K)
    ax.set_title('Dynamics Clustering with Variational Inference')

    plt.show()

if task=='matlab_dump':

    data_j = np.concatenate((Xt,Ut,Xt_1),axis=1)
    data_ee = np.concatenate((EXt,Ft,EXt_1),axis=1)
    mm_result = pickle.load( open( "2_stage_result_blocks_1.p", "rb" ) )
    gmm = mm_result['vbgmm_object']
    labels = gmm.predict(data_j)

    dump={}
    dump['data_j'] = data_j
    dump['data_ee'] = data_ee
    dump['labels'] = labels

    scipy.io.savemat('/home/shahbaz/Research/Software/model_learning/mat_dump.mat',mdict={'dump': dump})


if task == 'gp_fit':
    XY = np.concatenate((Xt,Ut,Xt_1),axis=1)
    X = np.concatenate((Xt,Ut),axis=1)
    N = X.shape[0]
    Y = Xt_1
    dY = Y.shape[1]
    # mm_result = pickle.load( open( "2_stage_result_blocks_1.p", "rb" ) )
    mm_result = pickle.load( open( "yumi_l2_results_blocks_1.p", "rb" ) )
    print mm_result['Xt'].shape
    gmm = mm_result['vbgmm_object']
    K = gmm.weights_.shape[0]
    idx = gmm.predict(XY)
    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))
    print 'K:',K
    print 'len(labels):',len(labels)
    # assert(K == len(labels))
    K = len(labels)
    # fit gloab GP model
    gp_params = {
                    'alpha': 0., # use this when using white kernal
                    'K_RBF': RBF(np.ones(dX), (1e-3, 1e3)),
                    # 'K_RBF': RBF(1, (1e-3, 1e0)),
                    'K_W': W(noise_level=1., noise_level_bounds=(1e-3, 1e3)),
                    # 'K_W': W(noise_level=1., noise_level_bounds=(1e0, 5e0)),
                    'normalize_y': True,
                    'restarts': 10,
                    }

    alpha = gp_params['alpha']
    K_RBF = gp_params['K_RBF']
    K_W = gp_params['K_W']
    normalize_y = gp_params['normalize_y']
    restarts = gp_params['restarts']
    kernel = K_RBF + K_W

    print 'Global GP fit starting...', datetime.datetime.time(datetime.datetime.now())
    # fit global GP
    # global_GP = []
    # for i in range(Y.shape[1]):
    #     yi = Y[:,i]
    #     y_gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
    #     y_gp.fit(X, yi)
    #     global_GP.append(copy.deepcopy(y_gp))
    #     del y_gp
    global_gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
    global_gp.fit(X, Y)
    print 'Global GP fit ending...', datetime.datetime.time(datetime.datetime.now())
    # global GP score with test data
    X_test = np.concatenate((Xt_test,Ut_test),axis=1)
    Y_test = Xt_1_test
    XY_test = np.concatenate((Xt_test,Ut_test,Xt_1_test),axis=1)
    Y_pred = global_gp.predict(X_test)
    global_gp_score = np.linalg.norm(Y_pred - Y_test)
    print 'Global GP score:',global_gp_score
    # gp_plot(X_test, Y_test,global_gp)
    # global_gp_score = 0
    # for i in range(Y_test.shape[1]):
    #     y_test = Y_test[:,i]
    #     y_pred = global_GP[i].predict(X_test)
    #     score = np.linalg.norm(y_pred - y_test)
    #     global_gp_score += score
        # plt.figure
        # plt.scatter(range(y_test.shape[0]),y_test)
        # plt.scatter(range(y_test.shape[0]),y_pred)
        # plt.show()

    print 'MOE GP fit starting...', datetime.datetime.time(datetime.datetime.now())
    moe_GP = {}
    for k in range(K):
        label = labels[k]
        x = X[(idx==label)]
        y = Y[(idx==label)]
        # Y_GP = []
        # for i in range(y.shape[1]):
        #     yi = y[:,i]
        #     y_gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
        #     y_gp.fit(x, yi)
        #     Y_GP.append(copy.deepcopy(y_gp))
        #     del y_gp
        # moe_GP[label] = copy.deepcopy(Y_GP)
        # del Y_GP
        cluster_gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
        cluster_gp.fit(x, y)
        # gp_plot(x, y,cluster_gp)
        print 'Fitted cluster',k,'GP'
        moe_GP[label] = copy.deepcopy(cluster_gp)
        del cluster_gp
    print 'MOE GP fit ending...', datetime.datetime.time(datetime.datetime.now())
    idx = gmm.predict(XY_test)
    labels, counts = zip(*Counter(idx).items())
    labels, counts = zip(*sorted(Counter(idx).items(),key=operator.itemgetter(0)))
    moe_gp_score = 0
    assert(N==X_test.shape[0]==Y_test.shape[0])
    dY = Y_test.shape[1]
    Y_pred_moe = np.zeros((N,dY))
    for label in labels:
        x_test = X_test[(idx==label)]
        y_test = Y_test[(idx==label)]
        # Y_GP = moe_GP[label]
        # for i in range(y_test.shape[1]):
        #     yi_test = y_test[:,i]
        #     y_gp = Y_GP[i]
        #     yi_pred = y_gp.predict(x_test)
        #     score = np.linalg.norm(yi_pred - yi_test)
        #     moe_gp_score += score
            # plt.figure
            # plt.scatter(range(yi_test.shape[0]),yi_test)
            # plt.scatter(range(yi_test.shape[0]),yi_pred)
            # plt.show()
        cluster_gp = moe_GP[label]
        Y_pred_moe[(idx==label)] = y_pred_moe = cluster_gp.predict(x_test)
        score = np.linalg.norm(y_pred_moe - y_test)
        print 'Cluster',label,'score:',score
        # gp_plot(x_test, y_test,cluster_gp)
    moe_gp_score = np.linalg.norm(Y_pred_moe - Y_test)
    print 'MOE GP score:',moe_gp_score
    print 'MoE improvement', (global_gp_score - moe_gp_score)/global_gp_score*100.

    mm_result['global_gp'] = global_gp
    mm_result['global_gp_score'] = global_gp_score
    mm_result['moe_gp'] = moe_GP
    mm_result['moe_gp_score'] = moe_gp_score
    mm_result['idx_test'] = idx
    mm_result['labels_test'] = labels
    pickle.dump(mm_result, open( "yumi_l1_results_blocks_1.p", "wb" ) )

if task=='gp_plot':
    mm_result = pickle.load( open( "yumi_l2_results_blocks_1.p", "rb" ) )
    global_gp = mm_result['global_gp']
    moe_GP = mm_result['moe_gp']
    idx = mm_result['idx_test']
    labels = mm_result['labels_test']
    X_test = np.concatenate((Xt_test,Ut_test),axis=1)
    Y_test = Xt_1_test
    dY = Y_test.shape[1]
    assert(X_test.shape[0]==Y_test.shape[0])
    N = X_test.shape[0]

    Y_pred = global_gp.predict(X_test)
    global_gp_score = np.linalg.norm(Y_pred - Y_test)
    print 'Global GP score:',global_gp_score

    Y_pred_moe = np.zeros((N,dY))
    for label in labels:
        x_test = X_test[(idx==label)]
        y_test = Y_test[(idx==label)]
        cluster_gp = moe_GP[label]
        Y_pred_moe[(idx==label)] = y_pred_moe = cluster_gp.predict(x_test)
        score = np.linalg.norm(y_pred_moe - y_test)
        print 'Cluster',label,'score:',score
    moe_gp_score = np.linalg.norm(Y_pred_moe - Y_test)
    print 'MOE GP score:',moe_gp_score
    print 'MoE improvement', (global_gp_score - moe_gp_score)/global_gp_score*100.

    K = len(labels)
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    col = np.zeros([X_test.shape[0],3])
    k=0
    for label in labels:
        col[(idx==label)] = colors[k]
        k+=1

    fig = plt.figure()
    #ax = fig.add_subplot(1,1,1, projection='3d')
    #ax.scatter3D(px, py, pz, c = col)
    ax = fig.gca(projection='3d')
    col_quiver = np.zeros([N*3,3])
    Tj = N
    # num_traj = 3
    # Tj = (T-1)*num_traj
    for i in range(Tj):
        col_quiver[i] = col[i]
        col_quiver[i*2+Tj] = col[i]
        col_quiver[i*2+1+Tj] = col[i]
    ax.quiver(px[:Tj], py[:Tj], pz[:Tj], vx[:Tj], vy[:Tj], vz[:Tj], length=0.05, colors = col_quiver)
    # ax.scatter3D(px, py, pz, c = col)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('GMM with K=%d' %K)

    plt.figure()
    plt.title('Prediction errors')
    global_error = np.sum((Y_pred - Y_test)**2,axis=1)
    moe_error = np.sum((Y_pred_moe - Y_test)**2,axis=1)
    diff_error = global_error - moe_error
    # plt.scatter(range(N),np.sum((Y_pred - Y_test)**2,axis=1),color=col,marker='o')
    # plt.scatter(range(N),np.sum((Y_pred_moe - Y_test)**2,axis=1),color=col,marker='x')
    plt.scatter(range(N),diff_error,color=col,marker='x')
    plt.show()
