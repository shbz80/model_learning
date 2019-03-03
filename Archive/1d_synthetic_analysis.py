import sys
# sys.path.insert(0, '/home/shahbaz/Research/Software/Spyder_ws/gps/python')
import numpy as np
import matplotlib.pyplot as plt
# from discont_function import DiscontinuousFunction
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from scipy.special import digamma
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from copy import deepcopy
from utilities import get_N_HexCol
from utilities import plot_ellipse
from utilities import logsum
from scipy import stats
from sklearn import mixture
from collections import Counter
import operator
import datetime

def genSamples(discon_params, sec_list=None, plot=False, xgrid='lin'):
    dt = discon_params['dt']
    T = discon_params['T']
    Nsec = discon_params['Nsec']
    Nsam = discon_params['Nsam']
    noise_gain = discon_params['noise_gain']
    m = discon_params['lin_m']
    a = discon_params['quad_a']
    ap = discon_params['sin_a']
    o = discon_params['offset']

    sec_types = ['flat','lin','quad','sin']
    if sec_list is None:
        sec_list = []
        for sec_i in range(Nsec):
            sec_type = np.random.choice(sec_types)
            sec_list.append(sec_type)
    Nsec = len(sec_list)
    N = int(T/dt)
    Ns = N/Nsec
    Ts = T/float(Nsec)

    if xgrid=='rand':
        xf = np.sort(np.random.uniform(0.,1.,Ns))
        xl = np.sort(np.random.uniform(-0.5,0.5,Ns))
        xq = np.sort(np.random.uniform(0.,1.,Ns))
        xs = np.sort(np.random.uniform(0.,2.*np.pi,Ns))
    elif xgrid=='lin':
        xf = np.linspace(0.,1.,Ns)
        xl = np.linspace(-0.5,0.5,Ns)
        xq = np.linspace(0.,1.,Ns)
        xs = np.linspace(0.,2.*np.pi,Ns)

    xf_r = np.linspace(0.,1.,Ns*Nsam/2)
    xl_r = np.linspace(-0.5,0.5,Ns*Nsam/2)
    xq_r = np.linspace(0.,1.,Ns*Nsam/2)
    xs_r = np.linspace(0.,2.*np.pi,Ns*Nsam/2)

    yf = np.zeros(xf.shape)
    yl = xl*m
    yq = a*xq**2
    ys = ap*np.sin(xs)

    yf_r = np.zeros(Ns*Nsam/2)
    yl_r = xl_r*m
    yq_r = a*xq_r**2
    ys_r = ap*np.sin(xs_r)

    xl = xl + 0.5
    xs = xs/(2.*np.pi)

    xl_r = xl_r + 0.5
    xs_r = xs_r/(2.*np.pi)

    yq =  yq - a/2.
    yq_r =  yq_r - a/2.

    i=0
    for sec_type in sec_list:
        p = i%2
        if sec_type == 'flat':
            x = (xf + float(i))*Ts
            y = yf+o if p else yf-o
            y_n = y + np.random.normal(size=y.shape)*noise_gain*0.3
            x_r = (xf_r + float(i))*Ts
            y_r = yf_r+o if p else yf_r-o
        elif sec_type == 'lin':
            x = (xl + float(i))*Ts
            y = yl+o if p else yl-o
            y_n = y + np.random.normal(size=y.shape)*noise_gain
            x_r = (xl_r + float(i))*Ts
            y_r = yl_r+o if p else yl_r-o
        elif sec_type == 'quad':
            x = (xq + float(i))*Ts
            y = yq+o if p else yq-o
            y_n = y + np.random.normal(size=y.shape)*noise_gain*0.7
            x_r = (xq_r + float(i))*Ts
            y_r = yq_r+o if p else yq_r-o
        elif sec_type == 'sin':
            x = (xs + float(i))*Ts
            y = ys+o if p else ys-o
            y_n = y + np.random.normal(size=y.shape)*noise_gain*.5
            x_r = (xs_r + float(i))*Ts
            y_r = ys_r+o if p else ys_r-o
        else:
            raise NotImplementedError()
        if i==0:
            xt = x.reshape(-1,1)
            yt = y.reshape(-1,1)
            yt_n = y_n.reshape(-1,1)
            xt_r = x_r.reshape(-1,1)
            yt_r = y_r.reshape(-1,1)
        else:
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            y_n = y_n.reshape(-1,1)
            xt = np.concatenate((xt,x),axis=0)
            yt = np.concatenate((yt,y),axis=0)
            yt_n = np.concatenate((yt_n,y_n),axis=0)

            x_r = x_r.reshape(-1,1)
            y_r = y_r.reshape(-1,1)
            xt_r = np.concatenate((xt_r,x_r),axis=0)
            yt_r = np.concatenate((yt_r,y_r),axis=0)
        i += 1
    if plot:
        plt.figure()
        plt.scatter(xt,yt_n)
        plt.plot(xt_r,yt_r,color='r')
        plt.show()
    return xt_r,yt_r,xt,yt_n

def genNsamples(discon_params, sec_list=None, plot=False, xgrid='lin'):
    sec_types = ['flat','lin','quad','sin']
    Nsam = discon_params['Nsam']
    Nsec = discon_params['Nsec']
    if sec_list is None:
        sec_list = []
        for sec_i in range(Nsec):
            sec_type = np.random.choice(sec_types)
            sec_list.append(sec_type)
    x_r,y_r,xn,yn = genSamples(discon_params, sec_list, plot=False, xgrid=xgrid)
    X = xn
    Y = yn
    for i in range(1,Nsam):
        x_r,y_r,xn,yn = genSamples(discon_params, sec_list, plot=False, xgrid=xgrid)
        X = np.concatenate((X,xn),axis=1)
        Y = np.concatenate((Y,yn),axis=1)
    if plot:
        Xp = X.reshape(-1,1)[::2]
        Yp = Y.reshape(-1,1)[::2]
        plt.figure()
        plt.rcParams.update({'font.size': 15})
        plt.scatter(Xp,Yp, label='Noisy data')
        plt.plot(x_r,y_r,color='r', label='True function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig('1d_syn_raw.pdf')
        plt.show()
    return x_r,y_r,X.T,Y.T

# np.random.seed(10)

# works with the new function that can generate random segments
noise_g = 1.
discon_params = {
            # 'dt': 0.05,
            'dt': 0.0075,
            # 'T': 2., # *
            'T': 2.,
            # 'Nsam': 20,
            'Nsam': 3,
            'Nsec': 4, # *
            # 'Nsec': 5,
            'noise_gain': noise_g,
            # 'noise_gain': 2.,
            'lin_m': -7.,
            'quad_a': 10.,
            'sin_a': 3.,
            # 'offset': 7.5,
            'offset': noise_g*2.*3.,
            }
gpr_params = {
                'alpha': 0., # alpha=0 when using white kernal
                'K_RBF': RBF(1, (1e-3, 1e3)),
                'K_W': W(noise_level=1., noise_level_bounds=(1e-3, 1e3)),
                'normalize_y': True,
                'restarts': 10,
                'enable':True
                }
gpc_params = {
                'K_RBF': RBF(1, (1e-3, 1e3)),
                'restarts': 10,
                'enable':False,
                'down_sample': 4,
                }
svm_params = {
                'p_grid': {"C": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], "gamma": [.001, .01, .1, 1., 10., 100.]},
                'kernel': "rbf",
                # 'kernel': "linear",
                'scoring': 'accuracy',
                # 'scoring': 'f1_micro',
                # 'scoring': 'precision',
                'folds': 5,
                'enable': True,
                }


gmm_params = {
                'K': 10, # cluster size
                'restarts': 10, # number of restarts
                'enable': True,
                }
dpgmm_params = {
                'K': 10, # cluster size
                'restarts': 20, # number of restarts
                # 'alpha': 1e-1, *
                'alpha': 1e0,
                'v0': 1+2,
                'enable': True,
                'dpgmm_inference_enable': True,
                'gmm_inference_enable': False,
                }
vbgmm_params = {
                'K': 10, # cluster size
                'restarts': 10, # number of restarts
                'enable': False,
                }
moe_params = {
                'enable': True,
                }


Nsam = discon_params['Nsam']
N_train = 2 * Nsam//3

sec_list=None
sec_list=['flat','quad','sin','lin',]
# sec_list=['flat','lin','flat','quad',]
# sec_list=['sin','sin','sin','sin',]
x_r,y_r,X,Y = genNsamples(discon_params, sec_list,plot=False, xgrid='lin')
N,T = X.shape
assert(N==Nsam)
X_train_s = X[0:N_train,:]
Y_train_s = Y[0:N_train,:]
X_train_unsort = X_train_s.reshape(-1,1)
Y_train_unsort = Y_train_s.reshape(-1,1)
XY_train_unsort = np.concatenate((X_train_unsort,Y_train_unsort),axis=1)
XY_train = np.array(sorted(XY_train_unsort, key=lambda x:x[0]))
X_train = XY_train[:,0].reshape(-1,1)
Y_train = XY_train[:,1].reshape(-1,1)
scaler = StandardScaler().fit(XY_train)
XY_train_std = scaler.transform(XY_train)
X_train_std = XY_train_std[:,0].reshape(-1,1)
Y_train_std = XY_train_std[:,1].reshape(-1,1)

# noisy test data of X and Y
X_test_s = X[N_train:,:]
Y_test_s = Y[N_train:,:]
X_test_unsort = X_test_s.reshape(-1,1)
Y_test_unsort = Y_test_s.reshape(-1,1)
XY_test_unsort = np.concatenate((X_test_unsort,Y_test_unsort),axis=1)
XY_test = np.array(sorted(XY_test_unsort, key=lambda x:x[0]))
X_test = XY_test[:,0].reshape(-1,1)
Y_test = XY_test[:,1].reshape(-1,1)
XY_test_std = scaler.transform(XY_test)
X_test_std = XY_test_std[:,0].reshape(-1,1)
Y_test_std = XY_test_std[:,1].reshape(-1,1)

# uniform grid data of true function
# N_test = N-N_train
X_r = x_r.reshape(-1,1)
Y_r = y_r.reshape(-1,1)
XY_r = np.concatenate((X_r,Y_r),axis=1)
XY_r_std = scaler.transform(XY_r)
X_r_std = XY_r_std[:,0].reshape(-1,1)
Y_r_std = XY_r_std[:,1].reshape(-1,1)
# X_r_s = x_r.reshape(N_test,-1)
# Y_r_s = y_r.reshape(N_test,-1)

# XY_t = XY_r
# X_t = X_r
# Y_t = Y_r
# XY_t_std = XY_r_std
# X_t_std = X_r_std
# Y_t_std = Y_r_std

XY_t = XY_test
X_t = X_test
Y_t = Y_test
XY_t_std = XY_test_std
X_t_std = X_test_std
Y_t_std = Y_test_std

# GP regression
if gpr_params['enable']:
    kernel = gpr_params['K_RBF'] + gpr_params['K_W']
    gp = GaussianProcessRegressor(  kernel=kernel,
                                    alpha=gpr_params['alpha'],
                                    n_restarts_optimizer=gpr_params['restarts'],
                                    normalize_y=gpr_params['normalize_y'])
    gp.fit(X_train,Y_train)
    Xt = X_t.reshape(-1)
    Yt = Y_t.reshape(-1)
    Y_pred_gp, Y_std_gp = gp.predict(X_t,return_std=True)
    Y_pred_gp = Y_pred_gp.reshape(-1)
    Y_global_error = Y_pred_gp-Yt
    gp_global_score = np.linalg.norm(Y_global_error)/float(Yt.shape[0])

    # plt.figure()
    # plt.rcParams.update({'font.size': 15})
    # plt.title('GP prediction')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.plot(Xt,Y_pred_gp,color='b', label='Predicted value')
    # plt.fill_between(Xt, Y_pred_gp - 1.96*Y_std_gp, Y_pred_gp + 1.96*Y_std_gp, alpha=0.2, color='b')
    # plt.plot(x_r,y_r,color='r', label='True function')
    # plt.legend()
    # plt.savefig('1d_syn_gp_fit.pdf')
    print 'Global GP score:', gp_global_score
    # plt.show()

# gmm clustering
if gmm_params['enable']:
    gmm = mixture.GaussianMixture(  n_components=gmm_params['K'],
                                    covariance_type='full',
                                    tol=1e-6,
                                    n_init=gmm_params['restarts'],
                                    warm_start=False,
                                    init_params='random',
                                    max_iter=1000)
    gmm.fit(XY_train)
    print 'Converged GMM',gmm.converged_, 'on', gmm.n_iter_, 'iterations with lower bound',gmm.lower_bound_
    gmm_idx = gmm.predict(XY_train)
    K = gmm.weights_.shape[0]

    # plot cluster weights
    # get color array for data
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    plt.figure()
    plt.bar(range(K),gmm.weights_,color=colors)
    plt.title('GMM cluster weights')

    # plot cluster distribution
    # get labels and counts
    labels, counts = zip(*sorted(Counter(gmm_idx).items(),key=operator.itemgetter(0)))
    colors_in_labels = colors[list(labels)]
    plt.figure()
    plt.bar(labels,counts,color=colors_in_labels)
    plt.title('GMM cluster dist')

    # plot clustered data
    col = np.zeros([XY_train.shape[0],3])
    for label in labels:
        col[(gmm_idx==label)] = colors[label]
    plt.figure()
    plt.scatter(X_train,Y_train, color=col)
    plt.plot(x_r,y_r,color='r')
    plt.title('GMM clustering')

    plt.show()
    raw_input()

# dpgmm clustering
if dpgmm_params['enable']:
    dpgmm = mixture.BayesianGaussianMixture(n_components=dpgmm_params['K'],
                                            covariance_type='full',
                                            tol=1e-6,
                                            n_init=dpgmm_params['restarts'],
                                            max_iter=1000,
                                            weight_concentration_prior_type='dirichlet_process',
                                            weight_concentration_prior=dpgmm_params['alpha'],
                                            mean_precision_prior=None,
                                            mean_prior=None, # None = x_bar
                                            degrees_of_freedom_prior=dpgmm_params['v0'],
                                            covariance_prior=None,
                                            warm_start=False,
                                            init_params='random'
                                            )
    dpgmm.fit(XY_train)
    print 'Converged DPGMM',dpgmm.converged_, 'on', dpgmm.n_iter_, 'iterations with lower bound',dpgmm.lower_bound_
    dpgmm_idx = dpgmm.predict(XY_train)
    K = dpgmm.weights_.shape[0]

    # plot cluster weights
    # get color array for data
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.
    # plt.figure()
    # plt.bar(range(K),dpgmm.weights_,color=colors)
    # plt.title('DPGMM cluster weights')

    # plot cluster distribution
    # get labels and counts
    labels, counts = zip(*sorted(Counter(dpgmm_idx).items(),key=operator.itemgetter(0)))
    colors_in_labels = colors[list(labels)]
    # plt.figure()
    # plt.bar(labels,counts,color=colors_in_labels)
    # plt.title('DPGMM cluster dist')

    # plot clustered data
    col = np.zeros([XY_train.shape[0],3])
    for label in labels:
        col[(dpgmm_idx==label)] = colors[label]
    plt.figure()
    plt.scatter(X_train,Y_train, color=col)
    plt.plot(x_r,y_r,color='r')
    plt.title('DPGMM clustering')

# Fit MoE
if moe_params['enable']:
    MoE = {}
    for label in labels:
        x_train = X_train[(dpgmm_idx==label)]
        y_train = Y_train[(dpgmm_idx==label)]
        kernel = gpr_params['K_RBF'] + gpr_params['K_W']
        gp = GaussianProcessRegressor(  kernel=kernel,
                                        alpha=gpr_params['alpha'],
                                        n_restarts_optimizer=gpr_params['restarts'],
                                        normalize_y=gpr_params['normalize_y'])
        gp.fit(x_train,y_train)
        MoE[label] = deepcopy(gp)
        del gp

    # predict with XY data
    dpgmm_xy_idx = dpgmm.predict(XY_t)
    gpc_test_idx=svm_test_idx=dpgmm_test_idx=gmm_test_idx=dpgmm_xy_idx # just initialization
    # train gpc gate
    if gpc_params['enable']:
        kernel = gpc_params['K_RBF']
        gpc = GaussianProcessClassifier(  kernel=kernel,
                                        n_restarts_optimizer=gpc_params['restarts'],)
        gpc_ds = gpc_params['down_sample']
        print 'GPC fit starting...', datetime.datetime.time(datetime.datetime.now())
        gpc.fit(X_train[::gpc_ds],dpgmm_idx[::gpc_ds])
        # gpc.fit(X_train,dpgmm_idx)
        print 'GPC fit ending...', datetime.datetime.time(datetime.datetime.now())
        print 'GPC test starting...', datetime.datetime.time(datetime.datetime.now())
        gpc_test_idx = gpc.predict(X_t)
        print 'GPC test ending...', datetime.datetime.time(datetime.datetime.now())

    # train svm gate
    if svm_params['enable']:
        svm_kernel = svm_params['kernel']
        svm_param_grid = svm_params['p_grid']
        svm_scoring = svm_params['scoring']
        svm_folds = svm_params['folds']

        print "Tuning hyper-parameters for svm", datetime.datetime.time(datetime.datetime.now())
        clf = GridSearchCV(SVC(decision_function_shape='ovr', tol=1e-06), svm_param_grid, cv=svm_folds, scoring=svm_scoring, n_jobs=-1, iid=False)
        clf.fit(X_train_std, dpgmm_idx)

        print"Best parameters set found on development set:", datetime.datetime.time(datetime.datetime.now())
        # print(clf.best_params_)
        # print()
        # print("Grid scores on development set:")
        # print()
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        svm_test_idx = clf.predict(X_t_std)
        total_correct = np.float(np.sum(dpgmm_xy_idx==svm_test_idx))
        total = np.float(len(dpgmm_xy_idx))
        print 'Gating score: ', total_correct/total*100.0

    # predict with dpgmm model marginalized data
    if dpgmm_params['dpgmm_inference_enable']:
        _, XY_n_features = XY_t.shape
        _, X_n_features = X_t.shape
        X_means = dpgmm.means_[:,:X_n_features]
        X_precisions_cholesky = dpgmm.precisions_cholesky_[:,:X_n_features,:X_n_features]
        X_covariance_type = dpgmm.covariance_type
        dof = dpgmm.degrees_of_freedom_ - XY_n_features + X_n_features
        X_mean_precision = dpgmm.mean_precision_
        X_t_ = XY_t[:,0].reshape(-1,1)
        log_gauss_ = (mixture.gaussian_mixture._estimate_log_gaussian_prob(
            X_t_, X_means, X_precisions_cholesky, X_covariance_type) -
            .5 * X_n_features * np.log(dof))
        log_lambda_ = X_n_features * np.log(2.) + np.sum(digamma(
            .5 * (dof - np.arange(0, X_n_features)[:, np.newaxis])), 0)
        log_prob_X_test = log_gauss_ + .5 * (log_lambda_ - X_n_features / X_mean_precision)
        weighted_log_prob_dpgmm = log_prob_X_test + dpgmm._estimate_log_weights()
        dpgmm_test_idx = weighted_log_prob_dpgmm.argmax(axis=1)
        weighted_log_prob_dpgmm_norm = weighted_log_prob_dpgmm - logsum(weighted_log_prob_dpgmm, axis=1)
        weighted_prob_dpgmm = np.exp(weighted_log_prob_dpgmm_norm)

    # predict with gmm model marginalized data
    if dpgmm_params['gmm_inference_enable']:
        weighted_log_prob_gmm = mixture.gaussian_mixture._estimate_log_gaussian_prob(
            X_t_, X_means, X_precisions_cholesky, X_covariance_type) + np.log(dpgmm.weights_.reshape(1,-1))
        gmm_test_idx = weighted_log_prob_gmm.argmax(axis=1)
        alpha = 5. # to scale the probabilities exponetially
        weighted_log_prob_gmm_norm = alpha*weighted_log_prob_gmm - logsum(alpha*weighted_log_prob_gmm, axis=1)
        weighted_prob_gmm = np.exp(weighted_log_prob_gmm_norm)

    # plot clustered test data
    colors = get_N_HexCol(K)
    colors = np.asarray(colors)/255.

    col = np.zeros([XY_test.shape[0],3])
    for label in labels:
        col[(dpgmm_xy_idx==label)] = colors[label]
    plt.figure()
    plt.scatter(X_t,Y_t, color=col)
    plt.plot(x_r,y_r,color='r')
    plt.title('DPGMM XY clustering')

    if dpgmm_params['gmm_inference_enable']:
        col = np.zeros([XY_test.shape[0],3])
        for label in labels:
            col[(gmm_test_idx==label)] = colors[label]
        plt.figure()
        plt.scatter(X_t,Y_t, color=col)
        plt.plot(x_r,y_r,color='r')
        plt.title('GMM X clustering')
    if dpgmm_params['dpgmm_inference_enable']:
        col = np.zeros([XY_test.shape[0],3])
        for label in labels:
            col[(dpgmm_test_idx==label)] = colors[label]
        plt.figure()
        plt.scatter(X_t,Y_t, color=col)
        plt.plot(x_r,y_r,color='r')
        plt.title('DPGMM X clustering')
    if gpc_params['enable']:
        col = np.zeros([XY_test.shape[0],3])
        for label in labels:
            col[(gpc_test_idx==label)] = colors[label]
        plt.figure()
        plt.scatter(X_t,Y_t, color=col)
        plt.plot(x_r,y_r,color='r')
        plt.title('GPC X clustering')
    if svm_params['enable']:
        col = np.zeros([XY_test.shape[0],3])
        for label in labels:
            col[(svm_test_idx==label)] = colors[label]
        plt.figure()
        plt.scatter(X_t,Y_t, color=col)
        plt.plot(x_r,y_r,color='r')
        plt.title('SVM X clustering')



    Nd = XY_t.shape[0]
    # assert(Nd==len(dpgmm_xy_idx)==len(gmm_test_idx)==len(dpgmm_test_idx)==len(gpc_test_idx)==len(svm_test_idx))

    Y_std_dpxy = np.zeros(Nd)
    Y_pred_dpxy = np.zeros(Nd)
    Y_pred_mix_dpxy = np.zeros(Nd)
    Y_std_gmm = np.zeros(Nd)
    Y_pred_gmm = np.zeros(Nd)
    Y_pred_mix_gmm = np.zeros(Nd)
    Y_std_dpgmm = np.zeros(Nd)
    Y_pred_dpgmm = np.zeros(Nd)
    Y_pred_mix_dpgmm = np.zeros(Nd)
    Y_std_gpc = np.zeros(Nd)
    Y_pred_gpc = np.zeros(Nd)
    Y_pred_mix_gpc = np.zeros(Nd)
    Y_std_svm = np.zeros(Nd)
    Y_pred_svm = np.zeros(Nd)
    Y_pred_mix_svm = np.zeros(Nd)
    moe_dpxy_score = 0
    moe_gmm_score = 0
    moe_dpgmm_score = 0
    moe_gpc_score = 0
    moe_svm_score = 0
    for i in range(Nd):
        label_dpxy = dpgmm_xy_idx[i]
        gp_dpxy = MoE[label_dpxy]
        Y_pred_dpxy[i], Y_std_dpxy[i] = gp_dpxy.predict(X_t[i].reshape(1,1),return_std=True)
        if dpgmm_params['gmm_inference_enable']:
            label_gmm = gmm_test_idx[i]
            gp_gmm = MoE[label_gmm]
            Y_pred_gmm[i], Y_std_gmm[i] = gp_gmm.predict(X_t[i].reshape(1,1),return_std=True)
        if dpgmm_params['dpgmm_inference_enable']:
            label_dpgmm = dpgmm_test_idx[i]
            gp_dpgmm = MoE[label_dpgmm]
            Y_pred_dpgmm[i], Y_std_dpgmm[i] = gp_dpgmm.predict(X_t[i].reshape(1,1),return_std=True)
        if gpc_params['enable']:
            label_gpc = gpc_test_idx[i]
            gp_gpc = MoE[label_gpc]
            Y_pred_gpc[i], Y_std_gpc[i] = gp_gpc.predict(X_t[i].reshape(1,1),return_std=True)
        if svm_params['enable']:
            label_svm = svm_test_idx[i]
            gp_svm = MoE[label_svm]
            Y_pred_svm[i], Y_std_svm[i] = gp_svm.predict(X_t[i].reshape(1,1),return_std=True)

        # for label in labels:
        #     gp = MoE[label]
        #     Y_pred_t, Y_std_t = gp.predict(X_test[i].reshape(1,1),return_std=True)
        #     Y_pred_mix[i] += weighted_prob[i][label]*Y_pred_t
    Y_moe_dpxy_error = Y_pred_dpxy.reshape(-1,1)-Y_t.reshape(-1,1)
    Y_moe_gmm_error = Y_pred_gmm.reshape(-1,1)-Y_t.reshape(-1,1)
    Y_moe_dpgmm_error = Y_pred_dpgmm.reshape(-1,1)-Y_t.reshape(-1,1)
    Y_moe_gpc_error = Y_pred_gpc.reshape(-1,1)-Y_t.reshape(-1,1)
    Y_moe_svm_error = Y_pred_svm.reshape(-1,1)-Y_t.reshape(-1,1)
    # Y_moe_error_mix = Y_pred_mix.reshape(-1,1)-Y_test.reshape(-1,1)
    moe_dpxy_score = np.linalg.norm(Y_moe_dpxy_error)/float(Y_t.shape[0])
    moe_gmm_score = np.linalg.norm(Y_moe_gmm_error)/float(Y_t.shape[0])
    moe_dpgmm_score = np.linalg.norm(Y_moe_dpgmm_error)/float(Y_t.shape[0])
    moe_gpc_score = np.linalg.norm(Y_moe_gpc_error)/float(Y_t.shape[0])
    moe_svm_score = np.linalg.norm(Y_moe_svm_error)/float(Y_t.shape[0])
    # moe_gp_score_mix = np.linalg.norm(Y_moe_error_mix)/float(Y_test.shape[0])

    X_t = X_t.reshape(-1)
    Y_pred_dpxy = Y_pred_dpxy.reshape(-1)
    Y_pred_gmm = Y_pred_gmm.reshape(-1)
    Y_pred_dpgmm = Y_pred_dpgmm.reshape(-1)
    Y_pred_gpc = Y_pred_gpc.reshape(-1)
    Y_pred_svm = Y_pred_svm.reshape(-1)
    Y_std_dpxy = Y_std_dpxy.reshape(-1)
    Y_std_gmm = Y_std_gmm.reshape(-1)
    Y_std_dpgmm = Y_std_dpgmm.reshape(-1)
    Y_std_gpc = Y_std_gpc.reshape(-1)
    Y_std_svm = Y_std_svm.reshape(-1)


    plt.figure()
    plt.title('MoE prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(X_t.reshape(-1),Y_pred_gp,color='r', label='gp')
    plt.fill_between(X_t, Y_pred_gp - 1.96*Y_std_gp, Y_pred_gp + 1.96*Y_std_gp, alpha=0.1, color='r')
    plt.plot(X_t,Y_pred_dpxy,color='c',label='dpxy')
    plt.fill_between(X_t, Y_pred_dpxy - 1.96*Y_std_dpxy, Y_pred_dpxy + 1.96*Y_std_dpxy, alpha=0.1, color='c')
    print 'DPXY improvement ', (gp_global_score - moe_dpxy_score)/gp_global_score*100.
    if dpgmm_params['gmm_inference_enable']:
        plt.plot(X_t,Y_pred_gmm,color='b',label='gmm')
        plt.fill_between(X_t, Y_pred_gmm - 1.96*Y_std_gmm, Y_pred_gmm + 1.96*Y_std_gmm, alpha=0.1, color='b')
        print 'GMM improvement ', (gp_global_score - moe_gmm_score)/gp_global_score*100.
    if dpgmm_params['dpgmm_inference_enable']:
        plt.plot(X_t,Y_pred_dpgmm,color='g',label='dpgmm')
        plt.fill_between(X_t, Y_pred_dpgmm - 1.96*Y_std_dpgmm, Y_pred_dpgmm + 1.96*Y_std_dpgmm, alpha=0.1, color='g')
        print 'DPGMM improvement ', (gp_global_score - moe_dpgmm_score)/gp_global_score*100.
    if gpc_params['enable']:
        plt.plot(X_t,Y_pred_gpc,color='m',label='gpc')
        plt.fill_between(X_t, Y_pred_gpc - 1.96*Y_std_gpc, Y_pred_gpc + 1.96*Y_std_gpc, alpha=0.1, color='m')
        print 'GPC improvement ', (gp_global_score - moe_gpc_score)/gp_global_score*100.
    if svm_params['enable']:
        plt.plot(X_t,Y_pred_svm,color='y',label='svm')
        plt.fill_between(X_t, Y_pred_svm - 1.96*Y_std_svm, Y_pred_svm + 1.96*Y_std_svm, alpha=0.1, color='y')
        print 'SVM improvement ', (gp_global_score - moe_svm_score)/gp_global_score*100.
    plt.plot(x_r,y_r,color='k', label='truth')
    plt.legend()

    # plt.savefig('1d_syn_gp_me_comparison.pdf')

plt.show()
