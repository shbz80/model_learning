from __future__ import print_function
import numpy as np
import scipy as sp
from scipy.linalg import cholesky
import colorsys
import time
# from multidim_gp import MultidimGP
# from multidim_gp import MdGpyGP as MultidimGP
from multidim_gp import MdGpyGPwithNoiseEst as MultidimGP
from itertools import compress
from copy import deepcopy
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import copy
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
import operator
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
# from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg


'''
usage:

ugp_params = {
    'alpha': 1.,
    'kappa': 2.,
    'beta': 0.,
}
ugp = UGP(dX + dU, **ugp_params)
y_mu, y_var, _, _, xy_cor = ugp.get_posterior(gp, x_mu, x_var)

* gp should have predict method according to the scikit learn lib
'''
yumi_joint_limits = [
    (-2.9234, 2.9234),
    (-2.4870, 0.7417),
    (-2.9234, 2.9234),
    (-2.1380, 1.3788),
    (-5.0440, 5.0440),
    (-1.5184, 2.3911),
    (-3.9793, 3.9793),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-2*np.pi, 2*np.pi),
    (-2*np.pi, 2*np.pi),
    (-2*np.pi, 2*np.pi),
]

jitter_val = 1e-6

class UGP(object):
    def __init__(self, L, alpha=1e-3, kappa=0., beta=2.):
        '''
        Initialize the unscented transform parameters
        :param L: dim of the input
        a practical value set for the remaining params:
        :param alpha: 1.
        :param kappa: 2.
        :param beta: 0.
        '''
        self.L = L
        self.N = 2*L + 1
        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta


    def get_sigma_points(self, mu, var):
        '''
        generate and return 2L+1 sigma points along with their weights
        :param mu: mean of input
        :param var: variance of output
        :return:
        '''
        L = self.L
        N = self.N
        assert(mu.shape == (L,))
        assert (var.shape == (L, L))
        alpha = self.alpha
        kappa = self.kappa
        beta = self.beta
        sigmaMat = np.zeros((N, L))

        Lambda = (alpha**2) * (L + kappa) - L
        sigmaMat[0, :] = mu
        n, n = var.shape
        var = var + np.eye(n,n)*1e-6 # TODO: is this ok?
        try:
            chol = cholesky((L + Lambda)*var, lower=True)
        except np.linalg.LinAlgError:
            print('Cholesky failure')
            assert(False)
        for i in range(1, L+1):
            sigmaMat[i, :] = mu + chol[:,i-1]
            sigmaMat[L+i, :] = mu - chol[:, i-1]

        W_mu = np.zeros(N)
        Wi = 1. / (2. * (L + Lambda))
        W_mu.fill(Wi)
        W_mu[0] = Lambda / (L + Lambda)

        W_var = np.zeros(N)
        W_var.fill(Wi)
        W_var[0] = W_mu[0] + (1. - alpha**2 + beta)
        return sigmaMat, W_mu, W_var

    def get_posterior(self, fn, mu, var, t=None):
        '''
        Compute and return the output distribution along with the propagated sigma points
        :param fn: the nonlinear function through which to propagate
        :param mu: mean of the input
        :param var: variance of the output
        :return:
                Y_mu_post: output mean
                Y_var_post: output variance
                Y_mu: transformed sigma points
                Y_var: gp var for each transformed points
                XY_cross_cov: cross covariance between input and output
        '''
        sigmaMat, W_mu, W_var = self.get_sigma_points(mu, var)
        if t is None:
            Y_mu, Y_std = fn.predict(sigmaMat, return_std=True) # same signature as the predict function of gpr but can be
                                                            # any nonlinear function
        else:
            Y_mu, Y_std = fn.predict(sigmaMat, t,
                                     return_std=True)  # same signature as the predict function of gpr but can be
            # any nonlinear function
        N, Do = Y_mu.shape
        Y_var = Y_std **2
        Y_var = Y_var.reshape(N,Do)
        Y_mu_post = np.average(Y_mu, axis=0, weights=W_mu)    # DX1
        # Y_mu_post = Y_mu[0]
        Y_var_post = np.zeros((Do,Do))
        for i in range(N):
           y = Y_mu[i] - Y_mu_post
           yy_ = np.outer(y, y)
           Y_var_post += W_var[i]*yy_
           # yy_ = np.square(y)
           # Y_var_post += np.diag(W_var[i] * yy_)
        #Y_var_post = np.diag(np.diag(Y_var_post))     # makes it worse
        Y_var_post += np.diag(Y_var[0])
        a = np.zeros((Do, Do))
        np.fill_diagonal(a, 1e-6)
        # Y_var_post += a    #TODO: verify this
        if Do == 1:
            Y_mu_post = np.asscalar(Y_mu_post)
            Y_var_post = np.asscalar(Y_var_post)
        # compute cross covariance between input and output
        Di = mu.shape[0]
        XY_cross_cov = np.zeros((Di, Do))
        #TODO: XY_cross_cov may be wrong
        for i in range(N):
            y = Y_mu[i] - Y_mu_post
            x = sigmaMat[i] - mu
            xy_ = np.outer(x, y)
            XY_cross_cov += W_var[i] * xy_
        return Y_mu_post, Y_var_post, Y_mu, Y_var, XY_cross_cov

    def get_posterior_bnn(self, fn, mu, var):
        '''
        Compute and return the output distribution along with the propagated sigma points
        :param fn: the nonlinear function through which to propagate
        :param mu: mean of the input
        :param var: variance of the output
        :return:
                Y_mu_post: output mean
                Y_var_post: output variance
                Y_mu: transformed sigma points
                Y_var: gp var for each transformed points
                XY_cross_cov: cross covariance between input and output
        '''
        sigmaMat, W_mu, W_var = self.get_sigma_points(mu, var)
        Y_mu, Y_var = fn.predict(sigmaMat, factored=False) # same signature as the predict function of gpr but can be
        N, Do = Y_mu.shape
        Y_var = Y_var.reshape(N,Do)
        Y_mu_post = np.average(Y_mu, axis=0, weights=W_mu)    # DX1
        # Y_mu_post = Y_mu[0]
        Y_var_post = np.zeros((Do,Do))
        for i in range(N):
           y = Y_mu[i] - Y_mu_post
           yy_ = np.outer(y, y)
           Y_var_post += W_var[i]*yy_
        Y_var_post += np.diag(Y_var[0])
        # a = np.zeros((Do, Do))
        # np.fill_diagonal(a, 1e-6)
        # Y_var_post += a    #TODO: verify this
        if Do == 1:
            Y_mu_post = np.asscalar(Y_mu_post)
            Y_var_post = np.asscalar(Y_var_post)
        # compute cross covariance between input and output
        Di = mu.shape[0]
        XY_cross_cov = np.zeros((Di, Do))
        #TODO: XY_cross_cov may be wrong
        for i in range(N):
            y = Y_mu[i] - Y_mu_post
            x = sigmaMat[i] - mu
            xy_ = np.outer(x, y)
            XY_cross_cov += W_var[i] * xy_
        return Y_mu_post, Y_var_post, Y_mu, Y_var, XY_cross_cov


class dummySVM(object):
    def __init__(self, label):
        self.label = label

    def predict(self, ip):
        return np.full(ip.shape[0],self.label)


def get_N_HexCol(N=5):

    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    # hex_out = []
    rgb_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
        # hex_out.append("".join(map(lambda x: chr(x).encode('hex'),rgb)))
        rgb_out.append(list(rgb))
    return rgb_out

def train_trans_models(gp_param_list, XUs_t, labels_t, dX, dU):
    '''
    Trains the GP based transition models. To be moved out of this file
    :param gp_param_list:
    :param XUs_t:
    :param labels_t:
    :param dX:
    :param dU:
    :return:
    '''
    trans_dicts = {}
    for i in range(XUs_t.shape[0]):
        xu = XUs_t[i]
        x_labels = labels_t[i]
        iddiff = x_labels[:-1] != x_labels[1:]
        trans_data = zip(xu[:-1, :dX + dU], xu[1:, :dX], x_labels[:-1], x_labels[1:])
        trans_data_p = list(compress(trans_data, iddiff))
        for xu_, y, xid, yid in trans_data_p:
            if (xid, yid) not in trans_dicts:
                trans_dicts[(xid, yid)] = {'XU': [], 'Y': [], 'mdgp': None}
            trans_dicts[(xid, yid)]['XU'].append(xu_)
            trans_dicts[(xid, yid)]['Y'].append(y)
    for trans_data in trans_dicts:
        XU = np.array(trans_dicts[trans_data]['XU']).reshape(-1, dX + dU)
        Y = np.array(trans_dicts[trans_data]['Y']).reshape(-1, dX)
        mdgp = MultidimGP(gp_param_list, Y.shape[1])
        mdgp.fit(XU, Y)
        trans_dicts[trans_data]['mdgp'] = deepcopy(mdgp)
        del mdgp
    return trans_dicts

class SVMmodePrediction(object):
    def __init__(self, svm_grid_params, svm_params):
        self.svm_grid_params = svm_grid_params
        self.svm_params = svm_params

    def train(self, XUs_t, labels_t, labels):
        '''
        Trains SVMs for each cluster. To be moved out of this file
        :param svm_grid_params:
        :param svm_params:
        :param XU_t:
        :param labels_t:
        :return:
        '''
        start_time = time.time()

        XU_t = XUs_t.reshape(-1, XUs_t.shape[-1])
        self.scaler = StandardScaler().fit(XU_t)
        XU_t_std = self.scaler.transform(XU_t)
        self.XUs_t_std = XU_t_std.reshape(XUs_t.shape)
        # joint space SVM
        SVMs = {}
        XUnI_svm = []
        labels_t_svm = []
        for i in range(self.XUs_t_std.shape[0]):
            xu_t = self.XUs_t_std[i]
            labels_t_ = labels_t[i]
            labels_t_svm.extend(labels_t_[:-1])
            xuni = zip(xu_t[:-1, :], labels_t_[1:])
            XUnI_svm.extend(xuni)
        labels_t_svm = np.array(labels_t_svm)
        for label in labels:
            xui = list(compress(XUnI_svm, (labels_t_svm == label)))
            xu, i = zip(*xui)
            xu = np.array(xu)
            i = list(i)
            cnts_list = Counter(i).items()
            svm_check_ok = True
            for cnts in cnts_list:
                if cnts[1] < self.svm_grid_params['cv']:
                    svm_check_ok = False  # TODO: this check is disabled.
            if len(cnts_list) > 1 and svm_check_ok == True:
                clf = GridSearchCV(SVC(**self.svm_params), **self.svm_grid_params)
                clf.fit(xu, i)
                SVMs[label] = deepcopy(clf)
                del clf
            else:
                print ('detected dummy svm:', label)
                dummy_svm = dummySVM(cnts_list[0][0])
                SVMs[label] = deepcopy(dummy_svm)
                del dummy_svm
        print ('SVMs training time:', time.time() - start_time)
        self.SVMs = SVMs


    def predict(self, XU, i):
        XU_std = self.scaler.transform(XU)
        svm = self.SVMs[i]
        next_modes = svm.predict(XU_std)
        return next_modes

class SVMmodePredictionGlobal(object):
    def __init__(self, svm_grid_params, svm_params):
        self.svm_grid_params = svm_grid_params
        self.svm_params = svm_params

    def train(self, XUs_t, labels_t, labels):
        '''
        Trains SVMs for each cluster. To be moved out of this file
        :param svm_grid_params:
        :param svm_params:
        :param XU_t:
        :param labels_t:
        :return:
        '''

        XU_t = XUs_t.reshape(-1, XUs_t.shape[-1])
        self.scaler = StandardScaler().fit(XU_t)
        XU_t_std = self.scaler.transform(XU_t)
        self.XUs_t_std = XU_t_std.reshape(XUs_t.shape)
        # joint space SVM
        XUnI_svm = []
        for i in range(self.XUs_t_std.shape[0]):
            xu_t = self.XUs_t_std[i]
            labels_t_ = labels_t[i]
            xuni = zip(xu_t[:-1, :], labels_t_[1:])
            XUnI_svm.extend(xuni)
        xu, i = zip(*XUnI_svm)
        xu = np.array(xu)
        i = list(i)
        clf = GridSearchCV(SVC(**self.svm_params), **self.svm_grid_params)
        clf.fit(xu, i)
        self.svm = copy.deepcopy(clf)


    def predict(self, XU, i=None):
        XU_std = self.scaler.transform(XU)
        next_modes = self.svm.predict(XU_std)
        return next_modes

class SVMmodePredictionGlobalME(SVMmodePredictionGlobal):
    def train(self, XUs_t, labels_t, labels):
        '''
        Trains SVMs for each cluster. To be moved out of this file
        :param svm_grid_params:
        :param svm_params:
        :param XU_t:
        :param labels_t:
        :return:
        '''

        XU_t = XUs_t.reshape(-1, XUs_t.shape[-1])
        self.scaler = StandardScaler().fit(XU_t)
        XU_t_std = self.scaler.transform(XU_t)
        self.XUs_t_std = XU_t_std.reshape(XUs_t.shape)
        # joint space SVM
        XUnI_svm = []
        for i in range(self.XUs_t_std.shape[0]):
            xu_t = self.XUs_t_std[i]
            labels_t_ = labels_t[i]
            xuni = zip(xu_t, labels_t_)
            XUnI_svm.extend(xuni)
        xu, i = zip(*XUnI_svm)
        xu = np.array(xu)
        i = list(i)
        clf = GridSearchCV(SVC(**self.svm_params), **self.svm_grid_params)
        clf.fit(xu, i)
        self.svm = copy.deepcopy(clf)

def print_global_gp(global_gp, file):
    print('Global GP params', file=file)
    gps = global_gp.gp_list
    for i in range(len(gps)):
        gp = gps[i]
        print('Output dim', i, file=file)
        print(gp.rbf.variance, file=file)
        print(gp.rbf.lengthscale, file=file)
        print(gp.Gaussian_noise.variance, file=file)

def print_experts_gp(experts_gp, file):
    print('Experts GP params', file=file)
    for e in experts_gp:
        print('Expert', e, file=file)
        gps = experts_gp[e].gp_list
        for i in range(len(gps)):
            print('Output dim', i, file=file)
            gp = gps[i]
            print(gp.rbf.variance, file=file)
            print(gp.rbf.lengthscale, file=file)
            print(gp.Gaussian_noise.variance, file=file)

def print_transition_gp(transition_gp, file):
    print('Trans GP params', file=file)
    for t in transition_gp:
        print('Trans gp', t, file=file)
        gps = transition_gp[t]['mdgp'].gp_list
        for i in range(len(gps)):
            print('Output dim', i, file=file)
            gp = gps[i]
            print(gp.rbf.variance, file=file)
            print(gp.rbf.lengthscale, file=file)
            print(gp.Gaussian_noise.variance, file=file)

def obtian_joint_space_policy(params, xus, x_init):
    kp = params['kp']
    kd = params['kd']
    dX = params['dX']
    dP = params['dP']
    dV = params['dV']
    dU = params['dU']
    dt = params['dt']
    assert(dX==(dP+dV))
    N, T, _ = xus.shape
    xrs = np.zeros((N, T, dX))
    q_init = x_init[:dP]
    for n in range(N):
        xu = xus[n]
        u = xu[:,dX:dX+dU]
        x = xu[:, :dX]
        q = x[:, :dP]
        qd = x[:, dP:dP+dV]
        qr_t_ = q_init
        qr = np.zeros((T,dP))
        qrd = np.zeros((T,dV))
        for t in range(T):
            qrd[t] = (u[t] - kp*(qr_t_ - q[t]) + kd*qd[t])/(kp*dt + kd)
            qr[t] = qr_t_ + qrd[t]*dt
            qr_t_ = qr[t]
        xrs[n] = np.concatenate((qr, qrd), axis=1)
    return xrs

class DPGMMCluster(object):
    def __init__(self, params, params_extra, X):
        self.dpgmm = mixture.BayesianGaussianMixture(**params)
        self.params = params
        self.params_extra = params_extra
        if params_extra['standardize'] == True:
            self.scaler = StandardScaler().fit(X)
            self.X_std = self.scaler.transform(X)
        else:
            self.X_std = X

    def cluster(self):
        self.dpgmm.fit(self.X_std)
        print('Converged DPGMM', self.dpgmm.converged_, 'on', self.dpgmm.n_iter_,
              'iterations with lower bound', self.dpgmm.lower_bound_)
        y = self.dpgmm.predict(self.X_std)
        labels, counts = zip(*sorted(Counter(y).items(), key=operator.itemgetter(0)))
        min_clust_size = self.params_extra['min_clust_size']
        vbgmm_refine = self.params_extra['vbgmm_refine']
        if vbgmm_refine:
            selected_k_idx = list(np.where(np.array(counts) > min_clust_size)[0])
            K = len(selected_k_idx)
            vbgmm_params = self.params
            vbgmm_params['weight_concentration_prior_type'] = 'dirichlet_distribution'
            self.vbgmm = mixture.BayesianGaussianMixture(**vbgmm_params)
            dpgmm_params = self.dpgmm._get_parameters()
            self.vbgmm.converged_ = False
            self.vbgmm.lower_bound_ = -np.infty
            _, log_resp = self.dpgmm._e_step(self.X_std)
            nk, xk, sk = mixture.gaussian_mixture._estimate_gaussian_parameters(self.X_std, np.exp(log_resp),
                                                                                self.dpgmm.reg_covar,
                                                                                self.dpgmm.covariance_type)

            vbgmm_params = ((self.dpgmm.weight_concentration_prior_ + nk)[selected_k_idx],  # weight_concentration_
                            dpgmm_params[1][selected_k_idx],  # mean_precision_
                            dpgmm_params[2][selected_k_idx],  # means_
                            dpgmm_params[3][selected_k_idx],  # degrees_of_freedom_
                            dpgmm_params[4][selected_k_idx],  # covariances_
                            dpgmm_params[5][selected_k_idx])  # precisions_cholesky_

            self.vbgmm._set_parameters(vbgmm_params)
            self.vbgmm.covariances_ /= (self.vbgmm.degrees_of_freedom_[:, np.newaxis, np.newaxis])
            start_time = time.time()
            self.vbgmm.fit(self.X_std)
            print('Converged VBGMM', self.vbgmm.converged_, 'on', self.vbgmm.n_iter_, 'iterations with lower bound', self.vbgmm.lower_bound_)
            y = self.vbgmm.predict(self.X_std)
            labels, counts = zip(*sorted(Counter(y).items(), key=operator.itemgetter(0)))

        if self.params_extra['min_size_filter']:
            if vbgmm_refine:
                log_prob = self.vbgmm._estimate_weighted_log_prob(self.X_std)
            else:
                log_prob = self.dpgmm._estimate_weighted_log_prob(self.X_std)
            clust_discard = list(compress(zip(labels, counts), np.array(counts)<min_clust_size))
            label_discard, count_discard = zip(*clust_discard)
            for (label, count) in clust_discard:
                array_idx_label = (y == label)
                log_prob_label = log_prob[array_idx_label]
                reassigned_labels = np.zeros(log_prob_label.shape[0], dtype=int)
                for j in range(log_prob_label.shape[0]):
                    sorted_idx = np.argsort(log_prob_label[j, :])
                    for k in range(-2, -(len(sorted_idx)+1), -1):
                        if int(sorted_idx[k]) not in label_discard:
                            reassigned_labels[j] = int(sorted_idx[k])
                            break
                y[array_idx_label] = reassigned_labels
            y = np.array(y)
        n_train = self.params_extra['n_train']
        ys = y.reshape(n_train, -1)
        T = ys.shape[1]
        if self.params_extra['seg_filter']:
            for n in range(n_train):
                for t in range(T):
                    if t == 0:
                        l = ys[n:n + 1, t:t + 1]
                        l_n = ys[n:n + 1, t + 1:t + 2]
                        l_nn = ys[n:n + 1, t + 2:t + 3]
                        if l != l_n:
                            ys[n:n + 1, t:t + 1] = l_n
                        if l == l_n and l_n != l_nn:
                            ys[n:n + 1, t:t + 1] = l_nn
                            ys[n:n + 1, t+1:t + 2] = l_nn
                    elif t == T - 1:
                        l = ys[n:n + 1, t:t + 1]
                        l_p = ys[n:n + 1, t - 1:t]
                        if l != l_p:
                            ys[n:n + 1, t:t + 1] = l_p
                    else:
                        l = ys[n:n + 1, t:t + 1]
                        l_n = ys[n:n + 1, t + 1:t + 2]
                        l_nn = ys[n:n + 1, t + 2:t + 3]
                        l_p = ys[n:n + 1, t - 1:t]
                        if l != l_n and l_p != l and l_p == l_n:
                            ys[n:n + 1, t:t + 1] = l_n
                        if l == l_n and l_p != l and l_n != l_nn and l_p == l_nn:
                            ys[n:n + 1, t:t + 1] = l_nn
                            ys[n:n + 1, t+1:t + 2] = l_nn
            for n in range(n_train):
                for t in range(T):
                    if t == 0:
                        l = ys[n:n + 1, t:t + 1]
                        l_n = ys[n:n + 1, t + 1:t + 2]
                        l_nn = ys[n:n + 1, t + 2:t + 3]
                        if l != l_n:
                            ys[n:n + 1, t:t + 1] = l_n
                            ys[n:n + 1, t + 1:t + 2] = l_nn
                    elif t == T - 1:
                        l = ys[n:n + 1, t:t + 1]
                        l_p = ys[n:n + 1, t - 1:t]
                        if l != l_p:
                            ys[n:n + 1, t:t + 1] = l_p
                    else:
                        l = ys[n:n + 1, t:t + 1]
                        l_n = ys[n:n + 1, t + 1:t + 2]
                        l_p = ys[n:n + 1, t - 1:t]
                        if l != l_n and l_p != l:
                            ys[n:n + 1, t:t + 1] = l_n
            y = y.reshape(-1)
            # for i in range(len(counts)):
            #     if counts[i] < min_clust_size:
            #         array_idx_label = (y == labels[i])
            #         log_prob_label = log_prob[array_idx_label]
            #         reassigned_labels = np.zeros(log_prob_label.shape[0])
            #         for j in range(log_prob_label.shape[0]):
            #             sorted_idx = np.argsort(log_prob_label[j, :])
            #             reassigned_labels[j] = int(sorted_idx[-2])
            #         y[array_idx_label] = reassigned_labels
        labels, counts = zip(*sorted(Counter(y).items(), key=operator.itemgetter(0)))
        return y, labels, counts

    def predict(self, X):
        if hasattr(self, 'scaler'):
            X_std = self.scaler.transform(X)
        else:
            X_std = X
        if hasattr(self, 'vbgmm'):
            return self.vbgmm.predict(X_std)
        else:
            return self.dpgmm.predict(X_std)

def get_ee_points(offsets, ee_pos, ee_rot):
    """
    Helper method for computing the end effector points given a
    position, rotation matrix, and offsets for each of the ee points.

    Args:
        offsets: N x 3 array where N is the number of points.
        ee_pos: 1 x 3 array of the end effector position.
        ee_rot: 3 x 3 rotation matrix of the end effector.
    Returns:
        3 x N array of end effector points.
    """
    rotated = ee_rot.dot(offsets.T)
    translated = rotated + ee_pos.T
    return translated
    # return ee_rot.dot(offsets.T) + ee_pos.T

def logsum(vec, axis=0, keepdims=True):
    #TODO: Add a docstring.
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec-maxv), axis=axis, keepdims=keepdims)) + maxv

class traj_with_moe(object):
    def __init__(self, sim_data_tree, experts, trans_dicts, massSlideWorld, dlt_mdl=False):
        self.sim_data_tree = sim_data_tree
        self.experts = experts
        self.trans_dicts = trans_dicts
        self.massSlideWorld = massSlideWorld
        self.dlt_mdl = dlt_mdl

    def sample(self, num_samples, H):
        sim_data_tree = self.sim_data_tree
        experts = self.experts
        trans_dicts = self.trans_dicts
        massSlideWorld = self.massSlideWorld

        dX = sim_data_tree[0][0][2].shape[0]
        sample_trajs = np.zeros((num_samples, H, dX))
        for s in range(num_samples):
            tracks_0 = sim_data_tree[0]
            assert (len(tracks_0) == 1)
            curr_track = tracks_0[0]
            mu = curr_track[2]
            var = curr_track[3]
            sample_trajs[s][0] = np.random.multivariate_normal(mu, var)
            for t in range(1, H):
                tracks = sim_data_tree[t]
                curr_mode = curr_track[0]
                l = len(tracks)
                w_list = []
                id_list = []
                for i in range(l):
                    track = tracks[i]
                    if (track[1]==curr_mode or track[0]==curr_mode):
                        w_list.append(track[6])
                        id_list.append(i)
                w_sum = np.sum(w_list)
                w_list = list(np.array(w_list)/w_sum)
                id_sel = np.random.choice(id_list, p=w_list)
                next_track = tracks[id_sel]
                x = sample_trajs[s][t-1]
                x = x.reshape(-1)
                um, uv = massSlideWorld.predict(x.reshape(1,-1))
                um = np.asscalar(um)
                uv = np.asscalar(uv)
                u = np.random.normal(um, uv)
                xu = np.append(x, u)
                if next_track[1] == curr_track[0]:
                    gp = trans_dicts[(curr_track[0], next_track[0])]['mdgp']
                    mu, std = gp.predict(xu.reshape(1, -1))
                    mu = mu.reshape(-1)
                    std = std.reshape(-1) ** 2
                    var = np.diag(std)
                    sample_trajs[s][t] = np.random.multivariate_normal(mu, var)
                elif next_track[0] == curr_track[0]:
                    gp = experts[next_track[0]]
                    mu, std = gp.predict(xu.reshape(1, -1))
                    mu = mu.reshape(-1)
                    std = std.reshape(-1) ** 2
                    var = np.diag(std)
                    if not self.dlt_mdl:
                        sample_trajs[s][t] = np.random.multivariate_normal(mu, var)
                    else:
                        sample_trajs[s][t] = np.random.multivariate_normal(mu, var) + sample_trajs[s][t - 1]
                curr_track = next_track
        self.sample_trajs = sample_trajs
        return sample_trajs

    def plot_samples(self):
        H = self.sample_trajs.shape[1]
        sample_trajs = self.sample_trajs
        tm = range(H)
        plt.figure()
        plt.subplot(121)
        plt.title('Position')
        plt.plot(tm, sample_trajs[:, :, 0].T, color='g', alpha=0.1, linewidth=1)
        # for sample_traj in sample_trajs:
        #     plt.scatter(tm, sample_traj[:, 0], color='g', alpha=0.01)
        plt.subplot(122)
        plt.title('Velocity')
        plt.plot(tm, sample_trajs[:, :, 1].T, color='b', alpha=0.1, linewidth=1)
        # for sample_traj in sample_trajs:
        #     plt.scatter(tm, sample_traj[:, 1], color='b', alpha=0.01)
        plt.show(block=False)

    def plot_gmm_traj(self, Xs_t_test):
        K = self.params['n_components']
        dX = self.sample_trajs.shape[2]
        n_test, H, _ = Xs_t_test.shape
        traj_density = self.traj_density

        traj_means = np.zeros((H, K, dX + 1))
        traj_stds = np.zeros((H, K, dX + 1))
        for t in range(H):
            for k in range(K):
                traj_means[t, k, :dX] = traj_density[t][1][k]
                traj_means[t, k, dX:] = traj_density[t][0][k]
                traj_stds[t, k, :dX] = np.sqrt(np.diag(traj_density[t][2][k]))
                traj_stds[t, k, dX:] = traj_density[t][0][k]
        tm = range(H)
        plt.figure()
        plt.subplot(121)
        plt.title('Position')
        plt.subplot(122)
        plt.title('Velocity')
        for k in range(K):
            if k == 0:
                cl = 'g'
            elif k == 1:
                cl = 'b'
            plt.subplot(121)
            prob = traj_means[:, k, 2]
            rbg_g = plt_colors.to_rgba(cl)
            rbg_col = np.tile(rbg_g, (H, 1))
            rbg_col[:, 3] = prob.reshape(-1)
            plt.scatter(tm, traj_means[:, k, 0], color=rbg_col)
            plt.scatter(tm, traj_means[:, k, 0] + 1.96 * traj_stds[:, k, 0], color=rbg_col, marker='_')
            plt.scatter(tm, traj_means[:, k, 0] - 1.96 * traj_stds[:, k, 0], color=rbg_col, marker='_')
            plt.plot(tm, traj_means[:, k, 0], color='k', alpha=0.3)
            plt.subplot(122)
            plt.title('Velocity')
            rbg_g = plt_colors.to_rgba(cl)
            rbg_col = np.tile(rbg_g, (H, 1))
            rbg_col[:, 3] = prob.reshape(-1)
            plt.scatter(tm, traj_means[:, k, 1], color=rbg_col)
            plt.scatter(tm, traj_means[:, k, 1] + 1.96 * traj_stds[:, k, 1], color=rbg_col, marker='_')
            plt.scatter(tm, traj_means[:, k, 1] - 1.96 * traj_stds[:, k, 1], color=rbg_col, marker='_')
            plt.plot(tm, traj_means[:, k, 1], color='k', alpha=0.3)
        plt.subplot(121)
        for i in range(0, n_test):
            plt.plot(tm, Xs_t_test[i, :H, 0], ls='--', color='k', alpha=0.2)
        plt.subplot(122)
        for i in range(1, n_test):
            plt.plot(tm, Xs_t_test[i, :H, 1], ls='--', color='k', alpha=0.2)
        plt.show(block=False)

    def get_score(self, Xs_t_test):
        n_test, H, dX = Xs_t_test.shape
        K = self.params['n_components']
        traj_density = self.traj_density
        X_test_log_ll = np.zeros((H, n_test))
        X_test_rmse = np.zeros((H, n_test))

        for t in range(H):
            for i in range(n_test):
                X_test = Xs_t_test[i]
                x_t = X_test[t].reshape(-1)
                log_prob_mix = np.zeros(K)
                for k in range(K):
                    x_mu_t = traj_density[t][1][k]
                    x_var_t = traj_density[t][2][k]
                    # x_var_t = x_var_t + np.eye(dX) * jitter_val
                    x_var_t = np.diag(np.diag(x_var_t))
                    pi = traj_density[t][0][k]
                    log_prob_mix[k] = sp.stats.multivariate_normal.logpdf(x_t, x_mu_t, x_var_t) + np.log(pi)
                X_test_log_ll[t, i] = logsum(log_prob_mix)
                label = self.dpgmm.predict(x_t.reshape(1, -1))
                x_mu_l = self.dpgmm.means_[label].reshape(-1)
                X_test_rmse[t, i] = np.dot((x_mu_l - x_t), (x_mu_l - x_t).T)
        nll_mean = np.mean(X_test_log_ll.reshape(-1))
        nll_std = np.std(X_test_log_ll.reshape(-1))
        rmse = np.sqrt(np.mean(X_test_rmse.reshape(-1)))
        return nll_mean, nll_std, rmse, X_test_log_ll

    def estimate_gmm_traj_density(self, params, Xs_t_test, plot=True):
        self.params = params
        sample_trajs = self.sample_trajs
        traj_density = []
        H = sample_trajs.shape[1]
        dX = sample_trajs.shape[2]
        n_test, H, _ = Xs_t_test.shape

        self.dpgmm = mixture.BayesianGaussianMixture(**params)
        self.dpgmm.fit(sample_trajs[:, 0, :])
        traj_density.append([self.dpgmm.weights_, self.dpgmm.means_, self.dpgmm.covariances_])
        for t in range(1, H):
            self.dpgmm = mixture.BayesianGaussianMixture(**params)
            self.dpgmm.fit(sample_trajs[:,t,:])
            if np.linalg.norm(self.dpgmm.means_[0] - traj_density[t-1][1][0]) > np.linalg.norm(self.dpgmm.means_[0] - traj_density[t - 1][1][1]):
                traj_density.append([np.flip(self.dpgmm.weights_, axis=0), np.flip(self.dpgmm.means_, axis=0), np.flip(self.dpgmm.covariances_, axis=0)])
            else:
                traj_density.append([self.dpgmm.weights_, self.dpgmm.means_, self.dpgmm.covariances_])
        self.traj_density = traj_density
        if plot:
            self.plot_gmm_traj(Xs_t_test)

        return self.get_score(Xs_t_test)

class traj_with_globalgp(traj_with_moe):
    def __init__(self, x_mu_0, x_var_0, gp, massSlideWorld, dlt_mdl=False):
        self.x_mu_0 = x_mu_0
        self.x_var_0 = x_var_0
        self.gp = gp
        self.massSlideWorld = massSlideWorld
        self.dlt_mdl = dlt_mdl

    def sample(self, num_samples, H):
        dX = self.x_mu_0.shape[0]
        sample_trajs = np.zeros((num_samples, H, dX))
        for s in range(num_samples):
            mu = self.x_mu_0
            var = self.x_var_0
            sample_trajs[s][0] = np.random.multivariate_normal(mu, var)
            for t in range(1, H):
                x = sample_trajs[s][t-1]
                x = x.reshape(-1)
                um, uv = self.massSlideWorld.predict(x.reshape(1,-1), t)
                # um = np.asscalar(um)
                # uv = np.asscalar(uv)
                um = um.reshape(-1)
                uv = uv.reshape(-1)
                u = np.random.normal(um, uv)
                xu = np.append(x, u)
                mu, std = self.gp.predict(xu.reshape(1,-1))
                mu = mu.reshape(-1)
                std = std.reshape(-1)**2
                var = np.diag(std)
                if not self.dlt_mdl:
                    sample_trajs[s][t] = np.random.multivariate_normal(mu, var)
                else:
                    sample_trajs[s][t] = np.random.multivariate_normal(mu, var) + sample_trajs[s][t-1]

        self.sample_trajs = sample_trajs
        return sample_trajs

class SimplePolicy(object):
    def __init__(self, Xrs, Us, params):
        self.Xr = np.mean(Xrs, axis=0)
        self.U_var = np.var(Us, axis=0)
        self.kp = params['Kp']
        self.kd = params['Kd']
        self.dP = params['dP']
        self.dU = params['dU']

    def act(self, x, t):
        ex = (self.Xr[t] - x)
        eq = ex[:self.dP]
        eqd = ex[self.dP:]
        u = np.diag(self.kp).dot(eq) + np.diag(self.kd).dot(eqd)
        un = np.random.normal(u, np.sqrt(self.U_var[t]))
        return un, u

    def predict(self, X, t, return_std=True):
        U = np.zeros((X.shape[0], self.dU))
        U_noise = np.zeros((X.shape[0], self.dU))
        for i in range(X.shape[0]):
            _, U[i] = self.act(X[i], t)
            # U_noise[i] = np.maximum(np.sqrt(self.U_var[t]), np.full(7, 1e-1))
            U_noise[i] = np.sqrt(self.U_var[t])
        if return_std:
            return U, U_noise
        else:
            return U

def plotEllipsoid(center, radii, rotation, ax=None, plotAxes=False, col='b', alpha=0.2):
    """Plot an ellipsoid"""
    make_ax = ax == None
    if make_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0.0, 2.0 * np.pi, 20)
    v = np.linspace(0.0, np.pi, 20)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plotAxes:
        # make some purdy axes
        axes = np.array([[radii[0], 0.0, 0.0],
                         [0.0, radii[1], 0.0],
                         [0.0, 0.0, radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=col)

    # plot ellipsoid
    # ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=col, alpha=alpha)
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=col, alpha=alpha)

    if make_ax:
        plt.show()
        plt.close(fig)
        del fig

def plotEllipsiodError(mu_sq, cov_sq, col, ax=None, alpha=0.2):
    '''
    :param mu_sq: NX3 center points
    :param cov_sq: NX3X3 covariances for each point
    :param col:
    :param alpha:
    :return:
    '''
    assert(mu_sq.shape[0]==cov_sq.shape[0]==col.shape[0])
    assert(ax is not None)
    for i in range(mu_sq.shape[0]):
        cov = cov_sq[i]
        mu = mu_sq[i]
        L, U = np.linalg.eigh(cov)
        radii = 1.96 * np.sqrt(L)
        plotEllipsoid(mu, radii, U, ax=ax, plotAxes=False, col=col[i], alpha=alpha)

