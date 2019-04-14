from __future__ import print_function
import numpy as np
from scipy.linalg import cholesky
import pykdl_utils
import hrl_geom.transformations as trans
from hrl_geom.pose_converter import PoseConv
from urdf_parser_py.urdf import Robot
from pykdl_utils.kdl_kinematics import *
import colorsys
import time
# from multidim_gp import MultidimGP
from multidim_gp import MdGpyGP as MultidimGP
from itertools import compress
from copy import deepcopy
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import copy
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
import operator
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


class dummySVM(object):
    def __init__(self, label):
        self.label = label

    def predict(self, ip):
        return np.full(ip.shape[0],self.label)

class YumiKinematics(object):
    def __init__(self, params):
        self.file = params['urdf']
        f = file(self.file, 'r')
        robot = Robot.from_xml_string(f.read())
        self.euler_string = params['euler_string']
        self.reverse_angles = params['reverse_angles']
        # self.base_link = robot.get_root()
        self.base_link = params['base_link']
        self.end_link = params['end_link']
        self.kdl_kin = KDLKinematics(robot, self.base_link, self.end_link)
        self.dQ = 7
        self.ik_cl_alpha = 0.1
        self.ik_cl_max_itr = 100

    def fwd_pose(self, q):
        Tr = self.kdl_kin.forward(q, end_link=self.end_link, base_link=self.base_link)
        epos = np.array(Tr[:3, 3])
        epos = epos.reshape(-1)
        erot = np.array(Tr[:3, :3])
        euler_string = self.euler_string
        if self.reverse_angles and self.euler_string=='szyx':
            euler_string = 'sxyz'
        tmp = trans.euler_from_matrix(erot, euler_string)
        if self.reverse_angles:
            erot = copy.copy(tmp[::-1])
        else:
            erot = tmp
        ep = np.append(epos, erot)
        return ep

    def get_fwd_mat(self, q):
        Tr = self.kdl_kin.forward(q, end_link=self.end_link, base_link=self.base_link)
        epos = np.array(Tr[:3, 3])
        epos = epos.reshape(-1)
        erot = np.array(Tr[:3, :3])
        return epos, erot

    def forward(self, X):
        X = X.reshape(-1, self.dQ*2)
        EX = np.zeros((X.shape[0], 12))
        dQ = self.dQ
        for i in range(X.shape[0]):
            x = X[i]
            q = x[:dQ]
            q_dot = x[dQ:]
            ep = self.fwd_pose(q)
            J_A = self.get_analytical_jacobian(q)
            ep_dot = J_A.dot(q_dot)
            ex = np.concatenate((ep,ep_dot))
            EX[i] = ex
        return EX

    def get_analytical_jacobian(self, q):
        ep = self.fwd_pose(q)
        J_G = np.array(self.kdl_kin.jacobian(q))
        J_G = J_G.reshape((6, 7))
        J_A = YumiKinematics.jacobian_geometric_to_analytic(J_G, ep[3:], self.euler_string, self.reverse_angles)
        return J_A


    # @staticmethod
    # def jacobian_analytic_to_geometric(J_A, phi):
    #     '''
    #     assumes xyz Euler convention
    #     phi is Euler angle vector
    #     '''
    #     s = np.sin
    #     c = np.cos
    #
    #     assert (phi.shape == (3,))
    #     x = phi[0]
    #     y = phi[1]
    #     z = phi[2]
    #
    #     Tang = np.array([[1., 0., s(y)],
    #                      [0., c(x), -c(y) * s(x)],
    #                      [0., s(x), c(x) * c(y)]
    #                      ])
    #     Ttrans = np.diag(np.ones(3))
    #
    #     T_A = np.block([[Ttrans, np.zeros((3, 3))],
    #                     [np.zeros((3, 3)), Tang]
    #                     ])
    #     J_G = T_A.dot(J_A)
    #     return J_G

    @staticmethod
    def jacobian_geometric_to_analytic(J_G, phi, euler_string='sxyz', reverse_angles=False):
        '''
        phi is Euler angle vector
        '''
        s = np.sin
        c = np.cos

        assert (phi.shape == (3,))
        if reverse_angles:
            z = phi[0]
            y = phi[1]
            x = phi[2]
        else:
            x = phi[0]
            y = phi[1]
            z = phi[2]

        if euler_string == 'szyx':
            Tang_inv = np.array([[c(z) * s(y) / c(y),   s(y) * s(z) / c(y), 1.0],
                                 [-s(z),                c(z),               0.],
                                 [c(z) / c(y),          s(z) / c(y),        0.]
                                 ])
        elif euler_string == 'sxyz':
            Tang_inv = np.array([[1., s(x) * s(y) / c(y), -c(x) * s(y) / c(y)],
                                 [0., c(x), s(x)],
                                 [0., -s(x) / c(y), c(x) / c(y)]
                                 ])

        Ttrans_inv = np.diag(np.ones(3))
        T_A_inv = np.block([[Ttrans_inv, np.zeros((3, 3))],
                            [np.zeros((3, 3)), Tang_inv]
                            ])
        J_A = T_A_inv.dot(J_G)
        return J_A

    def closed_loop_IK(self, x, q0):
        alpha = self.ik_cl_alpha
        max_itr = self.ik_cl_max_itr
        q_k = q0
        for itr in range(max_itr):
            x_k = self.fwd_pose(q_k)
            J_A = self.get_analytical_jacobian(q_k)
            J_A_inv = np.linalg.pinv(J_A)
            dq = alpha*J_A_inv.dot(x - x_k)
            q_k += dq
            if np.linalg.norm(dq)<1e-6:
                return q_k
        return None

def get_N_HexCol(N=5):

    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in xrange(N)]
    # hex_out = []
    rgb_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
        # hex_out.append("".join(map(lambda x: chr(x).encode('hex'),rgb)))
        rgb_out.append(rgb)
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
    start_time = time.time()
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
    print ('Transition GP training time:', time.time() - start_time)
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
        start_time = time.time()

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
        print ('SVMs training time:', time.time() - start_time)
        self.svm = copy.deepcopy(clf)


    def predict(self, XU, i):
        XU_std = self.scaler.transform(XU)
        next_modes = self.svm.predict(XU_std)
        return next_modes

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
        start_time = time.time()
        self.dpgmm.fit(self.X_std)
        print('DPGMM clustering time:', time.time() - start_time)
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
            print('VBGMM clustering time:', time.time() - start_time)
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
                        if l != l_n:
                            ys[n:n + 1, t:t + 1] = l_n
                    elif t == T - 1:
                        l = ys[n:n + 1, t:t + 1]
                        l_p = ys[n:n + 1, t - 1:t]
                        if l != l_p:
                            ys[n:n + 1, t:t + 1] = l_p
                    else:
                        l = ys[n:n + 1, t:t + 1]
                        l_n = ys[n:n + 1, t + 1:t + 2]
                        l_p = ys[n:n + 1, t - 1:t]
                        if l != l_n and l_p != l and l_p == l_n:
                            ys[n:n + 1, t:t + 1] = l_n
            for n in range(n_train):
                for t in range(T):
                    if t == 0:
                        l = ys[n:n + 1, t:t + 1]
                        l_n = ys[n:n + 1, t + 1:t + 2]
                        if l != l_n:
                            ys[n:n + 1, t:t + 1] = l_n
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