from __future__ import print_function
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
import GPy
from GPy.util.normalizer import Standardize
import time

class MultidimGP(object):
    def __init__(self, gpr_params_list, out_dim):
        self.gp_list = [GaussianProcessRegressor(**gpr_params_list[i]) for i in range(out_dim)]
        self.out_dim = out_dim

    def fit(self, X, Y, shuffle=False):
        assert(Y.shape[1]>=2)
        assert (X.shape[1]>=2)
        if shuffle:
            N, dY = Y.shape
            YX = np.concatenate((Y,X), axis=1)
            np.random.shuffle(YX)
            Y = YX[:, :dY]
            X = YX[:, dY:]
        for i in range(self.out_dim):
            # print 'GP', i, 'fit started'
            self.gp_list[i].fit(X,Y[:,i])
            print ('GP',i,'fit ended')

    def predict(self, X, return_std=True):
        Y_mu = np.zeros((X.shape[0], self.out_dim))
        Y_std = np.zeros((X.shape[0], self.out_dim))
        for i in range(self.out_dim):
            gp = self.gp_list[i]
            mu, std = gp.predict(X, return_std=return_std)
            Y_mu[:, i] = mu
            Y_std[:, i] = std
        return Y_mu, Y_std

class MdGpyGP(object):
    def __init__(self, gpr_params, out_dim):
        self.gp_param = gpr_params
        self.out_dim = out_dim

    def fit(self, X, Y):
        assert(Y.shape[1]>=2)
        assert (X.shape[1]>=2)
        self.gp_list = []
        in_dim = X.shape[1]
        for i in range(self.out_dim):
            gp_params = self.gp_param
            normalize = gp_params['normalize']


            kernel = GPy.kern.RBF(input_dim=in_dim, ARD=True)
            y = Y[:,i].reshape(-1,1)
            m = GPy.models.GPRegression(X, y, kernel, normalizer=normalize)

            x_sig = np.sqrt(np.var(X, axis=0))
            len_scale = x_sig
            len_scale_lb = np.min(x_sig/10.)
            len_scale_ub = np.max(x_sig / 1.)
            len_scale_b = (len_scale_lb, len_scale_ub)
            noise_var = gp_params['noise_var'][i] #1e-3
            y_var = np.var(Y[:,i])
            sig_var = y_var
            # sig_var = y_var - noise_var
            sig_var_b = (sig_var/10., sig_var*10.)

            m.rbf.lengthscale[:] = len_scale
            # m.rbf.lengthscale.constrain_bounded(len_scale_b[0], len_scale_b[1])
            m.rbf.variance[:] = sig_var
            # m.rbf.variance.fix()
            # m.rbf.variance.constrain_bounded(sig_var_b[0], sig_var_b[1])
            m.Gaussian_noise[:] = noise_var
            m.Gaussian_noise.fix()
            # m.Gaussian_noise.constrain_bounded(noise_var_b[0], noise_var_b[1])
            start_time = time.time()
            m.optimize_restarts(optimizer='lbfgs', num_restarts=1)
            # m.optimize()
            print ('GP',i, 'fit time', time.time() - start_time)
            self.gp_list.append(m)

    def predict(self, X, return_std=True):
        Y_mu = np.zeros((X.shape[0], self.out_dim))
        Y_std = np.zeros((X.shape[0], self.out_dim))
        for i in range(self.out_dim):
            gp = self.gp_list[i]
            mu, var = gp.predict_noiseless(X)
            # mu, var = gp.predict(X)
            Y_mu[:, i] = mu.reshape(-1)
            Y_std[:, i] = np.sqrt(var).reshape(-1)
        return Y_mu, Y_std

class MdGpyGPwithNoiseEst(MdGpyGP):
    def fit(self, X, Y):
        assert(Y.shape[1]>=2)
        assert (X.shape[1]>=2)
        self.gp_list = []
        in_dim = X.shape[1]
        for i in range(self.out_dim):
            gp_params = self.gp_param
            normalize = gp_params['normalize']


            kernel = GPy.kern.RBF(input_dim=in_dim, ARD=True)
            y = Y[:,i].reshape(-1,1)
            m = GPy.models.GPRegression(X, y, kernel, normalizer=normalize)

            # normalizer = Standardize()
            # normalizer.scale_by(y)
            # y_normalized = normalizer.normalize(y)

            x_sig = np.sqrt(np.var(X, axis=0))
            len_scale = x_sig
            len_scale_lb = np.min(x_sig/10.)
            len_scale_ub = np.max(x_sig * 1.)
            len_scale_b = (len_scale_lb, len_scale_ub)
            y_var = np.var(Y[:,i])
            noise_var = gp_params['noise_var'][i]
            if noise_var is None or y_var < noise_var:
                noise_var = y_var
                sig_var = y_var
            else:
                sig_var = y_var - noise_var
            sig_var_b = (sig_var/10., sig_var*10.)

            # snr = np.array([10., 2.])
            # y_sig = np.sqrt(sig_var)
            # noise_sig = y_sig / 2.
            # noise_var = noise_sig**2
            #
            # noise_sig_b = np.reciprocal(snr) * y_sig
            # noise_var_b = np.square(noise_sig_b)
            noise_var_b = np.array([noise_var/3., noise_var*3])

            m.rbf.lengthscale[:] = len_scale
            # m.rbf.lengthscale.constrain_bounded(len_scale_b[0], len_scale_b[1])
            m.rbf.variance[:] = sig_var
            # m.rbf.variance.fix()
            # m.rbf.variance.constrain_bounded(sig_var_b[0], sig_var_b[1])
            m.Gaussian_noise[:] = noise_var
            # m.Gaussian_noise.fix()
            # m.Gaussian_noise.constrain_bounded(noise_var_b[0], noise_var_b[1])
            start_time = time.time()
            m.optimize_restarts(optimizer='lbfgs', num_restarts=3)
            # m.optimize()
            print ('GP',i, 'fit time', time.time() - start_time)
            self.gp_list.append(m)

class MdGpySparseGP(MdGpyGP):

    def fit(self, X, Y):
        assert(Y.shape[1]>=2)
        assert (X.shape[1]>=2)
        self.gp_list = []
        in_dim = X.shape[1]
        for i in range(self.out_dim):
            # kernel = GPy.kern.RBF(input_dim=in_dim, ARD=True) + GPy.kern.White(in_dim)
            kernel = GPy.kern.RBF(input_dim=in_dim, ARD=True)
            y = Y[:,i].reshape(-1,1)

            num_z = X.shape[0] / 50
            num_z_min = 3
            num_z = max(num_z, num_z_min)
            num_z = 10
            m = GPy.models.SparseGPRegression(X, y, kernel=kernel, normalizer=True, num_inducing=num_z)
            # print(m)
            x_sig = np.sqrt(np.var(X, axis=0))
            len_scale = x_sig
            len_scale_lb = np.min(x_sig/10.)
            len_scale_ub = np.max(x_sig / 1.)
            len_scale_b = (len_scale_lb, len_scale_ub)
            noise_var = 1e-3
            y_var = np.var(Y[:,i])
            sig_var = y_var
            # sig_var = y_var - noise_var
            sig_var_b = (sig_var/10., sig_var*10.)

            m.rbf.lengthscale[:] = len_scale
            # m.rbf.lengthscale.constrain_bounded(len_scale_b[0], len_scale_b[1])
            m.rbf.variance[:] = sig_var
            # m.rbf.variance.fix()
            # m.rbf.variance.constrain_bounded(sig_var_b[0], sig_var_b[1])
            m.Gaussian_noise[:] = noise_var
            m.Gaussian_noise.fix()
            # m.Gaussian_noise.constrain_bounded(noise_var_b[0], noise_var_b[1])
            start_time = time.time()
            m.optimize_restarts(optimizer='lbfgs', num_restarts=1)
            # m.optimize()
            print ('GP', i, 'fit time', time.time() - start_time)
            self.gp_list.append(m)

