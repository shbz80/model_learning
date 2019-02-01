import numpy as np
from scipy.linalg import cholesky

class UGP(object):
    '''
    This class assumes the multi dimensional GP in multidim_gp.py
    '''
    def __init__(self, L, alpha=1e-3, kappa=0., beta=2.):
        self.L = L
        self.N = 2*L + 1
        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta


    def get_sigma_points(self, mu, var):
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
        try:
            chol = cholesky((L + Lambda)*var, lower=True)
        except np.linalg.LinAlgError:
            assert(False)
        for i in range(1, L+1):
            sigmaMat[i, :] = mu + chol[:,i-1]
            sigmaMat[L+i, :] = mu - chol[:, i-1]
            # sigmaMat[i, :] = mu + chol[i - 1, :]
            # sigmaMat[L + i, :] = mu - chol[i - 1, :]

        W_mu = np.zeros(N)
        Wi = 1. / (2. * (L + Lambda))
        W_mu.fill(Wi)
        W_mu[0] = Lambda / (L + Lambda)

        W_var = np.zeros(N)
        W_var.fill(Wi)
        W_var[0] = W_mu[0] + (1. - alpha**2 + beta)
        return sigmaMat, W_mu, W_var

    def get_posterior(self, fn, mu, var):
        sigmaMat, W_mu, W_var = self.get_sigma_points(mu, var)
        Y_mu, Y_std = fn.predict(sigmaMat, return_std=True)
        N, Do = Y_mu.shape
        Y_var = Y_std **2
        Y_var = Y_var.reshape(N,Do)
        # Y_mu_post = np.average(Y_mu, axis=0, weights=W_mu)    # DX1
        Y_mu_post = Y_mu[0]
        Y_var_post = np.zeros((Do,Do))
        # for i in range(N):
        #    y = Y_mu[i] - Y_mu_post
        #    yy_ = np.outer(y, y)
        #    Y_var_post += W_var[i]*yy_
        #Y_var_post = np.diag(np.diag(Y_var_post))     # makes it worse
        Y_var_post += np.diag(Y_var[0])
        if Do == 1:
            Y_mu_post = np.asscalar(Y_mu_post)
            Y_var_post = np.asscalar(Y_var_post)
        # ip op cross covariance
        Di = mu.shape[0]
        XY_cross_cov = np.zeros((Di, Do))
        for i in range(N):
            y = Y_mu[i] - Y_mu_post
            x = sigmaMat[i] - mu
            xy_ = np.outer(x, y)
            XY_cross_cov += W_var[i] * xy_
        return Y_mu_post, Y_var_post, Y_mu, Y_var, XY_cross_cov

    def get_posterior_pol(self, fn, mu, var):
        sigmaMat, W_mu, W_var = self.get_sigma_points(mu, var)
        Y_mu, Y_std = fn.predict(sigmaMat, return_std=True)
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
        if Do == 1:
            Y_mu_post = np.asscalar(Y_mu_post)
            Y_var_post = np.asscalar(Y_var_post)
        # ip op cross covariance
        Di = mu.shape[0]
        XY_cross_cov = np.zeros((Di, Do))
        for i in range(N):
            y = Y_mu[i] - Y_mu_post
            x = sigmaMat[i] - mu
            xy_ = np.outer(x, y)
            XY_cross_cov += W_var[i] * xy_
        return Y_mu_post, Y_var_post, Y_mu, Y_var, XY_cross_cov
