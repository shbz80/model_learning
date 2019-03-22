import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
import GPy

class MultidimGP(object):
    def __init__(self, gpr_params_list, out_dim):

        self.gp_list = [GaussianProcessRegressor(**gpr_params_list[i]) for i in range(out_dim)]
        self.out_dim = out_dim
        # self.m = None

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
            print 'GP',i,'fit ended'

        # kernel = GPy.kern.RBF(input_dim=3, ARD=True)
        # self.m = GPy.models.GPRegression(X, Y, kernel)
        # self.m.rbf.lengthscale[:] = 1.0
        # self.m.rbf.lengthscale.constrain_bounded(1e-1, 1e1)
        # self.m.rbf.variance[:] = 1.0
        # self.m.rbf.variance.constrain_bounded(1e-2, 1e2)
        # self.m.Gaussian_noise[:] = 1.0
        # self.m.Gaussian_noise.constrain_bounded(1e-4, 1e1)
        # self.m.optimize_restarts(optimizer='lbfgs', num_restarts=1)


    def predict(self, X, return_std=True):
        Y_mu = np.zeros((X.shape[0], self.out_dim))
        Y_std = np.zeros((X.shape[0], self.out_dim))
        for i in range(self.out_dim):
            gp = self.gp_list[i]
            mu, std = gp.predict(X, return_std=return_std)
            Y_mu[:, i] = mu
            Y_std[:, i] = std
        return Y_mu, Y_std

        # Y_mu, Y_std = self.m.predict_noiseless(X)
        # Y_std = np.sqrt(Y_std)
        # return Y_mu, Y_std

