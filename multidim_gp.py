import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

class MultidimGP(object):
    def __init__(self, gpr_params, out_dim):

        self.gp_list = [GaussianProcessRegressor(**gpr_params) for i in range(out_dim)]
        self.out_dim = out_dim

    def fit(self, X, Y):
        assert(Y.shape[1]>=2)
        assert (X.shape[1]>=2)
        for i in range(self.out_dim):
            self.gp_list[i].fit(X,Y[:,i])

    def predict(self, X, return_std=True):
        Y_mu = np.zeros((X.shape[0], self.out_dim))
        Y_std = np.zeros((X.shape[0], self.out_dim))
        for i in range(self.out_dim):
            gp = self.gp_list[i]
            mu, std = gp.predict(X, return_std)
            Y_mu[:, i] = mu
            Y_std[:, i] = std
        return Y_mu, Y_std