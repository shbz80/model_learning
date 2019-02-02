import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV

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
            self.gp_list[i].fit(X,Y[:,i])

    def predict(self, X, return_std=True):
        Y_mu = np.zeros((X.shape[0], self.out_dim))
        Y_std = np.zeros((X.shape[0], self.out_dim))
        for i in range(self.out_dim):
            gp = self.gp_list[i]
            mu, std = gp.predict(X, return_std=return_std)
            Y_mu[:, i] = mu
            Y_std[:, i] = std
        return Y_mu, Y_std

    def cv_fit(self, X, Y, param_grid):
        assert(Y.shape[1]>=2)
        assert (X.shape[1]>=2)
        for i in range(self.out_dim):
            gp = GaussianProcessRegressor()
            grid_search = GridSearchCV(gp, param_grid=param_grid)
            grid_search.fit(X, Y[:, i])
            self.gp_list[i] = grid_search.best_estimator_