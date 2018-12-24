""" This file defines a Gaussian mixture model class. """

import numpy as np
import scipy.linalg


def logsum(vec, axis=0, keepdims=True):
    #TODO: Add a docstring.
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec-maxv), axis=axis, keepdims=keepdims)) + maxv


class GMM(object):
    """ Gaussian Mixture Model. """
    def __init__(self, init_sequential=False, eigreg=False, warmstart=True,dxu=21,dxux=35):
        self.init_sequential = init_sequential
        self.eigreg = eigreg
        self.warmstart = warmstart
        self.sigma = None
        self.dxu = dxu
        self.dxux = dxux

    # def getClustW(self):
    #     """
    #     Return current cluster weights.
    #     Returns:
    #         w: A N x K array, N number of data points, K number of clusters
    #         weight for data point for each cluster
    #         wn: A N x K array, N number of data points, K number of clusters
    #         normalized cluster distribution over the points
    #     """
    #     return self.w, self.wn
    # def getLogLikelihood(self):
    #     """
    #     Return current log likelihood.
    #     """
    #     return self.ll
    def predict(self, xu):
        """
        Predict X' from XU
        Args:
            pts: A N x Dxu array of points.
        Returns:
            max of mode/mean of conditionals p(X'|XU)
        """
        # self.sigma = np.zeros((K, Do, Do))
        # self.mu = np.zeros((K, Do))
        sigma = self.sigma # (4,2,2)
        mu = self.mu # (4,2)
        K = mu.shape[0] # 4
        N = xu.shape[0] # 100
        dxu = self.dxu # 1
        dxux = self.dxux # 2
        xui = slice(dxu)
        xuxi = slice(dxu,dxux)
        mu_x1_xu = np.zeros((N,K,dxux-dxu)) # (100,1)

        for k in range(K):
            mu_xu = mu[k][xui].T #(1,) 21X1
            mu_x1 = mu[k][xuxi].T #(1,) 14X1
            sig = sigma[k]
            # print sigma.shape
            sig_xuxu = sig[xui,xui] # symmetric 21X21
            inv_sig_xuxu = np.linalg.pinv(sig_xuxu) #(2,2)
            sig_xux1 = sig[xui,xuxi] # 21X14
            sig_x1xu = sig_xux1.T # 14X21
            sig_x1x1 = sig[xuxi,xuxi] # symmetric 14X14
            for n in range(N):
                dxu = xu[n].T - mu_xu # 21X1
                mu_x1_xu[n][k] = mu_x1 + sig_x1xu.dot(inv_sig_xuxu.dot(dxu))
        return mu_x1_xu

    def inference(self, pts):
        """
        Evaluate dynamics prior.
        Args:
            pts: A N x D array of points.
        """
        # Compute posterior cluster weights.
        logwts = self.clusterwts(pts)

        # Compute posterior mean and covariance.
        mu0, Phi = self.moments(logwts)

        # Set hyperparameters.
        m = self.N
        n0 = m - 2 - mu0.shape[0]

        # Normalize.
        m = float(m) / self.N
        n0 = float(n0) / self.N
        return mu0, Phi, m, n0

    def estep(self, data):
        """
        Compute log observation probabilities under GMM.
        Args:
            data: A N x D array of points.
        Returns:
            logobs: A N x K array of log probabilities (for each point
                on each cluster).
        """
        # Constants.
        N, D = data.shape
        K = self.sigma.shape[0]

        logobs = -0.5*np.ones((N, K))*D*np.log(2*np.pi)
        for i in range(K):
            mu, sigma = self.mu[i], self.sigma[i]
            L = scipy.linalg.cholesky(sigma, lower=True)
            logobs[:, i] -= np.sum(np.log(np.diag(L)))

            diff = (data - mu).T
            soln = scipy.linalg.solve_triangular(L, diff, lower=True)
            logobs[:, i] -= 0.5*np.sum(soln**2, axis=0)

        logobs += self.logmass.T
        return logobs

    def estep_m(self, data):
        """
        Compute log observation probabilities under GMM.
        Args:
            data: A N x D array of points.
        Returns:
            logobs: A N x K array of log probabilities (for each point
                on each cluster).
        """
        # Constants.
        N, D = data.shape
        K = self.sigma.shape[0]

        logobs = -0.5*np.ones((N, K))*D*np.log(2*np.pi)
        for i in range(K):
            mu, sigma = self.mu[i], self.sigma[i]
            L = scipy.linalg.cholesky(sigma, lower=True)
            logobs[:, i] -= np.sum(np.log(np.diag(L)))

            diff = (data - mu).T
            soln = scipy.linalg.solve_triangular(L, diff, lower=True)
            logobs[:, i] -= 0.5*np.sum(soln**2, axis=0)

        # logobs += self.logmass.T
        return logobs

    def moments(self, logwts):
        """
        Compute the moments of the cluster mixture with logwts.
        Args:
            logwts: A K x 1 array of log cluster probabilities.
        Returns:
            mu: A (D,) mean vector.
            sigma: A D x D covariance matrix.
        """
        # Exponentiate.
        wts = np.exp(logwts)

        # Compute overall mean.
        mu = np.sum(self.mu * wts, axis=0)

        # Compute overall covariance.
        diff = self.mu - np.expand_dims(mu, axis=0)
        diff_expand = np.expand_dims(self.mu, axis=1) * \
                np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(wts, axis=2)
        sigma = np.sum((self.sigma + diff_expand) * wts_expand, axis=0)
        return mu, sigma

    def clusterwts(self, data):
        """
        Compute cluster weights for specified points under GMM.
        Args:
            data: An N x D array of points
        Returns:
            A K x 1 array of average cluster log probabilities.
        """
        # Compute probability of each point under each cluster.
        logobs = self.estep(data)

        # Renormalize to get cluster weights.
        logwts = logobs - logsum(logobs, axis=1)

        # Average the cluster probabilities.
        logwts = logsum(logwts, axis=0) - np.log(data.shape[0])
        return logwts.T

    def update(self, data, K, max_iterations=100):
        """
        Run EM to update clusters.
        Args:
            data: An N x D data matrix, where N = number of data points.
            K: Number of clusters to use.
        """
        # Constants.
        N = data.shape[0]
        Do = data.shape[1]

        print('Fitting GMM with %d clusters on %d points', K, N)

        if (not self.warmstart or self.sigma is None or
                K != self.sigma.shape[0]):
            # Initialization.
            print('Initializing GMM.')
            self.sigma = np.zeros((K, Do, Do))
            self.mu = np.zeros((K, Do))
            self.logmass = np.log(1.0 / K) * np.ones((K, 1))
            self.mass = (1.0 / K) * np.ones((K, 1))
            self.N = data.shape[0]
            self.w = np.zeros((N,K))
            self.wn = np.zeros((N,K))
            self.ll = 0.
            N = self.N

            # Set initial cluster indices.
            if not self.init_sequential:
                cidx = np.random.randint(0, K, size=(1, N))
            else:
                raise NotImplementedError()

            # Initialize.
            for i in range(K):
                cluster_idx = (cidx == i)[0]
                mu = np.mean(data[cluster_idx, :], axis=0)
                diff = (data[cluster_idx, :] - mu).T
                n = diff.shape[1]
                # sigma = (1.0 / K) * (diff.dot(diff.T)) # original
                sigma = (1.0 / n) * (diff.dot(diff.T))
                self.mu[i, :] = mu
                self.sigma[i, :, :] = sigma + np.eye(Do) * 2e-6

        prevll = -float('inf')
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logobs = self.estep(data)

            # Compute log-likelihood.
            ll = np.sum(logsum(logobs, axis=1))
            self.ll = ll
            print('GMM itr %d/%d. Log likelihood: %f',
                         itr, max_iterations, ll)

            if ll < prevll:
                # TODO: Why does log-likelihood decrease sometimes?
                print('Log-likelihood decreased! Ending on itr=%d/%d',
                             itr, max_iterations)
                break
            # if np.abs(ll-prevll) < 1e-5*prevll: # original
            if np.abs(ll-prevll) < np.abs(1e-10*prevll):
                print('GMM converged on itr=%d/%d',
                             itr, max_iterations)
                break
            prevll = ll

            # Renormalize to get cluster weights.
            logw = logobs - logsum(logobs, axis=1)
            assert logw.shape == (N, K)

            # Renormalize again to get weights for refitting clusters.
            logwn = logw - logsum(logw, axis=0)
            assert logwn.shape == (N, K)
            w = np.exp(logwn)
            self.w = np.exp(logw)
            self.wn = w
            # M-step: update clusters.
            # Fit cluster mass.
            self.logmass = logsum(logw, axis=0).T
            self.logmass = self.logmass - logsum(self.logmass, axis=0)
            assert self.logmass.shape == (K, 1)
            self.mass = np.exp(self.logmass)
            # Reboot small clusters.
            w[:, (self.mass < (1.0 / K) * 1e-4)[:, 0]] = 1.0 / N
            # Fit cluster means.
            w_expand = np.expand_dims(w, axis=2)
            data_expand = np.expand_dims(data, axis=1)
            self.mu = np.sum(w_expand * data_expand, axis=0)
            # Fit covariances.
            wdata = data_expand * np.sqrt(w_expand)
            assert wdata.shape == (N, K, Do)
            for i in range(K):
                # Compute weighted outer product.
                XX = wdata[:, i, :].T.dot(wdata[:, i, :])
                mu = self.mu[i, :]
                self.sigma[i, :, :] = XX - np.outer(mu, mu)

                if self.eigreg:  # Use eigenvalue regularization.
                    raise NotImplementedError()
                else:  # Use quick and dirty regularization.
                    sigma = self.sigma[i, :, :]
                    self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + \
                            1e-6 * np.eye(Do)
    def setGMMparams(self, mu, sigma):
        """
        Run EM to update clusters.
        Args:
            data: An N x D data matrix, where N = number of data points.
            K: Number of clusters to use.
        """
        # Constants.
        N = data.shape[0]
        Do = data.shape[1]

        print('Fitting GMM with %d clusters on %d points', K, N)

        if (not self.warmstart or self.sigma is None or
                K != self.sigma.shape[0]):
            # Initialization.
            print('Initializing GMM.')
            self.sigma = np.zeros((K, Do, Do))
            self.mu = np.zeros((K, Do))
            self.logmass = np.log(1.0 / K) * np.ones((K, 1))
            self.mass = (1.0 / K) * np.ones((K, 1))
            self.N = data.shape[0]
            self.w = np.zeros((N,K))
            self.wn = np.zeros((N,K))
            self.ll = 0.
            N = self.N

            # Set initial cluster indices.
            if not self.init_sequential:
                cidx = np.random.randint(0, K, size=(1, N))
            else:
                raise NotImplementedError()

            # Initialize.
            for i in range(K):
                cluster_idx = (cidx == i)[0]
                mu = np.mean(data[cluster_idx, :], axis=0)
                diff = (data[cluster_idx, :] - mu).T
                n = diff.shape[1]
                # sigma = (1.0 / K) * (diff.dot(diff.T)) # original
                sigma = (1.0 / n) * (diff.dot(diff.T))
                self.mu[i, :] = mu
                self.sigma[i, :, :] = sigma + np.eye(Do) * 2e-6

        prevll = -float('inf')
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logobs = self.estep(data)

            # Compute log-likelihood.
            ll = np.sum(logsum(logobs, axis=1))
            self.ll = ll
            print('GMM itr %d/%d. Log likelihood: %f',
                         itr, max_iterations, ll)

            if ll < prevll:
                # TODO: Why does log-likelihood decrease sometimes?
                print('Log-likelihood decreased! Ending on itr=%d/%d',
                             itr, max_iterations)
                break
            # if np.abs(ll-prevll) < 1e-5*prevll: # original
            if np.abs(ll-prevll) < np.abs(1e-6*prevll):
                print('GMM converged on itr=%d/%d',
                             itr, max_iterations)
                break
            prevll = ll

            # Renormalize to get cluster weights.
            logw = logobs - logsum(logobs, axis=1)
            assert logw.shape == (N, K)

            # Renormalize again to get weights for refitting clusters.
            logwn = logw - logsum(logw, axis=0)
            assert logwn.shape == (N, K)
            w = np.exp(logwn)
            self.w = np.exp(logw)
            self.wn = w
            # M-step: update clusters.
            # Fit cluster mass.
            self.logmass = logsum(logw, axis=0).T
            self.logmass = self.logmass - logsum(self.logmass, axis=0)
            assert self.logmass.shape == (K, 1)
            self.mass = np.exp(self.logmass)
            # Reboot small clusters.
            w[:, (self.mass < (1.0 / K) * 1e-4)[:, 0]] = 1.0 / N
            # Fit cluster means.
            w_expand = np.expand_dims(w, axis=2)
            data_expand = np.expand_dims(data, axis=1)
            self.mu = np.sum(w_expand * data_expand, axis=0)
            # Fit covariances.
            wdata = data_expand * np.sqrt(w_expand)
            assert wdata.shape == (N, K, Do)
            for i in range(K):
                # Compute weighted outer product.
                XX = wdata[:, i, :].T.dot(wdata[:, i, :])
                mu = self.mu[i, :]
                self.sigma[i, :, :] = XX - np.outer(mu, mu)

                if self.eigreg:  # Use eigenvalue regularization.
                    raise NotImplementedError()
                else:  # Use quick and dirty regularization.
                    sigma = self.sigma[i, :, :]
                    self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + \
                            1e-6 * np.eye(Do)
