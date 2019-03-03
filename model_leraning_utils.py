import numpy as np
from scipy.linalg import cholesky
import pykdl_utils
import hrl_geom.transformations as trans
from hrl_geom.pose_converter import PoseConv
from urdf_parser_py.urdf import Robot
from pykdl_utils.kdl_kinematics import *
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

    def get_posterior(self, fn, mu, var):
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
        Y_mu, Y_std = fn.predict(sigmaMat, return_std=True) # same signature as the predict function of gpr but can be
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

    def get_posterior_time_indexed(self, fn, mu, var, t):
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
        Y_mu, Y_std = fn.predict(sigmaMat, t, return_std=True) # same signature as the predict function of gpr but can be
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

    def get_posterior_pol(self, fn, mu, var):
        '''
        same as above but used for debugging
        :param fn:
        :param mu:
        :param var:
        :return:
        '''
        sigmaMat, W_mu, W_var = self.get_sigma_points(mu, var)
        Y_mu, Y_std = fn.predict(sigmaMat, return_std=True) # Y_std is the std dev of each points from gp
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
        Y_var_post += np.diag(Y_var[0]) # add gp var of the mean point, valid only if fn is a gp
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

class dummySVM(object):
    def __init__(self, label):
        self.label = label

    def predict(self, ip):
        return np.full(ip.shape[0],self.label)

class YumiKinematics(object):
    def __init__(self):
        self.f = f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')
        self.dQ = 7
        robot = Robot.from_xml_string(f.read())
        self.base_link = robot.get_root()
        self.end_link = 'left_contact_point'
        self.kdl_kin = KDLKinematics(robot, self.base_link, self.end_link)

    def forward(self, X):
        X.reshape(-1, self.dQ)
        EX = np.zeros((X.shape[0], 12))
        dQ = self.dQ
        for i in range(X.shape[0]):
            x = X[i]
            q = x[:dQ]
            q_dot = x[dQ:]
            Tr = self.kdl_kin.forward(q, end_link=self.end_link, base_link=self.base_link)
            epos = np.array(Tr[:3, 3])
            epos = epos.reshape(-1)
            erot = np.array(Tr[:3, :3])
            erot = trans.euler_from_matrix(erot)
            ep = np.append(epos, erot)

            J_G = np.array(self.kdl_kin.jacobian(q))
            J_G = J_G.reshape((6, 7))
            J_A = self.jacobian_geometric_to_analytic(J_G, ep[3:])
            ep_dot = J_A.dot(q_dot)
            ex = np.concatenate((ep,ep_dot))
            EX[i] = ex
        return EX

    def jacobian_analytic_to_geometric(self, J_A, phi):
        '''
        assumes xyz Euler convention
        phi is Euler angle vector
        '''
        s = np.sin
        c = np.cos

        assert (phi.shape == (3,))
        x = phi[0]
        y = phi[1]
        z = phi[2]

        Tang = np.array([[1., 0., s(y)],
                         [0., c(x), -c(y) * s(x)],
                         [0., s(x), c(x) * c(y)]
                         ])
        Ttrans = np.diag(np.ones(3))

        T_A = np.block([[Ttrans, np.zeros((3, 3))],
                        [np.zeros((3, 3)), Tang]
                        ])
        J_G = T_A.dot(J_A)
        return J_G

    def jacobian_geometric_to_analytic(self, J_G, phi):
        '''
        assumes xyz Euler convention
        phi is Euler angle vector
        '''
        s = np.sin
        c = np.cos

        assert (phi.shape == (3,))
        x = phi[0]
        y = phi[1]
        z = phi[2]

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


