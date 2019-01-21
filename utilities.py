import colorsys
import numpy as np
from math import pi
from matplotlib.patches import Ellipse
import scipy.special as sp
import scipy.linalg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
import matplotlib.pyplot as plt

def gp_plot(X_test, Y_test,gp):
    '''
    assumes the GP is already Fitted
    '''
    dY = Y_test.shape[1]
    N, dX = X_test.shape

    plt.figure()
    f=0
    for i in range(dY):
        for j in range(dX):
            f += 1
            plt.subplot(dY,dX,f)
            plt.scatter(X_test[:,j],Y_test[:,i])
            plt.title('Test')

    Y_pred = gp.predict(X_test)
    plt.figure()
    f=0
    for i in range(dY):
        for j in range(dX):
            f += 1
            plt.subplot(dY,dX,f)
            plt.scatter(X_test[:,j],Y_pred[:,i])
            plt.title('Pred')
            # plt.fill(X_grid[:,j],Y_grid[:,i] - 1.9600*Y_std,
            #         Y_grid[:,i] + 1.9600*Y_std, alpha=0.2)

    plt.show()

class MassSlideWorld(object):
    def __init__(self, m1=1., m1_init_pos=0, m2=2., m2_init_pos=3., mu=0.5, fp_start=6.,fp_end=10., block=10., dt=0.01):
        self.m1 = m1
        self.m1_init_pos = m1_init_pos
        self.m2 = m2
        self.m2_init_pos = m2_init_pos
        self.mu = mu
        self.dt = dt
        self.fp_start = fp_start
        self.fp_end = fp_end
        self.block = block
        self.reset()

    def dynamics(self,X,m,b,u):
        M = np.array([[0, 1], [0, -b/m]])
        U = np.array([[0], [u/m]])
        return M.dot(X) + U

    def reset(self):
        self.X = np.zeros((2,1))
        self.t = 0
        self.prev_m = self.m1
        self.m = self.m1

    def step(self, u):
        X = self.X
        dt = self.dt
        if (self.X[0]<self.block):
            if self.X[0]<self.m2_init_pos: # mode 1
                self.prev_m = self.m
                m, b = self.m1, 0
                self.m = m
                self.X[1] = self.prev_m/self.m * self.X[1]
            if (self.X[0]<self.fp_start) and (self.X[0]>=self.m2_init_pos): # mode 2
                self.prev_m = self.m
                m = self.m1 + self.m2
                self.m = m
                self.X[1] = self.prev_m/self.m * self.X[1]
                b = 0
            if (self.X[0]<self.fp_end) and (self.X[0]>=self.fp_start): # mode 3
                self.prev_m = self.m
                m = self.m1 + self.m2
                self.m = m
                self.X[1] = self.prev_m/self.m * self.X[1]
                N = m*9.8
                b = N*self.mu*self.X[1]
            # if (self.X[0]<self.block) and (self.X[0]>=self.fp_end):
            #     self.prev_m = self.m
            #     # m = self.m1
            #     m = self.m1 + self.m2
            #     self.m = m
            #     N = m*9.8
            #     # b = N*self.mu*self.X[1]
            #     b = 0
            #     self.X[1] = self.prev_m/self.m * self.X[1]
            k1 = self.dynamics(X,m,b,u)
            k2 = self.dynamics(X+0.5*dt*k1,m,b,u)
            k3 = self.dynamics(X+0.5*dt*k2,m,b,u)
            k4 = self.dynamics(X+dt*k3,m,b,u)
            self.X = X + dt*(1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
        if (self.X[0]>=self.block):
            self.X[1] = 0
        self.t = self.t + dt
        return self.t, self.X

def jacobian_analytic_to_geometric(J_A, phi):
    '''
    assumes xyz Euler convention
    phi is Euler angle vector
    '''
    s = np.sin
    c = np.cos

    assert(phi.shape==(3,))
    x = phi[0]
    y = phi[1]
    z = phi[2]

    Tang = np.array([[1., 0., s(y)],
                     [0., c(x), -c(y)*s(x)],
                     [0., s(x), c(x)*c(y)]
                    ])
    Ttrans = np.diag(np.ones(3))

    T_A = np.block([[Ttrans, np.zeros((3,3))],
                    [np.zeros((3,3)), Tang]
                   ])
    J_G = T_A.dot(J_A)
    return T_G

def jacobian_geometric_to_analytic(J_G, phi):
    '''
    assumes xyz Euler convention
    phi is Euler angle vector
    '''
    s = np.sin
    c = np.cos

    assert(phi.shape==(3,))
    x = phi[0]
    y = phi[1]
    z = phi[2]

    Tang_inv = np.array([[1., s(x)*s(y)/c(y), -c(x)*s(y)/c(y)],
                          [0., c(x), s(x)],
                          [0., -s(x)/c(y), c(x)/c(y)]
                         ])
    Ttrans_inv = np.diag(np.ones(3))
    T_A_inv = np.block([[Ttrans_inv, np.zeros((3,3))],
                        [np.zeros((3,3)), Tang_inv]
                       ])
    J_A = T_A_inv.dot(J_G)
    return J_A

def closed_loop_IK(x, q0, skel, gripper, euler_from_matrix, alpha=0.1):
    max_itr = 100
    q = q0
    q_dot = np.zeros(7)
    for itr in range(max_itr):
        skel.set_positions(q)
        skel.set_velocities(q_dot)
        T = gripper.T
        xpos = T[:3,3]
        xrot = T[:3,:3]
        xrot = euler_from_matrix(xrot)
        x_k = np.append(xpos,xrot)
        q_p = q
        J_G = gripper.world_jacobian()
        J_A = jacobian_geometric_to_analytic(J_G, x_k[3:])
        # s = np.linalg.svd(J_A,compute_uv=False)
        # print 'svd',s
        J_A_inv = np.linalg.pinv(J_A)
        dq = alpha*J_A_inv.dot(x - x_k)
        q += dq
        print 'q',q
        print 'dq',dq
        raw_input()
        if np.linalg.norm(dq)<1e-6:
            return q
    return None

    # def reset(self):
    #     self.pos = 0
    #     self.vel = 0
    #     self.acc = 0
    #     self.t = 0
    #     self.decoupled = True
    #     self.m = self.m1
    #     self.prev_m = self.m
    #     self.fric = 0
    #
    # def step(self,f):
    #     if self.pos >= self.m2_init_pos:
    #         self.prev_m = self.m
    #         self.m = self.m1 + self.m2
    #         self.vel = self.vel*self.prev_m/self.m
    #         f_acc = self.m*self.acc
    #         if (self.pos >= 6.) and (self.pos < 10.):
    #             N = self.m*9.8
    #             self.fric = self.mu*N*np.sign(self.vel)
    #         f_acc = f - self.fric
    #         self.acc = f_acc/self.m
    #         self.vel = self.vel + self.acc*self.dt
    #         self.pos = self.pos + self.vel*self.dt
    #         self.t = self.t + self.dt
    #     if self.pos < self.m2_init_pos:
    #         self.prev_m = self.m
    #         self.m = self.m1
    #         self.acc = f/self.m
    #         self.vel = self.vel*self.prev_m/self.m + self.acc*self.dt
    #         self.pos = self.pos + self.vel*self.dt
    #         self.t = self.t + self.dt
    #     return self.t, self.pos, self.vel


def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def get_N_HexCol(N=5):

    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in xrange(N)]
    # hex_out = []
    rgb_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
        # hex_out.append("".join(map(lambda x: chr(x).encode('hex'),rgb)))
        rgb_out.append(rgb)
    return rgb_out

def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.005
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

def plot_ellipse(ax, mu, sigma, color="k"):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(sigma)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    s=9.210 # 99%
    # s=5.991 # 95%
    w, h = 2 * np.sqrt(vals*s)


    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    # ellipse.set_clip_box(ax.bbox)
    # ellipse.set_alpha(0.2)
    # ax.add_artist(ellipse)

    ax.add_patch(Ellipse(mu, w, h, angle=theta, fc=color, linestyle='dashed', alpha=0.2))

def log_multivariate_t_distribution(x,mu,Sigma,v,D):
    '''
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (D dimensional numpy array or scalar)
        mu = mean (D dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        v = degrees of freedom
        D: dimension
    '''

    # if ((1. * (D+v)/2.) <= 170.):
    #     Num = sp.gamma(1. * (D+v)/2.)/sp.gamma(1.*v/2.)
    # else:
    #     Num = pow((1.*v/2.),(1.*D/2.)) # approximation for high values of v
    # Denom = pow(v*pi,1.*D/2.) * pow(np.linalg.det(Sigma),1./2.) \
    #         * pow(1 + (1./v)* np.dot((x - mu).T, np.dot(np.linalg.inv(Sigma),(x - mu))), 1.*(D+v)/2.)
    # p = Num/Denom
    # p1 = np.asscalar(p)
    log_num = sp.gammaln(1. * (D+v)/2.)-sp.gammaln(1.*v/2.)

    p1 = 1.*D/2. * np.log(v*pi)
    p2 = 1./2. * np.log(np.linalg.det(Sigma))
    p3 = 1.*(D+v)/2. * np.log(1 + (1./v)* np.dot((x - mu).T, np.dot(np.linalg.inv(Sigma),(x - mu))))
    log_denom = p1+p2+p3

    log_p = log_num - log_denom
    log_p = np.asscalar(log_p)
    return log_p

def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable


    '''
    d = len(m)
    m = np.asarray(m)
    m = np.reshape(m,(1,d))
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    xi = m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal
    return xi

def logsum(vec, axis=0, keepdims=True):
    #TODO: Add a docstring.
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec-maxv), axis=axis, keepdims=keepdims)) + maxv

def conditional_Gaussian_mixture(mu, sigma, xu, dxu,dxux):
    """
    Predict X' from XU
    Args:
        pts: A N x Dxu array of points.
    Returns:
        max of mode/mean of conditionals p(X'|XU)
        code duplicated in gmm.py TODO
    """
    K = mu.shape[0] # 4
    N = xu.shape[0]
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

def estep(data, mus, sigmas, pi):
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
    K = sigmas.shape[0]
    logobs = -0.5*np.ones((N, K))*D*np.log(2*np.pi)
    for i in range(K):
        mu, sigma = mus[i], sigmas[i]
        L = scipy.linalg.cholesky(sigma, lower=True)
        logobs[:, i] -= np.sum(np.log(np.diag(L)))

        diff = (data - mu).T
        soln = scipy.linalg.solve_triangular(L, diff, lower=True)
        logobs[:, i] -= 0.5*np.sum(soln**2, axis=0)

    logobs += np.log(pi).T
    return logobs
