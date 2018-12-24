import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform

class sim_1d(object):
    def __init__(self, params):
        self.x0 = params['x0']          # init state
        self.xT = params['xT']          # final state
        self.xt = self.xT               # current target state
        self.x = self.x0                # current state
        self.a1 = params['a1']
        self.b1 = params['b1']
        self.a2 = params['a2']
        self.b2 = params['b2']
        self.a = 0.                     # current dynamics
        self.b = 0.                     # current dynamics
        self.L1 = params['L1']          # feedback gain
        self.L2 = params['L2']          # feedback gain
        self.L3 = params['L3']          # feedback gain
        self.L = self.L1                # feedback gain
        self.dt = params['dt']          # sample time
        self.t = 0.                     # current time instant
        self.T = params['T']            # total time in seconds
        self.xt1 = params['xt1']        # a state target
        self.xt2 = params['xt2']        # a state target
        self.xt3 = params['xt3']        # a state target
        self.dx = self.xt1 - self.x0    # current error state
        self.t1 = self.T/4.             # mode time threshold
        self.t2 = 2.0*self.T / 4.       # mode time threshold
        self.t3 = 3.0 * self.T / 4.     # mode time threshold
        self.w_sigma_1 = params['w_sigma_1']  # exploration noise
        self.w_sigma_2 = params['w_sigma_2']  # exploration noise
        self.w_sigma_3 = params['w_sigma_3']  # exploration noise
        self.x0_var = params['init_x_var']  # init state variance
        self.w_sigma = self.w_sigma_1   # current exploration noise
        self.type = params['type']      # choose between continuous and discontinuous dynamics
        self.mode = 1                   # current dynamics mode

    def reset(self):
        self.x = np.random.normal(self.x0,self.x0_var)
        self.dx = self.xt - self.x
        self.t = 0.
        self.set_mode(1)
        self.L = self.L1

    def set_mode(self, mode):
        self.mode = mode
        if mode==1:
            self.a = self.a1
            self.b = self.b1
            self.w_sigma = self.w_sigma_1
        elif mode==2:
            self.a = self.a2
            self.b = self.b1
            self.w_sigma = self.w_sigma_2
        elif mode==3:
            self.a = self.a2
            self.b = self.b2
            self.w_sigma = self.w_sigma_3

    def step(self, u):
        if self.type == 'cont':
            self.set_mode(1)
        elif self.type == 'disc':
            if self.t < self.t1:
                self.set_mode(1)
            elif self.t < self.t2:
                self.set_mode(2)
            elif self.t < self.t3:
                self.set_mode(3)
        self.dx = self.xt - self.x
        self.dx = self.a * self.dx + self.b * u
        self.x = self.xt - self.dx
        self.t = self.t + self.dt

    def act(self, dx):
        u = self.L * dx
        u_n = np.random.normal(self.L * dx, self.w_sigma)
        return u, u_n, self.w_sigma

    def policy(self):
        if self.type == 'cont':
            self.xt = self.xT
            self.L = self.L1*2.
            self.w_sigma_1 = self.w_sigma_2
        elif self.type == 'disc':
            if self.mode==1:
                self.xt = self.xt1
                self.L = self.L1
            elif self.mode == 2:
                self.xt = self.xt2
                self.L = self.L2
            elif self.mode == 3:
                self.xt = self.xt3
                self.L = self.L3
        return self.act(self.dx)

    def sim_episode(self, noise=True):
        self.reset()
        N = len(np.arange(0,self.T,self.dt))
        traj = np.zeros((N,5))
        for i in range(N):
            u, u_n, w = self.policy()
            traj[i,0] = self.t
            traj[i,1] = self.x
            traj[i,2] = u_n
            traj[i,3] = u
            traj[i,4] = w
            if noise: self.step(u_n)
            else: self.step(u)
        return traj

    def sim_episodes(self, num_episodes, noise=True):
        traj_list = []
        for i in range(num_episodes):
            traj = self.sim_episode(noise)
            traj_list.append(traj)
        return traj_list

class MomentMatching(object):
    def __init__(self, gp):
        [alpha_sq, lambda_1, lambda_2, w_sigma] = np.exp(gp.kernel_.theta)
        Lambda = np.diag(np.array([lambda_1, lambda_2])) ** 2
        Lambda_inv = np.linalg.pinv(Lambda)
        Ky = gp.L_.dot(gp.L_.T)
        chol_inv = np.linalg.pinv(gp.L_)
        Ky_inv = chol_inv.T.dot(chol_inv)
        beta = gp.alpha_.reshape(-1)

        # D: dim of input, N: num of training data points
        self.XU = gp.X_train_  # input training points
        self.N, self.D = self.XU.shape
        self.Ky = Ky              # NXN covvariance matrix
        self.Ky_inv = Ky_inv      # NXN covvariance matrix inverse
        self.beta = beta        # NX1
        self.alpha_sq = alpha_sq      # 1X1 variance scale of latent function
        self.Lambda = Lambda    # DXD length scale diagonal matrix
        self.Lambda_inv = Lambda_inv    # DXD length scale diagonal matrix inverse

        N = self.N
        D = self.D
        assert(self.D==2)               # for scalar system D=2
        assert(self.Ky.shape==(N,N))
        assert(self.Ky_inv.shape==(N,N))
        assert(self.beta.shape==(N,))
        assert(self.Lambda.shape==(D,D))
        assert(self.Lambda_inv.shape==(D,D))

    def predict_dynamics_1_step(self, mu_xu_t, sigma_xu_t):
        # mu_xu_t: current state-action mean
        # sigma_xu_t: current state-action covariance
        assert(mu_xu_t.shape==(self.D,))
        assert(sigma_xu_t.shape==(self.D,self.D))
        # # normal gp prediction
        # XU = self.XU
        # N = XU.shape[0]
        # V = XU - mu_xu_t.reshape(1, -1)
        # L = np.zeros(N)
        # for i in range(N):
        #     q1 = np.dot(self.Lambda_inv, V[i])
        #     q2 = np.dot(V[i], q1)
        #     exponent = -0.5 * q2
        #     exp_term = np.exp(exponent)
        #     L[i] = self.alpha_sq*exp_term
        # assert(L.shape==(N,))
        # mu_x_t1 = self.beta.dot(L)

        # uncertain gp prediction
        XU = self.XU
        L = np.zeros(self.N)
        V = XU - mu_xu_t.reshape(1,-1)
        q1 = sigma_xu_t.dot(self.Lambda_inv) + np.eye(self.D)
        alpha_term = (self.alpha_sq)*(np.linalg.det(q1)**-0.5)
        sig_Lamb_inv = np.linalg.pinv(sigma_xu_t + self.Lambda)
        for i in range(self.N):
            exponent = -0.5*V[i].dot(sig_Lamb_inv.dot(V[i]))
            exp_term = np.exp(exponent)
            L[i] = alpha_term*exp_term
        mu_x_t1 = self.beta.dot(L)

        Z = np.zeros((self.N,self.N,self.D))
        for i in range(self.N):
            for j in range(self.N):
                Z[i,j] = 0.5*(XU[i] + XU[j])

        L_ = np .zeros((self.N,self.N))
        q2 = np.linalg.pinv((sigma_xu_t + 0.5*self.Lambda))
        q3 = sigma_xu_t.dot(self.Lambda_inv)
        q4 = q2.dot(q3)
        q5 = 2.0*sigma_xu_t.dot(self.Lambda_inv) + np.eye(self.D)
        q6 = np.linalg.det(q5)**0.5
        for i in range(self.N):
            for j in range(self.N):
                z_u = Z[i,j] - mu_xu_t
                kxi = (self.alpha_sq)*np.exp(-0.5*(XU[i] - mu_xu_t).dot(self.Lambda_inv.dot(XU[i] - mu_xu_t)))
                kxj = (self.alpha_sq)*np.exp(-0.5*(XU[j] - mu_xu_t).dot(self.Lambda_inv.dot(XU[j] - mu_xu_t)))
                exponent = z_u.dot(q4.dot(z_u))
                exp_term = np.exp(exponent)
                L_[i,j] = kxi*kxj/q6*exp_term
        beta_term = self.beta.dot(L_.dot(self.beta))
        trace_term = np.trace(self.Ky_inv.dot(L_))
        sigma_x_t1 = beta_term + self.alpha_sq - trace_term - mu_x_t1**2
        if sigma_x_t1<0:
            sigma_x_t1=0.
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting it to 0.")
        # sigma_x_t1 = 0.
        return mu_x_t1, sigma_x_t1



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim_1d_params = {
        'x0': 0.,
        'xT': 5.,
        'a': .95,
        'b': 0.1,
        'L': -.2,
        'dt': 0.01,
        'T': 1.,
        'w_sigma': 1.0, # std dev
        'num_episodes': 20,
    }

    sim_1d_sys = sim_1d(sim_1d_params)
    traj_gt = sim_1d_sys.sim_episode(noise=False)
    num_episodes = sim_1d_params['num_episodes']
    traj_list = sim_1d_sys.sim_episodes(num_episodes)

    plt.figure()
    for i in range(num_episodes):
        plt.subplot(211)
        plt.plot(traj_list[i][:,0],traj_list[i][:,1])
        plt.subplot(212)
        plt.plot(traj_list[i][:,0],traj_list[i][:,2])
    plt.subplot(211)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(traj_gt[:,0],traj_gt[:,1],color='k')
    plt.subplot(212)
    plt.xlabel('t')
    plt.ylabel('u')
    plt.plot(traj_gt[:,0],traj_gt[:,2],color='k')
    plt.show()
