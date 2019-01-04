import numpy as np
import warnings
import copy
from scipy.spatial.distance import pdist, squareform

class sim_1d(object):
    def __init__(self, params, type='cont', mode_seq=None, mode_num=None):
        self.x0 = params['x0']          # init state
        self.dt = params['dt']          # sample time
        self.T = params['T']            # total time in seconds
        self.type = type                # choose between continuous and discontinuous dynamics
        self.t = 0.
        self.mode_id = 0
        if type == 'cont':
            self.mode_c = copy.deepcopy(params['mode_c'])
            self.mode_num = 1
            self.mode_seq = ['mc']
        elif type == 'disc':
            self.mode_d = copy.deepcopy(params['mode_d'])
            if mode_seq is None:
                self.mode_num = mode_num
                mode_list = [key for key, val in self.mode_d.iteritems()]
                mode_seq = np.random.choice(mode_list, mode_num)
                self.mode_seq = list(mode_seq)
            else:
                self.mode_seq = mode_seq
                self.mode_num = len(mode_seq)
        self.reset()

    def init_cont_mode(self):
        self.mode_id = 0
        mode = self.mode_seq[self.mode_id]
        dynamics = self.mode_c[mode]['dynamics']
        self.a, self.b = dynamics
        self.w_sigma = self.mode_c[mode]['noise']
        mu_x0 = self.mode_c[mode]['range'][0]
        sigma_x0 = self.mode_c[mode]['init_x_var']
        self.x = np.random.normal(mu_x0, np.sqrt(sigma_x0))

    def set_disc_mode(self, mode_id):
        self.mode_id = mode_id
        mode = self.mode_seq[mode_id]
        dynamics = self.mode_d[mode]['dynamics']
        self.a, self.b = dynamics
        self.w_sigma = self.mode_d[mode]['noise']
        mu_x0 = self.mode_d[mode]['range'][0]
        sigma_x0 = self.mode_d[mode]['init_x_var']
        self.x = np.random.normal(mu_x0, np.sqrt(sigma_x0))

    def set_policy(self):
        if self.type == 'cont':
            self.xt = self.mode_c['mc']['target']
            self.L = self.mode_c['mc']['L']
        elif self.type == 'disc':
            mode = self.mode_seq[self.mode_id]
            self.xt = self.mode_d[mode]['target']
            self.L = self.mode_d[mode]['L']

    def reset(self):
        if self.type == 'cont':
            self.init_cont_mode()
        elif self.type == 'disc':
            self.set_disc_mode(0)
        self.mode_id = 0
        self.set_policy()
        self.dx = self.xt - self.x
        self.t = 0.
        self.x = 0.

    def transit_mode(self):
        x = self.x
        mode_seq = self.mode_seq
        mode_id = self.mode_id
        mode = mode_seq[mode_id]
        if mode_id > 0: prev_mode_id = mode_id - 1
        else: prev_mode_id = 0
        if mode_id < len(mode_seq)-1 : next_mode_id = mode_id +1
        else: next_mode_id = len(mode_seq)-1
        if (x >= self.mode_d[mode]['range'][1]) and (mode_id is not next_mode_id):
            self.set_disc_mode(next_mode_id)
        # elif (x < self.mode_d[mode]['range'][0]) and (mode_id is not prev_mode_id):
        #     self.set_disc_mode(prev_mode_id)

    def step(self, u):
        self.dx = self.xt - self.x
        mode = self.mode_seq[self.mode_id]

        # if mode == 'mc':
        #     dx_d = self.a * self.dx + self.b * u
        # if mode == 'm1':
        #     # dx_d = self.a * self.dx **2 + self.b * u
        #     dx_d = self.a * self.dx ** 2 + self.b * u
        # if mode == 'm2':
        #     dx_d = self.a * self.dx + self.b * u
        # if mode == 'm3':
        #     # dx_d = self.a * np.sin(self.dx) + self.b * u
        #     dx_d = self.a * np.sin(self.dx) + self.b * u
        dx_d = self.a * self.dx + self.b * u
        self.dx += dx_d*self.dt
        self.x = self.xt - self.dx
        self.t = self.t + self.dt

        if self.type == 'disc':
            self.transit_mode()
            self.set_policy()

    def act(self, dx):
        u = self.L * dx
        u_n = np.random.normal(self.L * dx, self.w_sigma)
        return u, u_n, self.w_sigma**2

    def sim_episode(self, noise=True):
        self.reset()
        N = len(np.arange(0,self.T,self.dt))
        traj = np.zeros((N,5))
        for i in range(N):
            u, u_n, w = self.act(self.dx)
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
                # exponent = z_u.dot(q4.dot(z_u))
                # exp_term = np.exp(exponent)
                # L_[i,j] = kxi*kxj/q6*exp_term
                k_term = kxi * kxj
                exponent = z_u.dot(q4.dot(z_u))
                # q7 = np.log(k_term) - np.log(q6) + exponent
                q7 = np.log(k_term/q6) + exponent
                L_[i, j] = np.exp(q7)
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
