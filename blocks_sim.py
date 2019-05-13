import numpy as np

class MassSlideWorld(object):
    def __init__(self, m=1., m_init_pos=0, mu_1=0.05, mu_2=0.05, slip_start=0., fp_start=7., stick_start = 10., static_fric = 5., dt=0.01, noise_obs=np.zeros(2)):
        self.m = m
        self.m_init_pos = m_init_pos
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.dt = dt
        self.slip_start = slip_start
        self.fp_start = fp_start
        self.stick_start = stick_start
        self.static_fric = static_fric
        self.noise_obs = noise_obs
        self.reset()

    def dynamics(self, X, m, b, k, u):
        M = np.array([[0, 1], [-k/m, -b/m]])
        U = np.array([0, u/m])
        return M.dot(X) + U

    def reset(self):
        self.X = np.zeros(2)
        self.t = 0
        self.mode = 'm1'

    def step_mode(self, X, u):
        if X[0] < self.stick_start:  # mode 1: free motion
        # if X[0] < self.fp_start:  # mode 1: free motion
            self.mode = 'm1'
        elif (X[0] < self.stick_start) and (X[0] >= self.fp_start):  # mode 2: friction
            self.mode = 'm2'
        elif (self.mode is not 'm3') and (self.mode is not 'm4') and (X[0] >= self.stick_start):  # mode 3: sticking/contact
            self.mode = 'm3'
            X[1] = 0.
        elif (self.mode is 'm3') and (u <= self.static_fric): # mode 4: sticking/contact
            X[1] = 0.
            self.mode = 'm3'
        elif (self.mode is 'm3') and (u > self.static_fric): # mode 4: slipping
            self.mode = 'm4'
            X[1]=5.0
        elif self.mode == 'm4':
            None
        else:
            assert(False)
        #     # if X[0] < self.stick_start:  # mode 1: free motion
        # if X[0] < self.slip_start:  # mode 2: friction
        #     self.mode = 'm2'
        # elif (X[0] < self.stick_start) and (X[0] >= self.slip_start):  # mode 2: slip
        #     if self.mode != 'm1':
        #         X[1]= 3.
        #     self.mode = 'm1'
        # elif (self.mode is not 'm3') and (self.mode is not 'm4') and (
        #         X[0] >= self.stick_start):  # mode 3: sticking/contact
        #     self.mode = 'm3'
        #     X[1] = 0.
        # elif (self.mode is 'm3') and (u <= self.static_fric):  # mode 4: sticking/contact
        #     X[1] = 0.
        #     self.mode = 'm3'
        # elif (self.mode is 'm3') and (u > self.static_fric):  # mode 4: slipping
        #     self.mode = 'm4'
        #     X[1] = 1.5
        # elif self.mode == 'm4':
        #     None
        # else:
        #     assert (False)
        return self.mode, X
        # return 'm1', X

    def step(self, X, u):
        # X = self.X
        dt = self.dt
        mode = self.mode
        if mode=='m1': # mode 1: free motion
            b = 0.
            k = 0.
            m = self.m
            f = u
        elif mode=='m2': # mode 2: friction
            m = self.m
            N = m * 9.8
            b = N * self.mu_1 * X[1]
            k = 0.
            f = u
        elif mode=='m3': # mode 3: sticking
            m = self.m
            b = 0.
            k = 0.
            f = 0.
        elif mode=='m4':    # mode 4: slipping
            m = self.m
            N = m * 9.8
            b = N * self.mu_2 * X[1]
            # b = 0.
            k = 0.
            f = u

        k1 = self.dynamics(X,m,b,k,f)
        k2 = self.dynamics(X+0.5*dt*k1,m,b,k,f)
        k3 = self.dynamics(X+0.5*dt*k2,m,b,k,f)
        k4 = self.dynamics(X+dt*k3,m,b,k,f)
        X = X + dt*(1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)

        self.t = self.t + dt
        self.mode, self.X = self.step_mode(X, u)
        self.X[0] = np.random.normal(self.X[0], np.sqrt(self.noise_obs[0]))
        self.X[1] = np.random.normal(self.X[1], np.sqrt(self.noise_obs[1]))
        return self.t, self.X, self.mode

    def set_policy(self, policy_params):
        self.policy_params = policy_params

    def act(self, X, mode):
        # mode = self.mode
        mode_policy = self.policy_params[mode]
        L = mode_policy['L']
        Xtrg =  mode_policy['target']
        noise_pol = mode_policy['noise_pol']
        dX = np.array([Xtrg, 0.]) - X
        u = L.dot(dX)
        un = np.random.normal(u, np.sqrt(noise_pol))
        return un, u, noise_pol

    def predict(self, X, return_std=True):
        '''
        this methods is made to provide compatibility to gpr.predict API
        for unscented particle method propogation
        :param X:
        :param return_std:
        :return:
        '''
        assert(X.shape[0]>=1)
        assert (X.shape[1] == 2)
        mode = 'm1' # for control purpose only 1 mode, all modes are the same
        mode_policy = self.policy_params[mode]
        L = mode_policy['L']
        Xtrg =  mode_policy['target']
        noise_pol = mode_policy['noise_pol']
        dX = np.array([Xtrg, 0.]).reshape(1,2) - X
        U = np.dot(dX, L)
        U = U.reshape(X.shape[0],1)
        if return_std:
            U_noise = np.full((U.shape), np.sqrt(noise_pol))
        return U, U_noise
