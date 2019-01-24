import numpy as np

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
        self.mode = 'm1'

    def step_mode(self, X):
        if (X[0] < self.block):
            if X[0] < self.m2_init_pos:  # mode 1
                self.mode = 'm1'
            elif (X[0] < self.fp_start) and (X[0] >= self.m2_init_pos):  # mode 2
                self.mode = 'm2'
            elif (X[0] < self.fp_end) and (X[0] >= self.fp_start):  # mode 3
                self.mode = 'm3'
            else:
                assert(False)
        if (X[0] >= self.block):  # mode 4
            self.mode = 'm4'
        return self.mode

    def step(self, X, u):
        # X = self.X
        dt = self.dt
        mode = self.step_mode(X)
        if mode=='m4': # mode 4
            X[1] = 0.
            X[0] = self.block
        else:
            if mode=='m1': # mode 1
                self.prev_m = self.m
                m, b = self.m1, 0
                self.m = m
                X[1] = self.prev_m/self.m * X[1]
            elif mode=='m2': # mode 2
                self.prev_m = self.m
                m = self.m1 + self.m2
                self.m = m
                X[1] = self.prev_m/self.m * X[1]
                b = 0
            elif mode=='m3': # mode 3
                self.prev_m = self.m
                m = self.m1 + self.m2
                self.m = m
                X[1] = self.prev_m/self.m * X[1]
                N = m*9.8
                b = N*self.mu*X[1]
            k1 = self.dynamics(X,m,b,u)
            k2 = self.dynamics(X+0.5*dt*k1,m,b,u)
            k3 = self.dynamics(X+0.5*dt*k2,m,b,u)
            k4 = self.dynamics(X+dt*k3,m,b,u)
            X = X + dt*(1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)

        self.t = self.t + dt
        self.X = X
        return self.t, X

    def set_policy(self, policy_params):
        self.policy_params = policy_params

    def act(self, mode, X):
        mode_policy = self.policy_params[mode]
        L = mode_policy['L']
        Xtrg =  mode_policy['target']
        noise = mode_policy['noise']
        dX = np.array([Xtrg, 0.]).reshape(-1,1) - X
        u = L.dot(dX)
        # Xtgt = np.array([17., 0.]).reshape(2,1)
        # dX = Xtgt - X
        # L = np.array([1., 0.])
        # u = 18L.dot(dX)
        # noise = 0.
        un = np.random.normal(u, np.sqrt(noise))
        return un, u, noise
