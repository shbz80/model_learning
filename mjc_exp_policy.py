import numpy as np
import sys
import copy
from model_leraning_utils import YumiKinematics

sys.path.append('/home/shahbaz/Research/Software/Spyder_ws/gps/python')
from gps.agent.agent_utils import generate_noise
import scipy as sp

# use this for yumi_ABB_left.urdf
f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')
base_link = 'world'
end_link = 'left_gripper_base'
yumiKin = YumiKinematics(f, base_link, end_link, euler_string='szyx', reverse_angles=True)

# # use this for yumi_gps_generated.urdf
# f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_gps_generated.urdf', 'r')
# base_link = 'yumi_base_link'
# end_link = 'gripper_l_base'
# yumiKin = YumiKinematics(f, base_link, end_link, euler_string='szyx', reverse_angles=True)
# # yumiKin = YumiKinematics(f, base_link, end_link, euler_string='sxyz', reverse_angles=False)

class Policy(object):
    def __init__(self, agent_params, exp_params):
        self.agent_params = agent_params
        self.exp_params = exp_params
        self.noise_gain = exp_params['noise_gain']
        self.joint_space_noise = exp_params['joint_space_noise']
        # self.noise_gain = 0

        self.ref_x_traj = []
        self.curr_x_traj = []
        self.ref_q_traj = []

        dP = self.exp_params['dP']
        dV = self.exp_params['dV']
        dU = self.exp_params['dU']
        self.dt = exp_params['dt']
        self.T = exp_params['T']
        self.Tc = self.T/exp_params['t_contact_factor'] # time till contact
        self.Kp = exp_params['Kp']
        self.Kd = exp_params['Kd']
        self.Kpx = exp_params['Kpx']

        # init q, q_dot
        self.init_q = self.exp_params['x0'][:dP]
        self.init_q_dot = self.exp_params['x0'][dP:]
        self.targ_x_delta = self.exp_params['target_x_delta']
        self.target_x = self.exp_params['target_x']

        self.curr_q = self.init_q
        self.curr_q_dot = self.init_q_dot

        self.init_x = yumiKin.fwd_pose(self.init_q)

        self.target_x = self.init_x + self.targ_x_delta
        # self.targ_x_delta = self.target_x - self.init_x

        # self.ref_x_dot_d = (self.target_x - self.init_x)/float(self.Tc)/self.dt
        self.ref_x_dot_d = np.zeros(6)
        self.ref_x = self.init_x

        self.ref_q = self.init_q
        self.t = 0
        # self.expl_noise_gain = np.array([1., 1., .8, .5, .1, .1, .1])
        self.ref_x_dot_noise = generate_noise(self.T,6,self.agent_params)
        self.ref_x_dot_noise_mask = np.array([1., 1., 0, 0, 0, 0])
        self.ref_x_dot = np.zeros(6)
        # raw_input()

    def act(self, x, obs, t, noise=None, joint_noise=True):
        dP = self.exp_params['dP']
        Kp = self.Kp
        Kd = self.Kd
        Kpx = self.Kpx


        self.curr_q = x[:dP]
        self.curr_q_dot = x[dP:]

        self.curr_x = yumiKin.fwd_pose(self.curr_q)
        self.curr_x_traj.append(copy.deepcopy(self.curr_x))

        mask = self.ref_x_dot_noise_mask
        # print mask
        common_gain = self.noise_gain
        noise = self.ref_x_dot_noise[t,:]

        if np.abs(self.ref_x[2] - self.target_x[2]) > 0.005:
            self.ref_x_dot_d[2] = self.targ_x_delta[2] / float(self.Tc) / self.dt
        elif np.abs(self.ref_x[1]-self.target_x[1]) > 0.005:
            self.ref_x_dot_d[2] = 0.0
            self.ref_x_dot_d[1] = self.targ_x_delta[1] / float(self.Tc) / self.dt
        elif np.abs(self.ref_x[0] - self.target_x[0]) > 0.005:
            self.ref_x_dot_d[2] = 0.0
            self.ref_x_dot_d[1] = 0.0
            self.ref_x_dot_d[0] = self.targ_x_delta[0] / float(self.Tc) / self.dt
        else:
            self.ref_x_dot_d[0] = 0.0
            self.ref_x_dot_d[1] = 0.0
            self.ref_x_dot_d[2] = 0.0

        self.ref_x_dot = self.ref_x_dot_d + mask*noise*common_gain
        self.ref_x += self.ref_x_dot*self.dt
        self.error_x = self.ref_x - self.curr_x
        # self.error_x[3:] = 0.
        self.ref_x_traj.append(copy.deepcopy(self.ref_x))

        J_A = yumiKin.get_analytical_jacobian(self.curr_q)
        b = self.ref_x_dot + np.diag(Kpx).dot(self.error_x)
        ref_q_dot, _, _, _ = sp.linalg.lstsq(J_A, b, lapack_driver='gelsd')

        # J_A_inv = np.linalg.pinv(J_A)
        # ref_q_dot = J_A_inv.dot(b)

        # ref_q_dot[3:] = 0.
        self.ref_q = self.ref_q + ref_q_dot*self.dt
        self.ref_q_traj.append(copy.deepcopy(self.ref_q))

        err = self.ref_q - self.curr_q
        err_dot = ref_q_dot - self.curr_q_dot
        u = np.diag(Kp).dot(err) + np.diag(Kd).dot(err_dot)

        self.t += 1
        u_noise = np.zeros(7)
        if self.joint_space_noise is not None:
            u_noise = np.random.normal(np.zeros(7), self.Kp*np.sqrt(self.joint_space_noise))

        return u + u_noise

    def get_traj_data(self):
        return self.ref_x_traj, self.curr_x_traj, self.ref_q_traj


    def predict(self, X, t, return_std=True):
        dP = self.exp_params['dP']
        dU = self.exp_params['dU']
        # Kp = np.array([.15, .15, .12, .075, .05, .05, .05])
        Kp = self.Kp
        Kd = self.Kd
        # Kpx = np.array([.5, .5, .5, .5, .5, .5])
        Kpx = self.Kpx

        U = np.zeros((X.shape[0],dU))
        U_std = np.zeros((X.shape[0], dU))

        if self.joint_space_noise is not None:
            U_std[:, :] = self.Kp*np.sqrt(self.joint_space_noise)

        mask = self.ref_x_dot_noise_mask
        # print mask
        common_gain = self.noise_gain
        noise = self.ref_x_dot_noise[t, :]
        ########## TODO: remove after debugging
        # noise = np.array([0.0176405, 0.00400157, 0., 0., 0., 0.])/common_gain
        ##########

        if np.abs(self.ref_x[2] - self.target_x[2]) > 0.005:
            self.ref_x_dot_d[2] = self.targ_x_delta[2] / float(self.Tc) / self.dt
        elif np.abs(self.ref_x[1] - self.target_x[1]) > 0.005:
            self.ref_x_dot_d[2] = 0.0
            self.ref_x_dot_d[1] = self.targ_x_delta[1] / float(self.Tc) / self.dt
        # elif np.abs(self.ref_x[0] - self.target_x[0]) > 0.005:
        #     self.ref_x_dot_d[2] = 0.0
        #     self.ref_x_dot_d[1] = 0.0
        #     self.ref_x_dot_d[0] = self.targ_x_delta[0] / float(self.Tc) / self.dt
        else:
            self.ref_x_dot_d[0] = 0.0
            self.ref_x_dot_d[1] = 0.0
            self.ref_x_dot_d[2] = 0.0

        self.ref_x_dot = self.ref_x_dot_d + mask * noise * common_gain
        self.ref_x += self.ref_x_dot * self.dt

        for i in range(X.shape[0]):
            x = X[i]
            curr_q = x[:dP]
            curr_q_dot = x[dP:]
            curr_x = yumiKin.fwd_pose(curr_q)
            error_x = self.ref_x - curr_x
            J_A = yumiKin.get_analytical_jacobian(curr_q)

            # J_G1 = np.array([[-0.138111,     0.159612,    -0.162782,   -0.0614139,    0.0338992,     -0.03541, -6.31548e-18],
            #                 [0.351635,    -0.130027,    -0.110888,     -0.26626,    0.0562792,    0.0213979, -2.51603e-17],
            #                 [-0.0167584,     0.207352,    -0.287125,     0.132814,  0.000146982,   -0.0264247, -1.61254e-19],
            #                 [0.813801,     0.579203,     0.128172,    -0.315976,     0.852908,     0.515967,   -0.0108093],
            #                 [0.342026,    -0.411338,      0.89857,    -0.364132,    -0.513494,     0.856605,   0.00912225],
            #                 [0.469837,    -0.703793,    -0.419694,    -0.876109,   -0.0941876,   0.00223717,      -0.9999]])
            # J_A1 = J_G_to_A(J_G1, curr_x[3:])

            # J_A2 = np.array([[-0.138111,     0.159612,    -0.162782,   -0.0614139,    0.0338992,     -0.03541, -6.31548e-18],
            #                 [0.351635,    -0.130027,    -0.110888,     -0.26626,    0.0562792,    0.0213979, -2.51603e-17],
            #                 [-0.0167584,     0.207352,    -0.287125,     0.132814,  0.000146982,   -0.0264247, -1.61254e-19],
            #                 [-27.9022,     -50.7503,      33.6251,     -0.40748,     -69.5856,      11.1827,  7.77156e-16],
            #                 [-0.786246,   -0.0592034,    -0.769374,     0.482068,    -0.157659,    -0.987412, -8.67362e-19],
            #                 [-28.3748,     -50.0515,      34.0482,     0.468676,     -69.4984,      11.1816,            1.]])

            # J_A_inv = np.linalg.pinv(J_A)
            # ref_q_dot = J_A_inv.dot(self.ref_x_dot + np.diag(Kpx).dot(error_x))

            b = self.ref_x_dot + np.diag(Kpx).dot(error_x)
            # b1 = np.array([0.018557, 0.00471871, -0.0623581, -0.356441, -0.00250345, -0.366273])
            ref_q_dot, _, _, _ = sp.linalg.lstsq(J_A, b, lapack_driver='gelsd')
            # ref_q_dot1, _, _, _ = sp.linalg.lstsq(J_A1, b, lapack_driver='gelsd')
            # ref_q_dot2, _, _, _ = sp.linalg.lstsq(J_A2, b, lapack_driver='gelsd')

            # ref_q_dot[-1] = 0.
            ref_q = self.ref_q + ref_q_dot * self.dt
            if i==0:
                self.ref_q = self.ref_q + ref_q_dot * self.dt

            err = ref_q - curr_q
            err_dot = ref_q_dot - curr_q_dot
            u = np.diag(Kp).dot(err) + np.diag(Kd).dot(err_dot)
            U[i] = u

        return U, U_std

class SimplePolicy(object):
    def __init__(self, Xrs, Us, params):
        self.Xr = np.mean(Xrs, axis=0)
        self.U_var = np.var(Us, axis=0)
        self.kp = params['Kp']
        self.kd = params['Kd']
        self.dP = params['dP']
        self.dU = params['dU']

    def act(self, x, t):
        ex = (self.Xr[t] - x)
        eq = ex[:self.dP]
        eqd = ex[self.dP:]
        u = np.diag(self.kp).dot(eq) + np.diag(self.kd).dot(eqd)
        un = np.random.normal(u, np.sqrt(self.U_var[t]))
        return un, u

    def predict(self, X, t, return_std=True):
        U = np.zeros((X.shape[0], self.dU))
        U_noise = np.zeros((X.shape[0], self.dU))
        for i in range(X.shape[0]):
            _, U[i] = self.act(X[i], t)
            U_noise[i] = np.sqrt(self.U_var[t])
        if return_std:
            return U, U_noise
        else:
            return U



