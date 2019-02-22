'''
blocks simulation and data processing for model learnig
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
from blocks_sim import MassSlideWorld

np.random.seed(1)       # both train and test has both modes, train: 6 block+4 slide, test: 3 block+2slide
# the following seeds go with 'static_fric': 6.5
# np.random.seed(9)
# np.random.seed(15)
# np.random.seed(19)
# np.random.seed(23)

logfile = "./Results/blocks_exp_raw_data_rs_1.p"

plot = True
# num_traj = num_samples  # first n samples to plot
n_train = 10  # first n samples to plot
n_test = 5

dt = 0.05
noise_pol = 3.      # variance
noise_obs =1e-3      # variance

exp_params = {
            'dt': dt,
            'T': 40,
            'num_samples': n_train + n_test, # only even number, to be slit into 2 sets
            'dP': 1,
            'dV': 1,
            'dU': 1,
            'p0_var': 1e-4, # initial position variance
            'massSlide': {    # for 4 mode case
                                'm': 1.,
                                'm_init_pos': 0.,
                                'mu_1': 0.5,
                                'mu_2': 0.1,
                                'fp_start': .5,
                                'stick_start': 1.,
                                # 'static_fric': 6.5,
                                'static_fric': 6.6,
                                'dt': dt,
                                'noise_obs': noise_obs,
            },
            'policy': {
                        'm1':{
                            'L': np.array([.2, 1.]),
                            # 'noise_pol': 7.5*2,
                            'noise_pol': noise_pol,
                            'target': 18.,
                        },
                        'm2': {
                            'L': np.array([.2, 1.]),
                            # 'noise_pol': 2.*2,
                            'noise_pol': noise_pol,
                            'target': 18.,
                        },
                        'm3':{
                            'L': np.array([.2, 1.]),
                            # 'noise_pol': 10.*2,
                            'noise_pol': noise_pol,
                            'target': 18.,
                        },
                        'm4':{
                            'L': np.array([.2, 1.]),
                            # 'noise_pol': 7.5*2,
                            'noise_pol': noise_pol,
                            'target': 18.,
                        },
            },

}

num_samples = exp_params['num_samples']
dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']
T = exp_params['T']
massSlideParams = exp_params['massSlide']
p0_mu = massSlideParams['m_init_pos']
p0_var = exp_params['p0_var']
policy_params = exp_params['policy']

massSlideWorld = MassSlideWorld(**massSlideParams)
massSlideWorld.set_policy(policy_params)

dX = dP+dV
Xs = np.zeros((num_samples,T,dX))
Us = np.zeros((num_samples,T,dU))

exp_data = {}
exp_data['exp_params'] = exp_params

# ground truth sim (without noise)
massSlideWorld.reset()
Pos=[]
Vel=[]
Action=[]
p0 = p0_mu
X0 = np.array([p0, 0.])
mode0 = 'm1'
Pos.append(p0)
Vel.append(0.)
un0,u0, _ = massSlideWorld.act(X0, mode0)
Action.append(u0)
t, X, mode = massSlideWorld.step(X0, u0)
for i in range(1,T):
    Pos.append(X[0])
    Vel.append(X[1])
    un, u, _ = massSlideWorld.act(X, mode)
    Action.append(u)
    t, X, mode = massSlideWorld.step(X, u)

p = np.array(Pos).reshape(T,dP)
v = np.array(Vel).reshape(T,dV)
Xg = np.concatenate((p,v),axis=1)
Ug = np.array(Action).reshape(T,dU)
exp_data['Xg'] = Xg
exp_data['Ug'] = Ug

for s in range(num_samples):
    massSlideWorld.reset()
    Pos=[]
    Vel=[]
    Action=[]
    p0 = np.random.normal(p0_mu, np.sqrt(p0_var))
    X0 = np.array([p0, 0.])
    Pos.append(p0)
    Vel.append(0.)
    un0, _, _ = massSlideWorld.act(X0, mode0)
    Action.append(un0)
    t, X, mode = massSlideWorld.step(X0, un0)
    for i in range(1,T):
        Pos.append(X[0])
        Vel.append(X[1])
        un, _, _ = massSlideWorld.act(X, mode)
        Action.append(un)
        t, X, mode = massSlideWorld.step(X, un)

    p = np.array(Pos).reshape(T,dP)
    v = np.array(Vel).reshape(T,dV)
    pv = np.concatenate((p,v),axis=1)
    Xs[s,:,:] = pv
    Us[s,:,:] = np.array(Action).reshape(T,dU)

exp_data['X'] = Xs
exp_data['U'] = Us
pickle.dump(exp_data, open(logfile, "wb"))

if plot:
    # plot samples
    plt.figure()
    plt.title('Train')
    plt.subplot(131)
    plt.title('Position')
    plt.xlabel('t')
    plt.ylabel('q(t)')
    tm = np.linspace(0,T*dt,T)
    # plot positions
    plt.plot(tm, Xg[:,:dP], ls='-', marker='^')
    for s in range(n_train):
        plt.plot(tm,Xs[s,:,0])
    plt.subplot(132)
    plt.xlabel('t')
    plt.ylabel('q_dot(t)')
    plt.title('Velocity')
    plt.plot(tm, Xg[:,dP:dP+dV], ls='-', marker='^')
    # plot velocities
    for s in range(n_train):
        plt.plot(tm,Xs[s,:,1])
    plt.subplot(133)
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.title('Action')
    plt.plot(tm, Ug, ls='-', marker='^')
    # plot actions
    for s in range(n_train):
        plt.plot(tm,Us[s,:,0])

    plt.figure()
    plt.title('Test')
    plt.subplot(131)
    plt.title('Position')
    plt.xlabel('t')
    plt.ylabel('q(t)')
    tm = np.linspace(0, T * dt, T)
    # plot positions
    plt.plot(tm, Xg[:, :dP], ls='-', marker='^')
    for s in range(n_train,n_train+n_test):
        plt.plot(tm, Xs[s, :, 0])
    plt.subplot(132)
    plt.xlabel('t')
    plt.ylabel('q_dot(t)')
    plt.title('Velocity')
    plt.plot(tm, Xg[:, dP:dP + dV], ls='-', marker='^')
    # plot velocities
    for s in range(n_train,n_train+n_test):
        plt.plot(tm, Xs[s, :, 1])
    plt.subplot(133)
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.title('Action')
    plt.plot(tm, Ug, ls='-', marker='^')
    # plot actions
    for s in range(n_train,n_train+n_test):
        plt.plot(tm, Us[s, :, 0])

    plt.show()
