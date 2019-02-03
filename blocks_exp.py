'''
blocks simulation and data processing for model learnig
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
from blocks_sim import MassSlideWorld

# np.random.seed(4)   # works for long term prediction - single mode
np.random.seed(1)

# logfile = "./Results/blocks_exp_raw_data_rs_4_2.p"
logfile = "./Results/blocks_exp_raw_data_disc_rs_1.p"
plot = True
# num_traj = num_samples  # first n samples to plot
num_traj = 10  # first n samples to plot

dt = 0.05
noise = 3.
exp_params = {
            'dt': dt,
            'T': 40,
            'num_samples': 15, # only even number, to be slit into 2 sets
            'dP': 1,
            'dV': 1,
            'dU': 1,
            'p0_var': 1e-4,
            'massSlide': {
                                'm': 1.,
                                'm_init_pos': 0.,
                                'mu': 0.5,
                                'fp_start': .5,
                                'stick_start': 1.,
                                'static_fric': 6.5,
                                'dt': dt,
            },
            'policy': {
                        'm1':{
                            'L': np.array([.2, 1.]),
                            # 'noise': 7.5*2,
                            'noise': noise,
                            'target': 18.,
                        },
                        'm2': {
                            'L': np.array([.2, 1.]),
                            # 'noise': 2.*2,
                            'noise': noise,
                            'target': 18.,
                        },
                        'm3':{
                            'L': np.array([.2, 1.]),
                            # 'noise': 10.*2,
                            'noise': noise,
                            'target': 18.,
                        },
                        'm4':{
                            'L': np.array([.2, 1.]),
                            # 'noise': 7.5*2,
                            'noise': noise,
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
t, X, mode = massSlideWorld.step(X0, u0)
Action.append(u0)
for i in range(1,T):
    un, u, _ = massSlideWorld.act(X, mode)
    t, X, mode = massSlideWorld.step(X, u)
    Pos.append(X[0])
    Vel.append(X[1])
    Action.append(u)
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
    t, X, mode = massSlideWorld.step(X0, un0)
    Action.append(un0)
    for i in range(1,T):
        # action = mean_action + np.random.normal(noise_mean,noise_std)
        un, _, _ = massSlideWorld.act(X, mode)
        t, X, mode = massSlideWorld.step(X, un)
        Pos.append(X[0])
        Vel.append(X[1])
        Action.append(un)
    p = np.array(Pos).reshape(T,dP)
    v = np.array(Vel).reshape(T,dV)
    pv = np.concatenate((p,v),axis=1)
    Xs[s,:,:] = pv
    Us[s,:,:] = np.array(Action).reshape(T,dU)

exp_data['X'] = Xs
exp_data['U'] = Us
# pickle.dump(exp_data, open("./Results/blocks_exp_raw_data_rs_4_disc.p", "wb"))
pickle.dump(exp_data, open(logfile, "wb"))

if plot:
    # plot samples
    plt.figure()
    plt.title('Position')
    plt.xlabel('t')
    plt.ylabel('q(t)')
    tm = np.linspace(0,T*dt,T)
    # plot positions
    plt.plot(tm, Xg[:,:dP], ls='-', marker='^')
    for s in range(num_traj):
        plt.plot(tm,Xs[s,:,0])
    plt.figure()
    plt.xlabel('t')
    plt.ylabel('q_dot(t)')
    plt.title('Velocity')
    plt.plot(tm, Xg[:,dP:dP+dV], ls='-', marker='^')
    # plot velocities
    for s in range(num_traj):
        plt.plot(tm,Xs[s,:,1])
    plt.figure()
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.title('Action')
    plt.plot(tm, Ug, ls='-', marker='^')
    # plot actions
    for s in range(num_traj):
        plt.plot(tm,Us[s,:,0])

    plt.show()
