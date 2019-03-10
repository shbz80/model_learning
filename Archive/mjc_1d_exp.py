'''
mujo agent for model learnig research
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from Archive.utilities import MassSlideWorld

# Add gps/python to path so that imports work.
sys.path.append('/home/shahbaz/Research/Software/Spyder_ws/gps/python')
# from gps.gui.gps_training_gui import GPSTrainingGUI
# from gps.utility.data_logger import DataLogger

from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
    ACTION

SENSOR_DIMS = {
    JOINT_ANGLES: 1,
    JOINT_VELOCITIES: 1,
    ACTION: 1,
}

common = {
    'conditions': 1,
}

agent_hyperparams = {
    'type': AgentMuJoCo,
    # 'filename': '/home/shahbaz/Research/Software/Spyder_ws/gps/mjc_models/yumi_right_peg_mjcf.xml',
    'filename': '/home/shahbaz/Research/Software/model_learning/model_learning_1d.xml',
    'x0': np.array([0., 0.]),
    'dt': 0.02,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [[np.array([0, 0, 0])]],
    'T': 150,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [],
    'camera_pos': np.array([5.0, 0.5, 3., 0., 0., 0.]),
}

exp_params = {
            'dt': agent_hyperparams['dt'],
            'T': agent_hyperparams['T'],
            'num_samples': 20, # only even number, to be slit into 2 sets
            'dP': SENSOR_DIMS[JOINT_ANGLES],
            'dV': SENSOR_DIMS[JOINT_VELOCITIES],
            'dU': SENSOR_DIMS[ACTION],
            'noise_gain': 7.5,
            # 'noise_gain': 0.,
            'mean_action': 15.,

}

num_samples = exp_params['num_samples']
cond = 0
dU = SENSOR_DIMS[ACTION]
noise_gain = exp_params['noise_gain']
noise_mean = np.zeros(dU)
noise_std = np.ones(dU)*noise_gain
mean_action = exp_params['mean_action']
T = exp_params['T']
dt = exp_params['dt']


class Policy(object):
    def act(self, x, obs, t, noise=None):
        # return np.ones(dU)*mean_action + np.random.normal(noise_mean,noise_std)
        return np.ones(dU)*mean_action + noise_gain*noise

massSlideWorld = MassSlideWorld(m1=1., m1_init_pos=0, m2=1., m2_init_pos=7., mu=0.05, fp_start=12.,fp_end=15., block=15., dt=dt)

pol = Policy()
# mjc_agent = AgentMuJoCo(agent_hyperparams)
# for i in range(num_samples):
#     mjc_agent.sample(
#                     pol, cond,
#                     verbose=True,
#                     noisy=True
#                 )

# SampleList = mjc_agent.get_samples(cond, -num_samples)
# mjc_agent.clear_samples()
# X = SampleList.get_X()
# U = SampleList.get_U()

dX = 2
dU = 1
Xs = np.zeros((num_samples,T,dX))
Us = np.zeros((num_samples,T,dU))
for s in range(num_samples):
    massSlideWorld.reset()
    Pos=[]
    Vel=[]
    Action=[]
    for i in range(T):
        action = mean_action + np.random.normal(noise_mean,noise_std)
        t, X = massSlideWorld.step(action)
        Pos.append(X[0])
        Vel.append(X[1])
        Action.append(action)
    p = np.array(Pos).reshape(T,-1)
    v = np.array(Vel).reshape(T,-1)
    pv = np.concatenate((np.array(p),np.array(v)),axis=1)
    Xs[s,:,:] = pv
    Us[s,:,:] = np.array(Action).reshape(T,-1)

# plot samples
plt.figure()
tm = np.linspace(0,T*dt,T)
# plot positions
for s in range(num_samples):
    plt.plot(tm,Xs[s,:,0])
plt.figure()
# plot velocities
for s in range(num_samples):
    plt.plot(tm,Xs[s,:,1])
plt.figure()
# plot actions
for s in range(num_samples):
    plt.plot(tm,Us[s,:,0])
plt.show()

sample_data = {}
sample_data['exp_params'] = exp_params
sample_data['X'] = Xs
sample_data['U'] = Us

pickle.dump( sample_data, open( "mjc_1d_4mode_raw.p", "wb" ) )
