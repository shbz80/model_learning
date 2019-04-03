import imp
import numpy as np
import pickle
from model_leraning_utils import YumiKinematics

from gps.utility.data_logger import DataLogger
from gps.proto.gps_pb2 import JOINT_ANGLES
from gps.proto.gps_pb2 import JOINT_VELOCITIES
from gps.proto.gps_pb2 import ACTION


# logfile = './Results/yumi_peg_exp_raw_data_34.p'
logfile = './Results/yumi_peg_exp_new_raw_data_train.p'

data_logger = DataLogger()

# gps_dir = '/home/shahbaz/Research/Software/Spyder_ws/gps_model_learning/'
# exp_name = 'yumi_model_learning_3'
gps_dir = '/home/shahbaz/Research/Software/Spyder_ws/gps/'
exp_name = 'yumi_model_learning_new_1'
itr = 0

exp_dir = gps_dir + 'experiments/' + exp_name + '/'
hyperparams_file = exp_dir + 'hyperparams.py'
hyperparams = imp.load_source('hyperparams', hyperparams_file)
config = hyperparams.config
_data_files_dir = config['common']['data_files_dir']

# use this for yumi_gps_generated.urdf
f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_gps_generated.urdf', 'r')
base_link = 'yumi_base_link'
end_link = 'gripper_l_base'
yumiKin = YumiKinematics(f, base_link, end_link, euler_string='szyx', reverse_angles=True)

traj_sample_lists = data_logger.unpickle(_data_files_dir +
    ('traj_sample_itr_%02d.pkl' % itr))
# alg_sample_lists = data_logger.unpickle(_data_files_dir +
#     ('algorithm_itr_%02d.pkl' % itr))
traj_sample_list = traj_sample_lists[0] # cond 0

num_samples = len(traj_sample_list)
Ts = traj_sample_list[0].T
Xs = []
Us = []
for sample in traj_sample_list:
    jp_pt = sample.get(JOINT_ANGLES)
    jv_pt = sample.get(JOINT_VELOCITIES)
    Xs.append(np.concatenate((jp_pt,jv_pt),axis=1))
    jt_pt = sample.get(ACTION)
    Us.append(jt_pt)

Xs = np.array(Xs)
Us = np.array(Us)

dP = 7
dV = 7
dU = 7
N, T, dX = Xs.shape
assert(T==Ts)
assert(Xs.shape[2]==(dP+dV))
assert(dP==dV)

Qts = Xs[:, :, :dP]
Qts_d = Xs[:, :, dP:dP+dV]
Uts = Us
Ets = np.zeros((N, T, 6))
Ets_d = np.zeros((N, T, 6))
Fts = np.zeros((N, T, 6))
for n in range(N):
    for i in range(T):
        Ets[n, i] = yumiKin.fwd_pose(Qts[n,i])
        J_A = yumiKin.get_analytical_jacobian(Qts[n,i])
        Ets_d[n,i] = J_A.dot(Qts_d[n,i])
        Fts[n,i] = np.linalg.pinv(J_A.T).dot(Uts[n,i])

EXts = np.concatenate((Ets,Ets_d),axis=2)


exp_params = {}
exp_params['dP'] = 7
exp_params['dV'] = 7
exp_params['dU'] = 7
exp_params['dt'] = 0.05
exp_params['T'] = T
exp_params['num_samples'] = N



exp_data={}
exp_data['exp_params'] = exp_params
exp_data['X'] = Xs
exp_data['U'] = Us
exp_data['EX'] = EXts
exp_data['F'] = Fts
# pickle.dump( exp_data, open( logfile, "wb" ) )
