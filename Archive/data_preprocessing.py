import pickle
# import pydart2 as pydart
from model_leraning_utils import YumiKinematics

#import PyKDL as kdl
from pykdl_utils.kdl_kinematics import *

f = file('/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf', 'r')
# euler_from_matrix = pydart.utils.transformations.euler_from_matrix
euler_from_matrix = trans.euler_from_matrix
# J_G_to_A = jacobian_geometric_to_analytic
J_G_to_A = YumiKinematics.jacobian_geometric_to_analytic

#pykdl stuff
robot = Robot.from_xml_string(f.read())
base_link = robot.get_root()
# end_link = 'left_tool0'
end_link = 'left_contact_point'
kdl_kin = KDLKinematics(robot, base_link, end_link)

sample_data = pickle.load( open( "mjc_blocks_raw_10_10.p", "rb" ) )
Xs = sample_data['X']
Us = sample_data['U']
exp_params = sample_data['exp_params']
num_samples = exp_params['num_samples']
dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']
N, T, dX = Xs.shape
train_data_id = num_samples/2
assert(Xs.shape[2]==(dP+dV))
assert(dP==dV)

XU = np.zeros((N,T,dX+dU))
for n in range(N):
    XU[n] = np.concatenate((Xs[n,:,:],Us[n,:,:]),axis=1)
XU_t = XU[:,:-1,:]
X_t1 = XU[:,1:,:dX]
X_t = XU[:,:-1,:dX]
delX = X_t1 - X_t
dynamics_data = np.concatenate((XU_t,X_t1),axis=2)
train_data = dynamics_data[0:train_data_id,:,:]
test_data = dynamics_data[train_data_id:,:,:]

train_data_flattened = train_data.reshape((-1,train_data.shape[-1]))
# X_train = train_data_flattened[:,0:dP+dV]
data_train = train_data_flattened

Qt = data_train[:,0:dP].reshape((-1,dP))
Qt_d = data_train[:,dP:dP+dV].reshape((-1,dV))
Ut = data_train[:,dP+dV:dP+dV+dU].reshape((-1,dU))
Qt_1 = data_train[:,dP+dV+dU:dP+dV+dU+dP].reshape((-1,dP))
Qt_1_d = data_train[:,dP+dV+dU+dP:dP+dV+dU+dP+dV].reshape((-1,dV))
Xt = np.concatenate((Qt,Qt_d),axis=1)
Xt_1 = np.concatenate((Qt_1,Qt_1_d),axis=1)
Et = np.zeros((Qt.shape[0],6))
Et_d = np.zeros((Qt.shape[0],6))
Et_1 = np.zeros((Qt.shape[0],6))
Et_1_d = np.zeros((Qt.shape[0],6))
Ft = np.zeros((Qt.shape[0],6))

for i in range(Qt.shape[0]):
    Tr = kdl_kin.forward(Qt[i], end_link=end_link, base_link=base_link)
    epos = np.array(Tr[:3,3])
    epos = epos.reshape(-1)
    erot = np.array(Tr[:3,:3])
    erot = euler_from_matrix(erot)
    Et[i] = np.append(epos,erot)
    Tr = kdl_kin.forward(Qt_1[i], end_link=end_link, base_link=base_link)
    epos = np.array(Tr[:3,3])
    epos = epos.reshape(-1)
    erot = np.array(Tr[:3,:3])
    erot = euler_from_matrix(erot)
    Et_1[i] = np.append(epos,erot)

    J_G = np.array(kdl_kin.jacobian(Qt[i]))
    J_G = J_G.reshape((6,7))
    J_A = J_G_to_A(J_G, Et[i][3:])
    Et_d[i] = J_A.dot(Qt_d[i])
    J_G = np.array(kdl_kin.jacobian(Qt_1[i]))
    J_G = J_G.reshape((6,7))
    J_A = J_G_to_A(J_G, Et_1[i][3:])
    Et_1_d[i] = J_A.dot(Qt_1_d[i])
    Ft[i] = np.linalg.pinv(J_A.T).dot(Ut[i])

EXt = np.concatenate((Et,Et_d),axis=1)
EXt_1 = np.concatenate((Et_1,Et_1_d),axis=1)


sample_data['train_data'] = train_data
sample_data['test_data'] = test_data
sample_data['EXt'] = EXt
sample_data['EXt_1'] = EXt_1
sample_data['Ft'] = Ft
sample_data['Xt'] = Xt
sample_data['Xt_1'] = Xt_1
sample_data['Ut'] = Ut
pickle.dump( sample_data, open( "mjc_blocks_processed_10_10.p", "wb" ) )
