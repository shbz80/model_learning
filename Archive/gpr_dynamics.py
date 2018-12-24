import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import logging
import imp
import os
import os.path
import sys

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
# from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps import __file__ as gps_filepath
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS
from gps.proto.gps_pb2 import END_EFFECTOR_POINT_VELOCITIES
from gps.proto.gps_pb2 import JOINT_ANGLES
from gps.proto.gps_pb2 import JOINT_VELOCITIES
from gps.proto.gps_pb2 import ACTION

data_logger = DataLogger()

gps_filepath = os.path.abspath(gps_filepath)
gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
exp_name = 'yumi_robot_example_9_1'
exp_dir = gps_dir + 'experiments/' + exp_name + '/'
hyperparams_file = exp_dir + 'hyperparams.py'
hyperparams = imp.load_source('hyperparams', hyperparams_file)
config = hyperparams.config
_data_files_dir = config['common']['data_files_dir']

np.random.seed(1)

itr = 30
sample_num = 1 #sample number

traj_sample_lists = data_logger.unpickle(_data_files_dir +
    ('traj_sample_itr_%02d.pkl' % itr))

for s in range(sample_num):
    sample = traj_sample_lists[0][s]
    xp_pt = sample.get(END_EFFECTOR_POINTS)
    xv_pt = sample.get(END_EFFECTOR_POINT_VELOCITIES)
    jp_pt = sample.get(JOINT_ANGLES)
    jv_pt = sample.get(JOINT_VELOCITIES)
    jt_pt = sample.get(ACTION)

xo = np.concatenate((jp_pt,jv_pt),axis=1)
x = xo[0:-1,:] # x[t]
x1 = xo[1:,:] # x[t+1]
u = jt_pt[0:-1,:] # u[t]
xu = np.concatenate((x,u),axis=1) # function input
xu_max = np.amax(xu)
xu_min = np.amin(xu)
xu_n = np.divide((xu - xu_min),(xu_max - xu_min))
xd = x1-x # function output
xu_max = np.amax(xu,axis=0)
# xu_max = np.amax(xu,axis=0)[0].reshape(1)
# print "xu_max",xu_max.shape
xu_min = np.amin(xu,axis=0)
# xu_min = np.amin(xu,axis=0)[0].reshape(1)
# X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
X = xu
# X = xu_n[0::1]
# Xtest = xu_n[1::2]/100.
# X = xu[:,0].reshape(-1,1)
print "X",X.shape
# Observations
# y = f(X).ravel()
y = x1
# y = x1[0::1]
# ytest = x1[1::2]*100.
# y = x1[:,0].reshape(-1,1)
print "y",y.shape

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE

xq = np.zeros([100,21])
# xq = np.zeros([100,1])
# print "xq.shape",xq.shape
# print "xq",xq[:,0]
for i in range(xq.shape[1]):
    xq[:,i] = np.linspace(xu_min[i], xu_max[i], 100).T

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(xq, return_std=True)
# y_pred, sigma = gp.predict(X, return_std=True)
print y_pred.shape
print sigma.shape

# score = gp.score(Xtest,ytest)
# print 'score',score

# plt.figure(0)
# plt.plot(sigma)
#
#
# plt.figure(1)
# for i in range(7):
#     plt.subplot(2,7,1+i)
#     plt.title('j%dPos' %i)
#     plt.plot(y_pred[:,i],color='c')
#     # plt.subplot(2,7,8+i)
#     plt.plot(y[:,i],color='m')

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE

# plt.figure(0)
# plt.plot(X[:,0], y[:,0], 'r.', markersize=10, label=u'Observations')
#
#
# fig = plt.figure(1)
# plt.plot(X[:,0], y[:,0], 'r.', markersize=10, label=u'Observations')
# plt.plot(xq[:,0], y_pred[:,0], 'b-', label=u'Prediction')
# plt.fill(np.concatenate([xq[:,0], xq[:,0][::-1]]),
#          np.concatenate([y_pred[:,0] - 1.9600 * sigma,
#                         (y_pred[:,0] + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$xu$')
# plt.ylabel('$x_{t+1}$')
# # plt.ylim(-10, 20)
# plt.legend(loc='upper right')

# ----------------------------------------------------------------------


plt.show()
