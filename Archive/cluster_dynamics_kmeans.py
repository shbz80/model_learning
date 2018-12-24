import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
print "mat",mat.__version__
# from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
# mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
# import sys
import pydart2 as pydart
# import copy
# import argparse
# import threading
# import time
# import traceback

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
skel_path = "/home/shahbaz/Research/Software/Spyder_ws/gps/yumi_model/yumi_ABB_left.urdf"

def kernel_exp(a, b, w):
    """ GP squared exponential kernel """
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * w.dot(sqdist))

def kernel_dist(a, b, w):
    """ GP squared exponential kernel """
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return w.dot(sqdist)

pydart.init()
# print("pydart init OK")
world = pydart.World(0.0002)
# print("World init OK")
# world.g = [0.0, 0.0, -9.8]
world.g = [0.0, 0.0, 0] # gravity is set to zero
# print("gravity = %s" % str(world.g))
skel = world.add_skeleton(skel_path)
# print("Skeleton add OK")

itr = 1
sample_num = 1 #sample number
dt = 0.05
cluster_size = 3

traj_sample_lists = data_logger.unpickle(_data_files_dir +
    ('traj_sample_itr_%02d.pkl' % itr))

for s in range(sample_num):
    sample = traj_sample_lists[0][s]
    xp_pt = sample.get(END_EFFECTOR_POINTS)
    xv_pt = sample.get(END_EFFECTOR_POINT_VELOCITIES)
    jp_pt = sample.get(JOINT_ANGLES)
    jv_pt = sample.get(JOINT_VELOCITIES)
    jt_pt = sample.get(ACTION)

print 'jp_pt:',jp_pt.shape
print 'jv_pt:',jv_pt.shape
assert jp_pt.shape[0]==jv_pt.shape[0]
time_steps = jp_pt.shape[0]-1
print 'time_steps:',time_steps

yumi_stat_fric = np.zeros(7)
yumi_dyn_fric = np.zeros(7)
yumi_stat_fric[0] = 2.43
yumi_stat_fric[1] = 2.76
yumi_stat_fric[2] = 1.11
yumi_stat_fric[3] = 0.52
yumi_stat_fric[4] = 0.4
yumi_stat_fric[5] = 0.2
yumi_stat_fric[6] = 0.05

yumi_dyn_fric[0] = 1.06
yumi_dyn_fric[1] = 1.09
yumi_dyn_fric[2] = 0.61
yumi_dyn_fric[3] = 0.08
yumi_dyn_fric[4] = 0.08
yumi_dyn_fric[5] = 0.08
yumi_dyn_fric[6] = 0.08

gripper = skel.bodynodes[-1]

ja_pt = np.zeros([99,7])
# ja_pt = np.divide(jv_pt[1:,:]-jv_pt[0:-1,:],dt)
ja_pt[1:,:] = np.divide(jv_pt[2:,:]-jv_pt[0:-2,:],2.*dt)
ja_pt[0,:] = np.divide(jv_pt[1,:]-jv_pt[0,:],dt)
print 'ja_pt:',ja_pt.shape
coriolis_gravity_trq = np.zeros([time_steps,7])
inertia_trq = np.zeros([time_steps,7])
frict_trq = np.zeros([time_steps,7])
rbd_trq = np.zeros([time_steps,7])
ext_trq = np.zeros([time_steps,7])
action_frc = np.zeros([time_steps,6])
ext_frc = np.zeros([time_steps,6])
gripper_pos = np.zeros([time_steps,3])
for t in range(time_steps):
    skel.set_positions(jp_pt[t])
    skel.set_velocities(jv_pt[t])
    Jgripper = gripper.world_jacobian()
    frict_trq[t] = np.diag(yumi_dyn_fric).dot(jv_pt[t]) + 0.4*np.diag(yumi_stat_fric).dot(np.sign(jv_pt[t]))
    Inertia = skel.mass_matrix()
    inertia_trq[t] = Inertia.dot(ja_pt[t])
    coriolis_gravity_trq[t] = skel.coriolis_and_gravity_forces()
    rbd_trq[t] = inertia_trq[t] + coriolis_gravity_trq[t] + frict_trq[t]
    # rbd_trq[t] = frict_trq[t]
    ext_trq[t] = rbd_trq[t] - jt_pt[t]
    ext_frc[t] = np.linalg.pinv(Jgripper.T).dot(ext_trq[t])
    action_frc[t] = np.linalg.pinv(Jgripper.T).dot(jt_pt[t])
    gripper_pos[t] = gripper.to_world()

# # PLot basic signals
# plt.figure(2)
#
# #EE1
# plt.subplot(571)
# plt.title('EE1')
# plt.plot(xp_pt[:,0:3])
#
# #EE2
# plt.subplot(572)
# plt.title('EE2')
# plt.plot(xp_pt[:,3:6])
#
# #EE3
# plt.subplot(573)
# plt.title('EE3')
# plt.plot(xp_pt[:,6:9])
#
# #f_ext
# plt.subplot(574)
# plt.title('f_ext')
# plt.plot(ext_frc[:,0:3])
#
# #f_ext_norm
# plt.subplot(575)
# plt.title('f_ext_norm')
# plt.plot(np.linalg.norm(ext_frc[:,0:3],axis=1))


# #jPos
# for j in range(7):
#     plt.subplot(5,7,8+j)
#     plt.title('j%dPos' %(j+1))
#     plt.plot(jp_pt[:,j],color='g')
#
# #jVel
# for j in range(7):
#     plt.subplot(5,7,15+j)
#     plt.title('j%dVel' %(j+1))
#     plt.plot(jv_pt[:,j],color='b')
#
# #jTrq
# for j in range(7):
#     plt.subplot(5,7,22+j)
#     plt.title('j%dTrq' %(j+1))
#     plt.plot(jt_pt[:,j],color='r')
#     plt.plot(rbd_trq[:,j],color='m')
#
# #jTrqExt
# for j in range(7):
#     plt.subplot(5,7,29+j)
#     plt.title('j%dTrqExt' %(j+1))
#     plt.plot(ext_trq[:,j])
#     # plt.plot(frict_trq[:,j])
#     # plt.plot(coriolis_gravity_trq[:,j])
#     # plt.plot(inertia_trq[:,j])
#
# figvec = plt.figure()
# ax = figvec.gca(projection='3d')
# px = gripper_pos[:,0]
# py = gripper_pos[:,1]
# pz = gripper_pos[:,2]
# fx = ext_frc[:,0]
# fy = ext_frc[:,1]
# fz = ext_frc[:,2]
# ax.quiver(px, py, pz, fx, fy, fz, length=0.01)
# ax.plot3D(px, py, pz, 'r')
# plt.show()

xo = np.concatenate((jp_pt,jv_pt),axis=1)
print 'xo:',xo.shape
# xo = jp_pt
# xo = gripper_pos
x = xo[0:-1,:] # x[t]
print 'x:',x.shape

x1 = xo[1:,:] # x[t+1]
print 'x1:',x1.shape
xd = x1-x # function output
# u = jt_pt[0:-1,:] # u[t]
u = ext_trq # u_ext[t]
# u = ext_trq[0:-1,:] # u_ext[t]
print 'u:',u.shape
xu = np.concatenate((x,u),axis=1) # function input
print 'xu:',xu.shape
# xu = np.array(range(100),dtype=float).reshape(20,5)
xu_max_g = np.amax(xu,axis=0)
xu_min_g = np.amin(xu,axis=0)
xu_n = np.zeros(xu.shape)
for i in range(xu.shape[1]):
    xu_n[:,i] = np.divide((xu[:,i] - xu_min_g[i]),(xu_max_g[i] - xu_min_g[i]))


w = np.eye(cluster_size)
xu_max = np.amax(xu_n,axis=0)
xu_min = np.amin(xu_n,axis=0)
# print xu_n
# print xu_max
# print xu_min
centers = np.random.uniform(xu_min,xu_max,(cluster_size,xu_n.shape[1]))
# print centers

terminate_falg = False;

clusters = []
def checkForElement(clusters,t):
    hit_list = []
    for c,_ in enumerate(clusters):
        for i,_ in enumerate(clusters[c]["t"]):
            if clusters[c]["t"][i]==t:
                hit_list.append([c,i])
    return hit_list

def removeElement(clusters,hit_list):
    c = hit_list[0][0]
    i = hit_list[0][1]
    del clusters[c]["t"][i]
    del clusters[c]["xu"][i]
    del clusters[c]["xd"][i]

def insertElement(clusters,c,t,xu_n,xd):
    clusters[c]["t"].append(t)
    clusters[c]["xu"].append(xu_n)
    clusters[c]["xd"].append(xd)

def updateCenters(clusters,centers):
    for c,_ in enumerate(clusters):
        if len(clusters[c]["t"])!=0:
            data = np.mean(np.array(clusters[c]["xu"]),axis=0)
            centers[c] = data
            clusters[c]["c"] = centers[c]
        else:
            centers[c] = np.random.uniform(xu_min,xu_max,(1,xu_n.shape[1]))
            clusters[c]["c"] = centers[c]
def printClusters(clusters):
    for c,_ in enumerate(clusters):
        print "cluster %d: %d" %(c,len(clusters[c]["t"]))

for i in range(cluster_size):
    clusters.append(dict(
        c=centers[i],
        xu=list(),
        xd=list(),
        t=list()
    ))

itr=0
while (terminate_falg==False) or itr<100:
# while (terminate_falg==False):
    # print "itr:",itr
    terminate_falg=True
    Kc = kernel_exp(centers, xu_n, w)
    # print "Kc",Kc.shape
    for t in range(xu_n.shape[0]):
        cluster = np.argmax(Kc[:,t])
        # print cluster
        hit_list = []
        if itr==0:
            insertElement(clusters,cluster,t,xu_n[t],xd[t])
            terminate_falg=False
        else:
            hit_list = checkForElement(clusters,t)
            assert len(hit_list)==1
            if (cluster != hit_list[0][0]):
                removeElement(clusters,hit_list)
                insertElement(clusters,cluster,t,xu_n[t],xd[t])
                terminate_falg=False

    for c,_ in enumerate(clusters):
        assert len(clusters[c]["t"])==len(clusters[c]["xu"])==len(clusters[c]["xd"])

    updateCenters(clusters,centers)
    # print "itr:",itr
    # printClusters(clusters)
    itr+=1
print "itr:",itr
printClusters(clusters)
# plt.show()



# plt.figure(2)
# #p'=f(p)
# for j in range(7):
#     p = jp_pt[::2,j]
#     p1 = jp_pt[1::2,j]
#     plt.subplot(5,7,1+j)
#     plt.title("p%d'=f(p%d)" % (j+1,j+1))
#     plt.plot(p,p1,'go')
# #v'=f(v)
# for j in range(7):
#     v = jv_pt[::2,j]
#     v1 = jv_pt[1::2,j]
#     plt.subplot(5,7,8+j)
#     plt.title("v%d'=f(v%d)" % (j+1,j+1))
#     plt.plot(v,v1,'bo')
# #p'=f(u)
# for j in range(7):
#     u = jt_pt[::2,j]
#     p1 = jp_pt[1::2,j]
#     plt.subplot(5,7,15+j)
#     plt.title("p%d'=f(u%d)" % (j+1,j+1))
#     plt.plot(u,p1,'ro')
# #v'=f(u)
# for j in range(7):
#     u = jt_pt[::2,j]
#     v1 = jv_pt[1::2,j]
#     plt.subplot(5,7,22+j)
#     plt.title("v%d'=f(u%d)" % (j+1,j+1))
#     plt.plot(u,v1,'mo')
#
# fig = plt.figure(3)
# tm = range(100)
# t = np.asarray(tm[::2])
#
# #p'=f(p,t)
# for j in range(7):
#     p = jp_pt[::2,j]
#     p1 = jp_pt[1::2,j]
#     ax = fig.add_subplot(4,7,1+j, projection='3d')
#     plt.title("p%d'=f(p%d,t)" % (j+1,j+1))
#     ax.plot3D(p, t, p1, 'g')
#
# #v'=f(v,t)
# for j in range(7):
#     v = jv_pt[::2,j]
#     v1 = jv_pt[1::2,j]
#     ax = fig.add_subplot(4,7,8+j, projection='3d')
#     plt.title("v%d'=f(v%d,t)" % (j+1,j+1))
#     ax.plot3D(v, t, v1, 'b')
#
# #p'=f(u,t)
# for j in range(7):
#     u = jt_pt[::2,j]
#     p1 = jp_pt[1::2,j]
#     ax = fig.add_subplot(4,7,15+j, projection='3d')
#     plt.title("p%d'=f(u%d,t)" % (j+1,j+1))
#     ax.plot3D(u, t, p1, 'r')
#
# #v'=f(u,t)
# for j in range(7):
#     u = jt_pt[::2,j]
#     v1 = jv_pt[1::2,j]
#     ax = fig.add_subplot(4,7,22+j, projection='3d')
#     plt.title("v%d'=f(u%d,t)" % (j+1,j+1))
#     ax.plot3D(u, t, v1, 'm')
#
# fig=plt.figure(4)
# #p'=f(p,v)
# for j in range(7):
#     p = jp_pt[::2,j]
#     v = jv_pt[::2,j]
#     p1 = jp_pt[1::2,j]
#     ax = fig.add_subplot(4,7,1+j, projection='3d')
#     plt.title("p%d'=f(p%d,v%d)" % (j+1,j+1,j+1))
#     ax.plot3D(p, v, p1, 'g')
# #v'=f(p,v)
# for j in range(7):
#     p = jp_pt[::2,j]
#     v = jv_pt[::2,j]
#     v1 = jv_pt[1::2,j]
#     ax = fig.add_subplot(4,7,8+j, projection='3d')
#     plt.title("v%d'=f(p%d,v%d)" % (j+1,j+1,j+1))
#     ax.plot3D(p, v, v1, 'b')
#
# plt.show()
