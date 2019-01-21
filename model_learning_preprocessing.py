import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mat_col
from utilities import plot_ellipse
from utilities import get_N_HexCol
import pickle
from mixture_model_gibbs_sampling import ACF
from collections import Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn import mixture
from scipy.special import digamma
from collections import Counter
from utilities import conditional_Gaussian_mixture
from utilities import estep
from utilities import gp_plot
from copy import deepcopy
from collapsed_Gibbs_sampler import predictive_ll_cluster
import operator
from gmm import GMM
import datetime
import copy

exp_data = pickle.load( open("./Results/blocks_exp_raw_data.p", "rb" ) )
exp_params = exp_data['exp_params']
Xs = exp_data['X']  # state
Us = exp_data['U']  # action
Xg = exp_data['Xg']  # sate ground truth
Ug = exp_data['Ug']  # action ground truth

dP = exp_params['dP']
dV = exp_params['dV']
dU = exp_params['dU']
dX = dP+dV
T = exp_params['T']

n_trials, n_time_steps, dim_state = Xs.shape
_, _, dim_action = Us.shape
assert(dX==dim_state)
assert(n_time_steps==T)
assert(dU==dim_action)

