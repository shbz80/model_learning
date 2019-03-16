from unittest import TestCase
import numpy as np
import pickle
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from model_leraning_utils import train_trans_models

logfile = "../Results/yumi_peg_exp_new_preprocessed_data_train.p"
# GP parameters
gpr_params = {
    'alpha': 0.,  # alpha=0 when using white kernal
        'kernel': C(1.0, (1e-2, 1e2)) * RBF(np.ones(21), (1e-1, 1e1)) + W(noise_level=1.,
                                                                     noise_level_bounds=(1e-4, 1e1)),
    'n_restarts_optimizer': 10,
    'normalize_y': False,
}

class TestTrain_trans_models(TestCase):
    def setUp(self):
        exp_data = pickle.load(open(logfile, "rb"))
        exp_params = exp_data['exp_params']
        self.dP = exp_params['dP']
        self.dV = exp_params['dV']
        self.dU = exp_params['dU']
        self.XUs_t_train = exp_data['XUs_t_train']
        self.dpgmm_train_labels = exp_data['clust_result']['assign']
        self.dpgmm_labels = exp_data['clust_result']['labels']
        self.dX = self.dP + self.dV

        self.gpr_params_list = []
        for i in range(self.dX):
            self.gpr_params_list.append(gpr_params)
        self.train_trans_models = train_trans_models

    def test_train_trans_models_count(self):
        '''
        The transition models are trained as usual. We check the number of transition models
        with respect to the number of experts or clusters
        '''
        trans_dicts = self.train_trans_models(self.gpr_params_list, self.XUs_t_train, self.dpgmm_train_labels, self.dX, self.dU)
        # for n experts we can have up to (n-1)*2 two way transitions between adjacent experts.
        self.assertTrue(len(trans_dicts) < (len(self.dpgmm_labels) - 1)*2)


