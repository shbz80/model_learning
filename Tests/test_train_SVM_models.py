from unittest import TestCase
import numpy as np
import pickle
from model_leraning_utils import train_SVM_models
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

logfile = "../Results/yumi_peg_exp_new_preprocessed_data_train.p"
svm_grid_params = {
                    'param_grid': {"C": np.logspace(-10, 10, endpoint=True, num=11, base=2.),
                                   "gamma": np.logspace(-10, 10, endpoint=True, num=11, base=2.)},
                    'scoring': 'accuracy',
                    # 'cv': 5,
                    'n_jobs':-1,
                    'iid': False,
                    'cv':3,
                    }
svm_params = {

    'kernel': 'rbf',
    'decision_function_shape': 'ovr',
    'tol': 1e-06,
}

class TestTrain_SVM_models(TestCase):
    def setUp(self):
        exp_data = pickle.load(open(logfile, "rb"))
        exp_params = exp_data['exp_params']
        self.dP = exp_params['dP']
        self.dV = exp_params['dV']
        self.dU = exp_params['dU']
        self.XUs_t_train = exp_data['XUs_t_train']
        self.dpgmm_train_labels = exp_data['clust_result']['assign']
        self.dpgmm_labels = exp_data['clust_result']['labels']
        self.train_SVM_models = train_SVM_models

    def test_train_SVM_models(self):
        '''
        SVM mode prediction models are trained with the same data in the actual code.
        We test for the correct number of SVMs.
        '''
        SVMs = self.train_SVM_models(svm_grid_params, svm_params, self.XUs_t_train,
                              self.dpgmm_train_labels, self.dpgmm_labels)
        # there should be as many SVMs as there are experts or clusters
        self.assertTrue(len(SVMs)==len(self.dpgmm_labels))

