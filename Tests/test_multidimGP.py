from unittest import TestCase
from multidim_gp import MultidimGP
import numpy as np
import pickle
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W

logfile = "../Results/blocks_exp_preprocessed_data_rs_1.p"

class TestMultidimGP(TestCase):
    '''
    A 2 dimensional case is loaded and used to train a multidimensional GP.
    The two ouput dimensions are duplicated to testing purpose. We test the
    presence of two GP objects and that their kernels are identical. The mean
    value f the input data is used to do prediction on both GPs and tested to
    see if their values are the same.
    '''
    def setUp(self):
        # load training data that has 2 output dimensions
        exp_data = pickle.load( open(logfile, "rb" ) )
        XUs_t_train = exp_data['XUs_t_train']
        self.XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1])
        Xs_t1_train = exp_data['Xs_t1_train']
        self.X_t1_train = Xs_t1_train.reshape(-1, Xs_t1_train.shape[-1])
        # make both output dimensions to have the same data for testing purpose
        self.X_t1_train[:, 1] = self.X_t1_train[:, 0]

        # GP parameters
        gpr_params = {
            'alpha': 0.,  # alpha=0 when using white kernal
            'kernel': C(1.0, (1e-2, 1e2)) * RBF(np.ones(3), (1e-1, 1e1)) + W(noise_level=1.,
                                                                             noise_level_bounds=(1e-4, 1e1)),
            'n_restarts_optimizer': 10,
            'normalize_y': False,
        }
        # prepare gp_params list for both output dimensions
        gpr_params_list = []
        gpr_params_list.append(gpr_params)
        gpr_params_list.append(gpr_params)
        self.mdgp = MultidimGP(gpr_params_list, 2)
        # fit the multidimensional GP
        self.mdgp.fit(self.XU_t_train, self.X_t1_train)

    def test_fit(self):

        # assert there exists the same number of GPs as there are output dimensions
        self.assertTrue(len(self.mdgp.gp_list)==self.X_t1_train.shape[1])
        # since the data is the same in both dimensions, assert that the Cholesky factor
        # of both kernels to be same
        np.testing.assert_array_almost_equal(self.mdgp.gp_list[0].L_, self.mdgp.gp_list[1].L_)

    def test_predict(self):
        mean_x = np.mean(self.XU_t_train, axis=0)
        mean_x = mean_x.reshape(1, -1)
        # do prediction for the mean input point for both GPs
        mu_0, sig_0 = self.mdgp.gp_list[0].predict(mean_x, return_std=True)
        mu_1, sig_1 = self.mdgp.gp_list[1].predict(mean_x, return_std=True)
        # since both GPs are identical the predictions should be the same
        np.testing.assert_array_almost_equal(mu_0, mu_1)
        np.testing.assert_array_almost_equal(sig_0, sig_1)

