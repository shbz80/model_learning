from unittest import TestCase
import numpy as np
from model_leraning_utils import UGP

class TestUGP(TestCase):
    def setUp(self):
        self.ugp = UGP(L=2)

    def test_get_sigma_points(self):
        '''
        test the number of sigma point generated along with shapes of return values
        Also, verify the method by recomputing the input mean from weighted averaging of sigma points
        '''
        mu = np.zeros(2, dtype=float)
        var = np.eye(2, dtype=float)
        sigmaMat, W_mu, W_var = self.ugp.get_sigma_points(mu, var)
        self.assertEqual(sigmaMat.shape, (5,2)) # expecting 5 sigma points for L=2
        self.assertEqual(W_mu.shape, (5,))
        self.assertEqual(W_var.shape, (5,))
        np.testing.assert_array_almost_equal(W_mu[1:], W_var[1:]) # expecting equal weights for all but the first ones
        mu_1 = np.average(sigmaMat, axis=0, weights=W_mu)
        np.testing.assert_array_almost_equal(mu, mu_1)  # recomputing mu from sigma points

    def test_get_posterior(self):
        '''
        Test the shape of outputs and positive definiteness of the output variance.
        Test the UT transform algo with a simple known case
        '''
        # define test nonlinear func
        class NonLinFn(object):
            def predict(self, X, return_std=True):
                return np.square(X), np.zeros(X.shape)
        fn = NonLinFn()
        mu = np.zeros(2, dtype=float)
        var = np.eye(2, dtype=float)
        Y_mu_post, Y_var_post, _, _, _ = self.ugp.get_posterior(fn, mu, var)
        # test shapes
        self.assertEqual(Y_mu_post.shape, (2,))
        self.assertEqual(Y_var_post.shape, (2,2))

        # output variance is pd
        self.assertTrue(np.all(np.linalg.eigvals(Y_var_post) > 0))

        # Test the UT transform algo with a simple known case
        expected_Y_mu = np.array([1.000001, 1.000001])
        expected_Y_var = np.array([[2.000005, 2.000003],
                                    [2.000003, 2.000005]])
        np.testing.assert_array_almost_equal(expected_Y_mu, Y_mu_post)
        np.testing.assert_array_almost_equal(expected_Y_var, Y_var_post)



