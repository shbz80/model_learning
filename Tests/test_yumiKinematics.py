from unittest import TestCase
from model_leraning_utils import YumiKinematics
import numpy as np


class TestYumiKinematics(TestCase):
    def setUp(self):
        self.yumiKinematics = YumiKinematics()

    def test_forward(self):
        '''
        Test forward kinematics with known paris of input output. Checks if the difference is withing 5% error
        '''
        # sample scenario obtained from a verified case
        q = np.array([-1.3751, -1.0998, 1.0831, 0.8614, 1.9888, 1.6301, -2.8828])
        q_dot = np.array([-0.2986, -0.1859, -0.0586, -1.3148, -0.3322, -0.1956, 0.9566])
        x = np.array([0.4019, 0.0618, 0.1084, -0.8041, 1.5658, 0.7933])
        x_dot = np.array([2.2389e-01, 7.4398e-02, -1.1647e-01, -9.140268e+01, 1.242933e-01, 9.124167e+01])

        X = np.concatenate((q, q_dot), axis=0)
        EX = self.yumiKinematics.forward(X)
        EX = EX.reshape(-1)
        expected_EX = np.concatenate((x, x_dot), axis=0)
        diff = EX-expected_EX
        diff_percentage = np.divide(diff, expected_EX)*100.
        self.assertTrue(np.all(diff_percentage < 5.))

    def test_jacobian_analytic_to_geometric(self):
        '''
        This function is not yet used.
        '''

    def test_jacobian_geometric_to_analytic(self):
        '''
        The geometric to analytic Jacobian is tested on its dot product with a joint pos
        '''
        # sample scenario obtained from a verified case
        q = np.array([-1.3751, -1.0998,  1.0831,  0.8614,  1.9888,  1.6301, -2.8828])
        q_dot = np.array([-0.2986, -0.1859, -0.0586, -1.3148, -0.3322, -0.1956, 0.9566])
        x = np.array([0.4019,  0.0618,  0.1084, -0.8041,  1.5658,  0.7933])
        x_dot = np.array([2.2389e-01,  7.4398e-02, -1.1647e-01, -9.140268e+01, 1.242933e-01,  9.124167e+01])

        J_G = self.yumiKinematics.kdl_kin.jacobian(q)
        J_A = self.yumiKinematics.jacobian_geometric_to_analytic(J_G, x[3:])
        x_dot_result = np.array(J_A.dot(q_dot))
        x_dot_result = x_dot_result.reshape(-1)
        # test for equality of x_dot with the sample case
        np.testing.assert_array_almost_equal(x_dot, x_dot_result, decimal=4)

    def test_closed_loop_IK(self):
        '''
        This function is not yet used.
        '''

