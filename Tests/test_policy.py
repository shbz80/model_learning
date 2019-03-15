from unittest import TestCase
from mjc_exp_policy import Policy
import numpy as np
from model_leraning_utils import YumiKinematics

agent_hyperparams = {
    'dt': 0.05,
    'T': 100,
    'smooth_noise': False,
    'smooth_noise_var': 1.,
    'smooth_noise_renormalize': False
}

exp_params = {
            'dt': agent_hyperparams['dt'],
            'T': agent_hyperparams['T'],
            'num_samples': 10,
            'dP': 7,
            'dV': 7,
            'dU': 7,
            'x0': np.concatenate([np.array([-1.3033, -1.3531, 0.9471, 0.3177, 2.0745, 1.4900, -2.1547]),
                          np.zeros(7)]),
            'target_x': np.array([ 0.39067804, 0.14011851, -0.06375249, 0.31984032, 1.55309358, 1.93199837]),
            'target_x_delta': np.array([-0.1, -0.1, -0.1, 0.0, 0.0, 0.0]),
            'Kp': np.array([0.22, 0.22, 0.18, 0.15, 0.05, 0.05, 0.025])*100.0*0.5,
            'Kd': np.array([0.07, 0.07, 0.06, 0.05, 0.015, 0.015, 0.01])*10.0*0.5,
            'Kpx': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*0.7,
            'noise_gain': 0.01,
            't_contact_factor': 3,
}

class TestPolicy(TestCase):
    def setUp(self):
        self.policy = Policy(agent_hyperparams, exp_params)

    def test_init(self):
        '''
        Test for only known initial Cartesian position (translation)
        '''
        init_x = self.policy.init_x[:3]     # only translations
        expected_init_x = np.array([0.490, 0.240, 0.036]) # known translations
        np.testing.assert_array_almost_equal(init_x, expected_init_x, decimal=3)

    def test_act_mode(self):
        '''
        Function act() cannot be tested fully without running from the beginning and
        interacting with other code. In this test only the three control modes are tested
        '''

        # set condition for mode z
        self.policy.ref_x[2] = 0.02
        u = self.policy.act(np.zeros(14), None, 0) # pass zero vector for state since it doesn't matter
        self.assertEqual(self.policy.ref_x_dot_d[0], 0.)    # ref vel for x should be 0.
        self.assertEqual(self.policy.ref_x_dot_d[1], 0.)    # ref vel for y should be 0.
        self.assertLess(self.policy.ref_x_dot_d[2], 0.)  # ref vel for z should be < 0.

        # set condition for mode y
        self.policy.ref_x[2] = -0.06375249
        self.policy.ref_x[1] = 0.2
        u = self.policy.act(np.zeros(14), None, 0)  # pass zero vector for state since it doesn't matter
        self.assertEqual(self.policy.ref_x_dot_d[0], 0.)  # ref vel for x should be 0.
        self.assertEqual(self.policy.ref_x_dot_d[2], 0.)  # ref vel for z should be 0.
        self.assertLess(self.policy.ref_x_dot_d[1], 0.)  # ref vel for y should be < 0.

        # set condition for mode x
        self.policy.ref_x[2] = -0.06375249
        self.policy.ref_x[1] = 0.14011851
        self.policy.ref_x[0] = 0.44
        u = self.policy.act(np.zeros(14), None, 0)  # pass zero vector for state since it doesn't matter
        self.assertEqual(self.policy.ref_x_dot_d[1], 0.)  # ref vel for y should be 0.
        self.assertEqual(self.policy.ref_x_dot_d[2], 0.)  # ref vel for z should be 0.
        self.assertLess(self.policy.ref_x_dot_d[0], 0.)  # ref vel for x should be < 0.

    def test_predict(self):
        '''
        The main algorithm is the same as act() (above) but takes in an array of state inputs.
        So test only for the correct shape of the return value
        '''

        x = np.concatenate((self.policy.init_q, self.policy.init_q_dot), axis=0)
        X = np.tile(x, (10,1))
        U, _ = self.policy.predict(X, 0)
        self.assertEqual(U.shape, (10,7))

