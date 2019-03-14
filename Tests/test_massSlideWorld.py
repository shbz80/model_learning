from unittest import TestCase
from blocks_sim import MassSlideWorld
import numpy as np

class TestMassSlideWorld(TestCase):
    def setUp(self):
        self.massSlideWorld = MassSlideWorld()

    def test_dynamics(self):
        '''
        A typical input and output pair is checked
        '''
        X = np.array([1., 2.])
        m = 1.
        b = 0.1
        k = 0.
        u = 0.5
        expected = np.array((2., 0.3))
        result = self.massSlideWorld.dynamics(X, m, b, k, u)
        self.assertEqual(expected.shape, result.shape)  # test output shape
        np.testing.assert_array_almost_equal(expected, result)


    def test_reset(self):
        '''
        check post conditions after reset method call
        '''
        self.massSlideWorld.reset()  # do reset
        self.assertIs(self.massSlideWorld.mode, 'm1') # mode should be 'm1' after reset
        self.assertEqual(self.massSlideWorld.t, 0) # time should reset to 0
        self.assertTrue(np.array_equal(self.massSlideWorld.X, np.zeros(2))) # state should reset to [0, 0]

    def test_step_mode_m1(self):
        '''
        test transition to m1 mode
        '''
        X = np.array([3., 0.])  # a pos in m1
        u = 0.
        result_mode, result_X = self.massSlideWorld.step_mode(X, u)
        self.assertEqual('m1', result_mode) # mode should be 'm1' for X
        np.testing.assert_array_equal(X, result_X) # X should be unchanged

    def test_step_mode_m2(self):
        '''
        test transition to m2 mode
        '''
        X = np.array([8., 0.])  # a pos in m2
        u = 0.
        result_mode, result_X = self.massSlideWorld.step_mode(X, u)
        self.assertEqual('m2', result_mode) # mode should be 'm2' for X
        np.testing.assert_array_equal(X, result_X) # X should be unchanged

    def test_step_mode_m3(self):
        '''
        test transition to mode
        '''
        X = np.array([12., 0.])  # a pos in m3
        u = 0.
        self.massSlideWorld.mode = 'm1' # set some mode other than 'm3'
        result_mode, result_X = self.massSlideWorld.step_mode(X, u)
        self.assertEqual('m3', result_mode) # mode should be 'm3' for X and mode != 'm3'
        self.assertAlmostEqual(0.0, result_X[1])    # velocity should be 0 in 'm3'
        u = 4. # force less than static friction
        result_mode, result_X = self.massSlideWorld.step_mode(X, u)
        self.assertEqual('m3', result_mode)  # mode should be 'm3' for X and u < static friction
        self.assertAlmostEqual(0.0, result_X[1])  # velocity should be 0 in 'm3'


    def test_step_mode_m4(self):
        '''
        test transition to m4 mode
        '''
        X = np.array([12., 0.])
        u = 6.  # force grater than static friction
        result_mode, result_X = self.massSlideWorld.step_mode(X, u)
        self.assertNotEquals('m4', result_mode)  # mode should not be 'm4' for mode != 'm3'
        self.massSlideWorld.mode = 'm3'  # set mode 'm3'
        result_mode, result_X = self.massSlideWorld.step_mode(X, u)
        self.assertEquals('m4', result_mode)  # mode should be 'm4' for given X and u
        self.assertAlmostEqual(1.0, result_X[1])  # velocity should be 1.0 on 'm4'

    def test_step_nom(self):
        '''
        the step method has different behaviours depending on the internal mode, but
        this is not tested here. Only the common code for state evolution is tested against an
        independently derived result.
        '''
        self.massSlideWorld.mode = 'm2'
        X = np.array([5., 1.])      # test inputs for which results are known
        expected_X = np.array([5.00947636, 0.89535659])
        u = -10.
        _, result_X, _ = self.massSlideWorld.step(X, u)
        np.testing.assert_array_almost_equal(expected_X, result_X)

    def test_step_equ(self):
        '''
        the same as above but for equilibrium condition
        '''
        self.massSlideWorld.mode = 'm2'
        X = np.array([5., 0.])      # test inputs for equilibrium condition
        u = 0.
        _, result_X, _ = self.massSlideWorld.step(X, u)
        np.testing.assert_array_almost_equal(X, result_X)   # no change in X expected in equ condition

    def test_act(self):
        '''
        test three special cases of act method
        '''
        # policy params for the test input
        policy_params = {
                            'm1': {
                                'L': np.array([.2, 1.]),
                                'noise_pol': 3.,
                                'target': 18.,
                            },
                        }
        self.massSlideWorld.set_policy(policy_params)
        X = np.array([18., 0.]) # X for final goal
        _, result_u, _ = self.massSlideWorld.act(X, 'm1')
        self.assertEqual(result_u, 0.) # u=0 if already reached final goal

        X = np.array([0., 0.])  # initial condition
        _, result_u, _ = self.massSlideWorld.act(X, 'm1')
        self.assertEqual(result_u, 3.6)  # u=3.6 for initial condition

        X = np.array([20., 5.])  # overshoot condition
        _, result_u, _ = self.massSlideWorld.act(X, 'm1')
        self.assertEqual(result_u, -5.4)  # u=-5.4 for the given overshoot condition

    def test_predict(self):
        '''
        only testing for shapes since predict() is a vectorized version of act()
        '''
        policy_params = {
            'm1': {
                'L': np.array([.2, 1.]),
                'noise_pol': 3.,
                'target': 18.,
            },
        }
        self.massSlideWorld.set_policy(policy_params)
        X = np.random.multivariate_normal(np.array([0., 0.]), np.eye(2, dtype=float), 10)
        U, _ = self.massSlideWorld.predict(X)
        self.assertEqual(X.shape, (10, 2))
        self.assertEqual(U.shape, (10, 1))
