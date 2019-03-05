from unittest import TestCase
from blocks_sim import MassSlideWorld
import numpy as np

class TestMassSlideWorld(TestCase):
    def setUp(self):
        self.massSlideWorld = MassSlideWorld()

    def test_dynamics(self):
        self.fail()

    def test_reset(self):
        self.massSlideWorld.reset()  # do reset
        self.assertIs( self.massSlideWorld.mode, 'm1')
        self.assertEqual(self.massSlideWorld.t, 0)
        self.assertTrue(np.array_equal(self.massSlideWorld.X, np.zeros(2)))

    def test_step_mode(self):
        self.fail()

    def test_step(self):
        self.fail()

    def test_set_policy(self):
        self.fail()

    def test_act(self):
        self.fail()

    def test_predict(self):
        self.fail()
