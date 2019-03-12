import unittest
import numpy as np
from a3c import calculate_lambda_returns, calculate_renormalized_lambda_returns


class TestCaseReturns(unittest.TestCase):
    def setUp(self):
        self.rewards = np.array([-20.0,  30.0, -30.0,  40.0, -40.0])
        self.values  = np.array([400.0, 300.0, 200.0, 100.0, 200.0, 1000.0])

    def assertNumpyEqual(self, x, y):
        x, y = map(np.array, [x, y])
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.dtype, y.dtype)
        self.assertTrue(np.allclose(x - y, 0.0))

    def test_montecarlo_done(self):
        returns = calculate_lambda_returns(self.rewards, self.values, done=True, discount=0.9, lambd=1.0)
        self.assertNumpyEqual(returns, [-14.384, 6.24, -26.4, 4.0, -40.0])

    def test_montecarlo_notdone(self):
        returns = calculate_lambda_returns(self.rewards, self.values, done=False, discount=0.9, lambd=1.0)
        self.assertNumpyEqual(returns, [576.106, 662.34, 702.6, 814, 860])

    def test_lambda_done(self):
        returns = calculate_lambda_returns(self.rewards, self.values, done=True, discount=0.9, lambd=0.8)
        self.assertNumpyEqual(returns, [92.9165056, 81.82848, 21.984, 47.2, -40.0])

    def test_lambda_notdone(self):
        returns = calculate_lambda_returns(self.rewards, self.values, done=False, discount=0.9, lambd=0.5)
        self.assertNumpyEqual(returns, [219.149125, 231.4425, 247.65, 517.0, 860.0])

    def test_renormalized_lambda_done(self):
        returns = calculate_renormalized_lambda_returns(self.rewards, self.values, done=True, discount=0.9, lambd=0.5)
        self.assertNumpyEqual(returns, [188.58632258, 158.976, 78.51428571, 148.0, -40.0])

    def test_renormalized_lambda_notdone(self):
        returns = calculate_renormalized_lambda_returns(self.rewards, self.values, done=False, discount=0.9, lambd=0.5)
        self.assertNumpyEqual(returns, [207.6343871, 202.716, 182.65714286, 418.0, 860.0])
