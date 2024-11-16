import unittest
import numpy as np
from utils import (
    statespace,
    patternhashing,
    signaturespace,
    distancematrix,
    patternspace,
    pastNNs,
    projectedNNs,
    predictionY,
    fillPCMatrix,
    natureOfCausality,
    databank,
    fcp
)

class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.time_series = np.sin(np.linspace(0, 10, 100))
        self.E = 3
        self.tau = 1

    def test_statespace(self):
        """Test state space creation"""
        from utils.statespace import statespace
        result = statespace(self.time_series.tolist(), self.E, self.tau)
        self.assertIsInstance(result, np.ndarray)
        expected_shape = (len(self.time_series) - (self.E - 1) * self.tau, self.E)
        self.assertEqual(result.shape, expected_shape)

    def test_patternhashing(self):
        """Test pattern hashing"""
        from utils.patternhashing import patternhashing
        result = patternhashing(self.E)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3**(self.E - 1))

    def test_distance_matrix(self):
        """Test distance matrix calculation"""
        from utils.distancematrix import distancematrix
        from utils.statespace import statespace
        state_space = statespace(self.time_series.tolist(), self.E, self.tau)
        result = distancematrix(state_space, metric="euclidean")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (len(state_space), len(state_space)))

    def test_fcp(self):
        """Test first causality point calculation"""
        from utils.fcp import fcp
        result = fcp(self.E, self.tau, 1, self.time_series.tolist())
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

if __name__ == '__main__':
    unittest.main()
