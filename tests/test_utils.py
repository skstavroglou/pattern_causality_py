import unittest
import numpy as np
from utils.statespace import statespace
from utils.patternhashing import patternhashing
from utils.signaturespace import signaturespace
from utils.distancematrix import distancematrix
from utils.patternspace import patternspace
from utils.pastNNs import pastNNs
from utils.projectedNNs import projectedNNs
from utils.predictionY import predictionY
from utils.fillPCMatrix import fillPCMatrix
from utils.natureOfCausality import natureOfCausality
from utils.databank import databank
from utils.fcp import fcp


class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.time_series = np.sin(np.linspace(0, 10, 100))
        self.E = 3
        self.tau = 1

    def test_statespace(self):
        """Test state space creation"""
        result = statespace(self.time_series.tolist(), self.E, self.tau)
        self.assertIsInstance(result, np.ndarray)
        expected_shape = (len(self.time_series) - (self.E - 1) * self.tau, self.E)
        self.assertEqual(result.shape, expected_shape)

    def test_patternhashing(self):
        """Test pattern hashing"""
        result = patternhashing(self.E)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), self.E ** 2)

    def test_distance_matrix(self):
        """Test distance matrix calculation"""
        state_space = statespace(self.time_series.tolist(), self.E, self.tau)
        result = distancematrix(state_space, metric="euclidean")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (len(state_space), len(state_space)))

    def test_fcp(self):
        """Test first causality point calculation"""
        result = fcp(self.E, self.tau, 1, self.time_series.tolist())
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_databank(self):
        """Test databank functionality"""
        # Test vector creation
        vector = databank("vector", [5])
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (5,))

        # Test matrix creation
        matrix = databank("matrix", [3, 3])
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.shape, (3, 3))

        # Test array creation
        array = databank("array", [2, 2, 2])
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.shape, (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
