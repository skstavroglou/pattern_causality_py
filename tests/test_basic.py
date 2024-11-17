import unittest
import numpy as np
import pandas as pd
from pattern_causality import pc_lightweight, load_data


class TestBasicFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load data once for all tests"""
        cls.data = load_data()
        cls.X = cls.data["NAO"].values
        cls.Y = cls.data["AAO"].values

    def test_pc_lightweight_basic(self):
        """Test basic functionality of pc_lightweight"""
        result = pc_lightweight(X=self.X, Y=self.Y, E=3, tau=1, h=1)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue("Total Causality" in result.columns)
        self.assertTrue("Positive Causality" in result.columns)
        self.assertTrue("Negative Causality" in result.columns)
        self.assertTrue("Dark Causality" in result.columns)

    def test_input_validation(self):
        """Test input validation"""
        # Test invalid E
        with self.assertRaises(ValueError):
            pc_lightweight(X=self.X, Y=self.Y, E=0, tau=1, h=1)

        # Test invalid tau
        with self.assertRaises(ValueError):
            pc_lightweight(X=self.X, Y=self.Y, E=3, tau=0, h=1)

        # Test invalid data type
        with self.assertRaises(TypeError):
            pc_lightweight(X="invalid", Y=self.Y, E=3, tau=1, h=1)

    def test_weighted_vs_unweighted(self):
        """Test that weighted and unweighted calculations give different results"""
        weighted_result = pc_lightweight(
            X=self.X, Y=self.Y, E=3, tau=1, h=1, weighted=True
        )

        unweighted_result = pc_lightweight(
            X=self.X, Y=self.Y, E=3, tau=1, h=1, weighted=False
        )

        # Results should be different
        self.assertNotEqual(
            weighted_result["Positive Causality"].values[0],
            unweighted_result["Positive Causality"].values[0],
        )


if __name__ == "__main__":
    unittest.main()
