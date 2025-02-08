import unittest
import numpy as np
import pandas as pd
from pattern_causality import pattern_causality, load_data


class TestBasicFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load data once for all tests"""
        cls.data = load_data()
        cls.X = cls.data["NAO"].values
        cls.Y = cls.data["AAO"].values
        cls.pc = pattern_causality(verbose=False)

    def test_pc_lightweight_basic(self):
        """Test basic functionality of pc_lightweight"""
        result = self.pc.pc_lightweight(X=self.X, Y=self.Y, E=3, tau=1, h=1)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue("Total Causality" in result.columns)
        self.assertTrue("Positive Causality" in result.columns)
        self.assertTrue("Negative Causality" in result.columns)
        self.assertTrue("Dark Causality" in result.columns)

    def test_input_validation(self):
        """Test input validation"""
        # Test with non-numeric data
        with self.assertRaises(TypeError):
            self.pc.pc_lightweight(X=["invalid"], Y=self.Y, E=3, tau=1, h=1)

        # Test with invalid dimensions
        with self.assertRaises(ValueError):
            self.pc.pc_lightweight(X=[], Y=self.Y, E=3, tau=1, h=1)

    def test_weighted_vs_unweighted(self):
        """Test that weighted and unweighted calculations give different results"""
        weighted_result = self.pc.pc_lightweight(
            X=self.X, Y=self.Y, E=3, tau=1, h=1, weighted=True
        )

        unweighted_result = self.pc.pc_lightweight(
            X=self.X, Y=self.Y, E=3, tau=1, h=1, weighted=False
        )

        # Results should be different
        self.assertNotEqual(
            weighted_result["Positive Causality"].values[0],
            unweighted_result["Positive Causality"].values[0],
        )


if __name__ == "__main__":
    unittest.main()
