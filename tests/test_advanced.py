import unittest
import numpy as np
import pandas as pd
from pattern_causality import pattern_causality, load_data


class TestAdvancedFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load data once for all tests"""
        cls.data = load_data()
        cls.X = cls.data["NAO"].values
        cls.Y = cls.data["AAO"].values
        cls.pc = pattern_causality(verbose=False)

    def test_pc_matrix(self):
        """Test pc_matrix functionality"""
        results = self.pc.pc_matrix(
            dataset=self.data.drop(columns=["Date"]),
            E=3,
            tau=1,
            metric="euclidean",
            h=1,
            weighted=True,
        )
        self.assertIsInstance(results, pd.DataFrame)
        self.assertTrue("from_var" in results.columns)
        self.assertTrue("to_var" in results.columns)
        self.assertTrue("positive" in results.columns)
        self.assertTrue("negative" in results.columns)
        self.assertTrue("dark" in results.columns)

    def test_cross_validation(self):
        """Test cross-validation functionality"""
        cv_results = self.pc.pc_cross_validation(
            X=self.X,
            Y=self.Y,
            E=3,
            tau=1,
            metric="euclidean",
            h=1,
            weighted=True,
            numberset=[100, 200, 300],
        )
        self.assertIsInstance(cv_results, pd.DataFrame)
        self.assertEqual(len(cv_results), 3)
        self.assertTrue("positive" in cv_results.columns)
        self.assertTrue("negative" in cv_results.columns)
        self.assertTrue("dark" in cv_results.columns)

    def test_parameter_optimization(self):
        """Test parameter optimization"""
        result = self.pc.optimal_parameters_search(
            Emax=3,
            tau_max=2,
            metric="euclidean",
            h=1,
            weighted=False,
            dataset=self.data.drop(columns=["Date"]),
        )
        self.assertIsInstance(result, pd.DataFrame)
        # Check if DataFrame contains necessary columns
        expected_columns = [
            "E",
            "tau",
            "Total",
            "of which Positive",
            "of which Negative",
            "of which Dark",
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)
        # Check data types and ranges
        self.assertTrue(all(result["E"] >= 2))
        self.assertTrue(all(result["tau"] >= 1))


if __name__ == "__main__":
    unittest.main()
