import unittest
import numpy as np
import pandas as pd
from pattern_causality import (
    pc_matrix,
    pc_cross_validation,
    optimal_parameters_search,
    load_data,
)


class TestAdvancedFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load data once for all tests"""
        cls.data = load_data()
        cls.X = cls.data["NAO"].values
        cls.Y = cls.data["AAO"].values

    def test_pc_matrix(self):
        """Test pc_matrix functionality"""
        results = pc_matrix(
            dataset=self.data.drop(columns=["Date"]),
            E=3,
            tau=1,
            metric="euclidean",
            h=1,
            weighted=True,
        )
        self.assertIsInstance(results, dict)
        self.assertIn("positive", results)
        self.assertIn("negative", results)
        self.assertIn("dark", results)
        self.assertIn("items", results)

    def test_cross_validation(self):
        """Test cross-validation functionality"""
        cv_results = pc_cross_validation(
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

    def test_parameter_optimization(self):
        """Test parameter optimization"""
        result = optimal_parameters_search(
            Emax=3,
            tau_max=2,
            metric="euclidean",
            dataset=self.data.drop(columns=["Date"]),
        )
        self.assertIsInstance(result, pd.DataFrame)
        # 检查DataFrame是否包含必要的列
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
        # 检查数据类型和范围
        self.assertTrue(all(result["E"] >= 2))
        self.assertTrue(all(result["tau"] >= 1))


if __name__ == "__main__":
    unittest.main()
