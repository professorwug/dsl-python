import unittest

import pandas as pd
from patsy import dmatrices

from dsl.dsl import dsl
from tests.PanChen_test.compare_panchen import load_panchen_data, prepare_data_for_dsl


class TestDSLInstallation(unittest.TestCase):
    def setUp(self):
        """Load test data before each test"""
        self.data = load_panchen_data()
        self.df = prepare_data_for_dsl(self.data)

    def test_imports(self):
        """Test that all required modules can be imported"""
        self.assertTrue(callable(dsl), "dsl should be a callable function")

    def test_data_loading(self):
        """Test that test data can be loaded"""
        self.assertIsInstance(
            self.data, pd.DataFrame, "Data should be a pandas DataFrame"
        )
        self.assertGreater(len(self.data), 0, "Data should not be empty")

    def test_dsl_functionality(self):
        """Test basic DSL functionality"""
        # Define formula
        formula = (
            "SendOrNot ~ countyWrong + prefecWrong + connect2b + "
            "prevalence + regionj + groupIssue"
        )

        # Prepare design matrix (X) and response (y)
        y, X = dmatrices(formula, self.df, return_type="dataframe")

        # Run DSL estimation
        result = dsl(
            X=X.values,
            y=y.values.flatten(),
            labeled_ind=self.df["labeled"].values,
            sample_prob=self.df["sample_prob"].values,
            model="logit",
            method="logistic",
        )

        # Check that the result has expected attributes
        self.assertTrue(
            hasattr(result, "success"), "Result should have success attribute"
        )
        self.assertTrue(hasattr(result, "niter"), "Result should have niter attribute")
        self.assertTrue(
            hasattr(result, "objective"), "Result should have objective attribute"
        )

        # Check that the estimation converged
        self.assertTrue(result.success, "DSL estimation should converge")

        # Check that we got reasonable number of iterations
        self.assertGreater(result.niter, 0, "Should have positive number of iterations")
        self.assertLess(
            result.niter, 1000, "Should converge in reasonable number of iterations"
        )


if __name__ == "__main__":
    unittest.main()
