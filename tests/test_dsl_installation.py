import unittest

import numpy as np
import pandas as pd
from patsy import dmatrices

from dsl_kit import dsl
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

    def test_fixed_effects(self):
        """Test DSL with fixed effects"""
        # Create a simple fixed effect
        self.df["fe_group"] = (self.df["regionj"] > 0).astype(
            int
        )  # Binary fixed effect

        # Update formula to use main effects only
        formula = "countyWrong ~ SendOrNot + prefecWrong + connect2b + prevalence"
        y, X = dmatrices(formula, self.df, return_type="dataframe")

        # Create fixed effects dummies
        fe_array = pd.get_dummies(self.df["regionj"], prefix="fe").values

        # Run DSL estimation with fixed effects
        result = dsl(
            X=X.values,
            y=y.values.flatten(),
            labeled_ind=self.df["labeled"].values,
            sample_prob=self.df["sample_prob"].values,
            model="felm",
            method="fixed_effects",
            fe_Y=fe_array,  # Use all dummy variables for fe_Y
            fe_X=fe_array,  # Use all dummy variables for fe_X
        )

        # Check dimensions of results
        n_main_params = X.shape[1]  # Number of main effect parameters
        n_fe_params = fe_array.shape[1]  # Number of fixed effect parameters
        total_params = n_main_params + n_fe_params

        # Basic checks
        self.assertEqual(len(result.coefficients), total_params)
        self.assertEqual(len(result.standard_errors), total_params)
        self.assertEqual(result.vcov.shape, (total_params, total_params))

        # Check that we have predicted values and residuals
        self.assertTrue(
            hasattr(result, "predicted_values"),
            "Result should have predicted_values attribute",
        )
        self.assertTrue(
            hasattr(result, "residuals"), "Result should have residuals attribute"
        )

        # Check that predicted values and residuals have correct dimensions
        self.assertEqual(
            len(result.predicted_values),
            len(self.df),
            "Predicted values should have same length as data",
        )
        self.assertEqual(
            len(result.residuals),
            len(self.df),
            "Residuals should have same length as data",
        )

        # Check that residuals are reasonable
        self.assertTrue(
            np.all(np.isfinite(result.residuals)), "Residuals should be finite"
        )
        self.assertTrue(
            np.all(np.isfinite(result.predicted_values)),
            "Predicted values should be finite",
        )


if __name__ == "__main__":
    unittest.main()
