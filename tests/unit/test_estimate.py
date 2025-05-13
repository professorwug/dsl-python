"""
Unit tests for the estimation helper functions
"""

import numpy as np
import pandas as pd
from dsl_kit.helpers.estimate import available_method, fit_model, fit_test


def test_fit_model():
    """Test the fit_model function"""
    # Create sample data
    n_samples = 100
    n_features = 5

    # Generate data
    data = pd.DataFrame(
        {
            "y": np.random.randn(n_samples),
            "labeled": np.random.binomial(1, 0.8, n_samples),
            "sample_prob": np.ones(n_samples) * 0.8,
        }
    )

    # Add features
    for i in range(n_features):
        data[f"x{i+1}"] = np.random.randn(n_samples)

    # Create outcome, labeled, and covariates
    outcome = "y"
    labeled = "labeled"
    covariates = [f"x{i+1}" for i in range(n_features)]

    # Test with different methods
    for method in ["grf", "ranger", "random_forest"]:
        # Fit model
        fit_out = fit_model(
            outcome=outcome,
            labeled=labeled,
            covariates=covariates,
            data=data,
            method=method,
            sample_prob="sample_prob",
            family="gaussian",
        )

        # Check that fit_out is not None
        assert fit_out is not None


def test_fit_test():
    """Test the fit_test function"""
    # Create sample data
    n_samples = 100
    n_features = 5

    # Generate data
    data = pd.DataFrame(
        {
            "y": np.random.randn(n_samples),
            "labeled": np.random.binomial(1, 0.8, n_samples),
            "sample_prob": np.ones(n_samples) * 0.8,
        }
    )

    # Add features
    for i in range(n_features):
        data[f"x{i+1}"] = np.random.randn(n_samples)

    # Create outcome, labeled, and covariates
    outcome = "y"
    labeled = "labeled"
    covariates = [f"x{i+1}" for i in range(n_features)]

    # Create a dummy fit_out
    fit_out = {"method": "grf", "model": "dummy_model"}

    # Test with different methods
    for method in ["grf", "ranger", "random_forest"]:
        # Predict
        Y_hat, RMSE = fit_test(
            fit_out=fit_out,
            outcome=outcome,
            labeled=labeled,
            covariates=covariates,
            data=data,
            method=method,
            family="gaussian",
        )

        # Check shapes
        assert Y_hat.shape == (n_samples,)
        assert isinstance(RMSE, float)
        assert RMSE >= 0


def test_available_method():
    """Test the available_method function"""
    # Get available methods
    methods = available_method(print_out=False)

    # Check that methods is a list
    assert isinstance(methods, list)

    # Check that "grf" is in the list
    assert "grf" in methods
