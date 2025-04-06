"""
Unit tests for the general DSL helper functions
"""

import numpy as np
import pandas as pd
from dsl.helpers.dsl_general import (
    dsl_general_Jacobian,
    dsl_general_moment,
    dsl_general_moment_base_decomp,
    dsl_general_moment_est,
)


def test_dsl_general_moment():
    """Test the dsl_general_moment function"""
    # Create sample data
    n_samples = 100
    n_features = 5

    # Generate parameters
    par = np.array([1.0, 0.5, -0.3, 0.2, 0.1])

    # Generate data
    labeled_ind = np.random.binomial(1, 0.8, n_samples)
    sample_prob_use = np.ones(n_samples) * 0.8

    # Generate outcome and features
    X_orig = np.random.randn(n_samples, n_features)
    Y_orig = X_orig @ par + np.random.randn(n_samples) * 0.1

    X_pred = np.random.randn(n_samples, n_features)
    Y_pred = X_pred @ par + np.random.randn(n_samples) * 0.1

    # Generate fixed effects
    n_fixed_effects = 10
    fe_Y = np.random.randn(n_samples)
    fe_X = np.random.randn(n_samples, n_fixed_effects)

    # Test with different models
    for model in ["lm", "logit", "felm"]:
        # Compute moment function
        g_out = dsl_general_moment(
            par,
            labeled_ind,
            sample_prob_use,
            Y_orig,
            X_orig,
            Y_pred,
            X_pred,
            fe_Y,
            fe_X,
            model,
            0.00001,
        )

        # Check that g_out is a scalar
        assert isinstance(g_out, float)
        assert g_out >= 0


def test_dsl_general_moment_base_decomp():
    """Test the dsl_general_moment_base_decomp function"""
    # Create sample data
    n_samples = 100
    n_features = 5

    # Generate parameters
    par = np.array([1.0, 0.5, -0.3, 0.2, 0.1])

    # Generate data
    labeled_ind = np.random.binomial(1, 0.8, n_samples)
    sample_prob_use = np.ones(n_samples) * 0.8

    # Generate outcome and features
    X_orig = np.random.randn(n_samples, n_features)
    Y_orig = X_orig @ par + np.random.randn(n_samples) * 0.1

    X_pred = np.random.randn(n_samples, n_features)
    Y_pred = X_pred @ par + np.random.randn(n_samples) * 0.1

    # Test with different models and clustering
    for model in ["lm", "logit", "felm"]:
        for clustered in [True, False]:
            # Create cluster indicator if needed
            cluster = None
            if clustered:
                cluster = np.random.randint(0, 10, n_samples)

            # Compute moment decomposition
            main_1, main_23 = dsl_general_moment_base_decomp(
                par,
                labeled_ind,
                sample_prob_use,
                Y_orig,
                X_orig,
                Y_pred,
                X_pred,
                model,
                clustered,
                cluster,
            )

            # Check shapes
            assert main_1.shape == (n_features, n_features)
            assert main_23.shape == (n_features, n_features)


def test_dsl_general_Jacobian():
    """Test the dsl_general_Jacobian function"""
    # Create sample data
    n_samples = 100
    n_features = 5

    # Generate parameters
    par = np.array([1.0, 0.5, -0.3, 0.2, 0.1])

    # Generate data
    labeled_ind = np.random.binomial(1, 0.8, n_samples)
    sample_prob_use = np.ones(n_samples) * 0.8

    # Generate outcome and features
    X_orig = np.random.randn(n_samples, n_features)
    Y_orig = X_orig @ par + np.random.randn(n_samples) * 0.1

    X_pred = np.random.randn(n_samples, n_features)
    Y_pred = X_pred @ par + np.random.randn(n_samples) * 0.1

    # Test with different models
    for model in ["lm", "logit", "felm"]:
        # Compute Jacobian
        J = dsl_general_Jacobian(
            par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, model
        )

        # Check shape
        assert J.shape == (n_features, n_features)

        # Check that Jacobian is symmetric
        assert np.allclose(J, J.T)


def test_dsl_general_moment_est():
    """Test the dsl_general_moment_est function"""
    # Create sample data
    n_samples = 100
    n_features = 5

    # Generate data
    data = pd.DataFrame(
        {
            "y": np.random.randn(n_samples),
            "labeled": np.random.binomial(1, 0.8, n_samples),
            "sample_prob": np.ones(n_samples) * 0.8,
            "cluster": np.random.randint(0, 10, n_samples),
            "fixed_effect": np.random.randint(0, 5, n_samples),
        }
    )

    # Add features
    for i in range(n_features):
        data[f"x{i+1}"] = np.random.randn(n_samples)

    # Create formula
    formula = "y ~ x1 + x2 + x3 + x4 + x5"

    # Create predicted variable
    predicted_var = ["y"]

    # Create prediction
    prediction = np.random.randn(n_samples)
    data["prediction"] = prediction

    # Test with different models
    for model in ["lm", "logit", "felm"]:
        # Compute moment estimation
        result = dsl_general_moment_est(
            model=model,
            formula=formula,
            labeled="labeled",
            sample_prob="sample_prob",
            predicted_var=predicted_var,
            data_orig=data,
            data_pred=data,
            index=["fixed_effect"] if model == "felm" else None,
            fixed_effect="oneway" if model == "felm" else None,
            clustered=True,
            cluster="cluster",
        )

        # Check result
        assert "coefficients" in result
        assert "standard_errors" in result
        assert "vcov" in result
        assert "Meat" in result
        assert "Meat_decomp" in result
        assert "J" in result
        assert "D" in result
        assert "vcov0" in result
