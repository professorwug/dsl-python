"""
Functional tests for the main DSL function
"""

import numpy as np

from dsl_kit import dsl


def test_dsl_linear_regression(sample_data, sample_prediction):
    """Test DSL with linear regression"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run DSL
    result = dsl(
        model="lm",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        seed=1234,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
    assert hasattr(result, "vcov")
    assert hasattr(result, "objective")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "niter")
    assert hasattr(result, "model")
    assert hasattr(result, "labeled_size")
    assert hasattr(result, "total_size")

    # Check shapes
    assert result.coefficients.shape == (6,)  # 5 features + intercept
    assert result.standard_errors.shape == (6,)
    assert result.vcov.shape == (6, 6)
    assert isinstance(result.objective, float)
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.niter, int)
    assert isinstance(result.model, str)
    assert isinstance(result.labeled_size, int)
    assert isinstance(result.total_size, int)
    assert result.labeled_size <= result.total_size


def test_dsl_logistic_regression(sample_data, sample_prediction):
    """Test DSL with logistic regression"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run DSL
    result = dsl(
        model="logit",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="binomial",
        cross_fit=2,
        sample_split=2,
        seed=1234,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
    assert hasattr(result, "vcov")
    assert hasattr(result, "objective")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "niter")
    assert hasattr(result, "model")
    assert hasattr(result, "labeled_size")
    assert hasattr(result, "total_size")

    # Check shapes
    assert result.coefficients.shape == (6,)  # 5 features + intercept
    assert result.standard_errors.shape == (6,)
    assert result.vcov.shape == (6, 6)
    assert isinstance(result.objective, float)
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.niter, int)
    assert isinstance(result.model, str)
    assert isinstance(result.labeled_size, int)
    assert isinstance(result.total_size, int)
    assert result.labeled_size <= result.total_size


def test_dsl_fixed_effects(sample_data, sample_prediction):
    """Test DSL with fixed effects"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run DSL
    result = dsl(
        model="felm",
        formula="y ~ x1 + x2 + x3 + x4 + x5 | fe1 + fe2",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        seed=1234,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
    assert hasattr(result, "vcov")
    assert hasattr(result, "objective")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "niter")
    assert hasattr(result, "model")
    assert hasattr(result, "labeled_size")
    assert hasattr(result, "total_size")

    # Check shapes
    assert result.coefficients.shape == (6,)  # 5 features + intercept
    assert result.standard_errors.shape == (6,)
    assert result.vcov.shape == (6, 6)
    assert isinstance(result.objective, float)
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.niter, int)
    assert isinstance(result.model, str)
    assert isinstance(result.labeled_size, int)
    assert isinstance(result.total_size, int)
    assert result.labeled_size <= result.total_size


def test_dsl_without_prediction(sample_data):
    """Test DSL without providing predictions"""
    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run DSL
    result = dsl(
        model="lm",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        seed=1234,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
    assert hasattr(result, "vcov")
    assert hasattr(result, "objective")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "niter")
    assert hasattr(result, "model")
    assert hasattr(result, "labeled_size")
    assert hasattr(result, "total_size")

    # Check shapes
    assert result.coefficients.shape == (6,)  # 5 features + intercept
    assert result.standard_errors.shape == (6,)
    assert result.vcov.shape == (6, 6)
    assert isinstance(result.objective, float)
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.niter, int)
    assert isinstance(result.model, str)
    assert isinstance(result.labeled_size, int)
    assert isinstance(result.total_size, int)
    assert result.labeled_size <= result.total_size


def test_dsl_without_labeled(sample_data, sample_prediction):
    """Test DSL without providing labeled indicator"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Remove labeled column
    sample_data_no_labeled = sample_data.drop(columns=["labeled"])

    # Run DSL
    result = dsl(
        model="lm",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data_no_labeled,
        labeled_ind=np.ones(len(sample_data_no_labeled)),
        sample_prob=sample_data["sample_prob"].values,
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        seed=1234,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
    assert hasattr(result, "vcov")
    assert hasattr(result, "objective")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "niter")
    assert hasattr(result, "model")
    assert hasattr(result, "labeled_size")
    assert hasattr(result, "total_size")

    # Check shapes
    assert result.coefficients.shape == (6,)  # 5 features + intercept
    assert result.standard_errors.shape == (6,)
    assert result.vcov.shape == (6, 6)
    assert isinstance(result.objective, float)
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.niter, int)
    assert isinstance(result.model, str)
    assert isinstance(result.labeled_size, int)
    assert isinstance(result.total_size, int)
    assert result.labeled_size <= result.total_size


def test_dsl_without_sample_prob(sample_data, sample_prediction):
    """Test DSL without providing sample probabilities"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run DSL
    result = dsl(
        model="lm",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        labeled_ind=labeled_ind,
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        seed=1234,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
    assert hasattr(result, "vcov")
    assert hasattr(result, "objective")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "niter")
    assert hasattr(result, "model")
    assert hasattr(result, "labeled_size")
    assert hasattr(result, "total_size")

    # Check shapes
    assert result.coefficients.shape == (6,)  # 5 features + intercept
    assert result.standard_errors.shape == (6,)
    assert result.vcov.shape == (6, 6)
    assert isinstance(result.objective, float)
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.niter, int)
    assert isinstance(result.model, str)
    assert isinstance(result.labeled_size, int)
    assert isinstance(result.total_size, int)
    assert result.labeled_size <= result.total_size
