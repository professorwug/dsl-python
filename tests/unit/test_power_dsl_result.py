"""
Unit tests for the PowerDSLResult class
"""

import numpy as np
import pytest

from dsl.dsl import DSLResult, PowerDSLResult


def test_power_dsl_result_initialization():
    """Test initialization of PowerDSLResult class"""
    # Create sample data
    power = np.array([0.8, 0.9, 0.95])
    predicted_se = np.array([[0.1, 0.2, 0.3], [0.05, 0.1, 0.15]])
    critical_value = 1.96
    alpha = 0.05

    # Create a dummy DSLResult object
    coefficients = np.array([1.0, 0.5, -0.3])
    standard_errors = np.array([0.1, 0.05, 0.03])
    predicted_values = np.array([0.8, 0.4, -0.2])
    residuals = np.array([0.2, 0.1, -0.1])
    vcov = np.eye(3) * 0.01
    objective = 0.5
    success = True
    message = "Optimization successful"
    niter = 10
    model = "lm"
    labeled_size = 100
    total_size = 150
    dsl_out = DSLResult(
        coefficients=coefficients,
        standard_errors=standard_errors,
        predicted_values=predicted_values,
        residuals=residuals,
        vcov=vcov,
        objective=objective,
        success=success,
        message=message,
        niter=niter,
        model=model,
        labeled_size=labeled_size,
        total_size=total_size,
    )

    # Create PowerDSLResult object
    result = PowerDSLResult(
        power=power,
        predicted_se=predicted_se,
        critical_value=critical_value,
        alpha=alpha,
        dsl_out=dsl_out,
    )

    # Check attributes
    assert np.array_equal(result.power, power)
    assert np.array_equal(result.predicted_se, predicted_se)
    assert result.critical_value == critical_value
    assert result.alpha == alpha
    assert result.dsl_out == dsl_out


def test_power_dsl_result_indexing():
    """Test indexing of PowerDSLResult class"""
    # Create sample data
    power = np.array([0.8, 0.9, 0.95])
    predicted_se = np.array([[0.1, 0.2, 0.3], [0.05, 0.1, 0.15]])
    critical_value = 1.96
    alpha = 0.05

    # Create a dummy DSLResult object
    coefficients = np.array([1.0, 0.5, -0.3])
    standard_errors = np.array([0.1, 0.05, 0.03])
    predicted_values = np.array([0.8, 0.4, -0.2])
    residuals = np.array([0.2, 0.1, -0.1])
    vcov = np.eye(3) * 0.01
    objective = 0.5
    success = True
    message = "Optimization successful"
    niter = 10
    model = "lm"
    labeled_size = 100
    total_size = 150
    dsl_out = DSLResult(
        coefficients=coefficients,
        standard_errors=standard_errors,
        predicted_values=predicted_values,
        residuals=residuals,
        vcov=vcov,
        objective=objective,
        success=success,
        message=message,
        niter=niter,
        model=model,
        labeled_size=labeled_size,
        total_size=total_size,
    )

    # Create PowerDSLResult object
    result = PowerDSLResult(
        power=power,
        predicted_se=predicted_se,
        critical_value=critical_value,
        alpha=alpha,
        dsl_out=dsl_out,
    )

    # Test indexing
    assert np.array_equal(result["power"], power)
    assert np.array_equal(result["predicted_se"], predicted_se)
    assert result["critical_value"] == critical_value
    assert result["alpha"] == alpha
    assert result["dsl_out"] == dsl_out

    # Test invalid key
    with pytest.raises(KeyError):
        result["invalid_key"]
