"""
Unit tests for the DSLResult class
"""

import numpy as np
import pytest

from dsl.dsl import DSLResult


def test_dsl_result_initialization():
    """Test initialization of DSLResult class"""
    # Create sample data
    coefficients = np.array([1.0, 0.5, -0.3, 0.2, 0.1])
    standard_errors = np.array([0.1, 0.05, 0.03, 0.02, 0.01])
    predicted_values = np.array([0.8, 0.4, -0.2, 0.1, 0.05])
    residuals = np.array([0.2, 0.1, -0.1, 0.1, 0.05])
    vcov = np.eye(5) * 0.01
    objective = 0.5
    success = True
    message = "Optimization successful"
    niter = 10
    model = "lm"
    labeled_size = 100
    total_size = 150

    # Create DSLResult object
    result = DSLResult(
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

    # Check attributes
    assert np.array_equal(result.coefficients, coefficients)
    assert np.array_equal(result.standard_errors, standard_errors)
    assert np.array_equal(result.predicted_values, predicted_values)
    assert np.array_equal(result.residuals, residuals)
    assert np.array_equal(result.vcov, vcov)
    assert result.objective == objective
    assert result.success == success
    assert result.message == message
    assert result.niter == niter
    assert result.model == model
    assert result.labeled_size == labeled_size
    assert result.total_size == total_size


def test_dsl_result_indexing():
    """Test indexing of DSLResult class"""
    # Create sample data
    coefficients = np.array([1.0, 0.5, -0.3, 0.2, 0.1])
    standard_errors = np.array([0.1, 0.05, 0.03, 0.02, 0.01])
    predicted_values = np.array([0.8, 0.4, -0.2, 0.1, 0.05])
    residuals = np.array([0.2, 0.1, -0.1, 0.1, 0.05])
    vcov = np.eye(5) * 0.01
    objective = 0.5
    success = True
    message = "Optimization successful"
    niter = 10
    model = "lm"
    labeled_size = 100
    total_size = 150

    # Create DSLResult object
    result = DSLResult(
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

    # Test indexing
    assert np.array_equal(result[0], coefficients)
    assert np.array_equal(result[1], standard_errors)
    assert np.array_equal(result[2], vcov)
    assert result[3] == objective
    assert result[4] == success

    # Test out of range indexing
    with pytest.raises(IndexError):
        result[5]
