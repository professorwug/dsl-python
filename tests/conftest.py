"""
Shared test fixtures for DSL framework tests
"""

import numpy as np
import pandas as pd
import pytest

# Set random seed for reproducibility
np.random.seed(1234)


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    n_samples = 1000
    n_features = 5

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate coefficients
    beta = np.array([1.0, 0.5, -0.3, 0.2, 0.1])

    # Generate outcome (linear model)
    y = X @ beta + np.random.randn(n_samples) * 0.1

    # Create labeled indicator (80% labeled)
    labeled = np.random.binomial(1, 0.8, n_samples)

    # Create sampling probability (equal probability)
    sample_prob = np.ones(n_samples) * 0.8

    # Create cluster indicator (10 clusters)
    cluster = np.random.randint(0, 10, n_samples)

    # Create fixed effects (5 groups)
    fixed_effect = np.random.randint(0, 5, n_samples)

    # Create DataFrame
    data = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_features)])
    data["y"] = y
    data["labeled"] = labeled
    data["sample_prob"] = sample_prob
    data["cluster"] = cluster
    data["fixed_effect"] = fixed_effect

    return data


@pytest.fixture
def sample_data_logit():
    """Generate sample data for logistic regression testing"""
    n_samples = 1000
    n_features = 5

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate coefficients
    beta = np.array([1.0, 0.5, -0.3, 0.2, 0.1])

    # Generate outcome (logistic model)
    logit = 1 / (1 + np.exp(-(X @ beta)))
    y = np.random.binomial(1, logit, n_samples)

    # Create labeled indicator (80% labeled)
    labeled = np.random.binomial(1, 0.8, n_samples)

    # Create sampling probability (equal probability)
    sample_prob = np.ones(n_samples) * 0.8

    # Create cluster indicator (10 clusters)
    cluster = np.random.randint(0, 10, n_samples)

    # Create fixed effects (5 groups)
    fixed_effect = np.random.randint(0, 5, n_samples)

    # Create DataFrame
    data = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_features)])
    data["y"] = y
    data["labeled"] = labeled
    data["sample_prob"] = sample_prob
    data["cluster"] = cluster
    data["fixed_effect"] = fixed_effect

    return data


@pytest.fixture
def sample_prediction(sample_data):
    """Generate sample predictions for testing"""
    # Simple prediction based on features
    X = sample_data[[f"x{i+1}" for i in range(5)]].values
    beta = np.array([1.0, 0.5, -0.3, 0.2, 0.1])
    prediction = X @ beta

    return prediction


@pytest.fixture
def sample_prediction_logit(sample_data_logit):
    """Generate sample predictions for logistic regression testing"""
    # Simple prediction based on features
    X = sample_data_logit[[f"x{i+1}" for i in range(5)]].values
    beta = np.array([1.0, 0.5, -0.3, 0.2, 0.1])
    logit = 1 / (1 + np.exp(-(X @ beta)))

    return logit
