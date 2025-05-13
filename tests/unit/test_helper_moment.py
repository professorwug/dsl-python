"""
Unit tests for the moment estimation helper functions
"""

import numpy as np
import pandas as pd
import pytest
from dsl_kit.helpers.moment import (
    demean_dsl,
    felm_dsl_moment_base,
    lm_dsl_Jacobian,
    lm_dsl_moment_base,
    lm_dsl_moment_orig,
    lm_dsl_moment_pred,
    logit_dsl_Jacobian,
    logit_dsl_moment_base,
    logit_dsl_moment_orig,
    logit_dsl_moment_pred,
)


def test_lm_dsl_moment_base():
    """Test the lm_dsl_moment_base function"""
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

    # Compute moment function
    m_dr = lm_dsl_moment_base(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_dr.shape == (n_samples, n_features)

    # Check that unlabeled observations have zero moment
    unlabeled_indices = np.where(labeled_ind == 0)[0]
    assert np.allclose(m_dr[unlabeled_indices], 0)


def test_lm_dsl_moment_orig():
    """Test the lm_dsl_moment_orig function"""
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

    # Compute moment function
    m_orig = lm_dsl_moment_orig(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_orig.shape == (n_samples, n_features)

    # Check that unlabeled observations have zero moment
    unlabeled_indices = np.where(labeled_ind == 0)[0]
    assert np.allclose(m_orig[unlabeled_indices], 0)


def test_lm_dsl_moment_pred():
    """Test the lm_dsl_moment_pred function"""
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

    # Compute moment function
    m_pred = lm_dsl_moment_pred(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_pred.shape == (n_samples, n_features)


def test_lm_dsl_Jacobian():
    """Test the lm_dsl_Jacobian function"""
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

    # Compute Jacobian
    J = lm_dsl_Jacobian(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, "lm"
    )

    # Check shape
    assert J.shape == (n_features, n_features)

    # Check that Jacobian is symmetric
    assert np.allclose(J, J.T)


def test_logit_dsl_moment_base():
    """Test the logit_dsl_moment_base function"""
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
    logit_orig = 1 / (1 + np.exp(-(X_orig @ par)))
    Y_orig = np.random.binomial(1, logit_orig, n_samples)

    X_pred = np.random.randn(n_samples, n_features)
    logit_pred = 1 / (1 + np.exp(-(X_pred @ par)))
    Y_pred = np.random.binomial(1, logit_pred, n_samples)

    # Compute moment function
    m_dr = logit_dsl_moment_base(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_dr.shape == (n_samples, n_features)

    # Check that unlabeled observations have zero moment
    unlabeled_indices = np.where(labeled_ind == 0)[0]
    assert np.allclose(m_dr[unlabeled_indices], 0)


def test_logit_dsl_moment_orig():
    """Test the logit_dsl_moment_orig function"""
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
    logit_orig = 1 / (1 + np.exp(-(X_orig @ par)))
    Y_orig = np.random.binomial(1, logit_orig, n_samples)

    X_pred = np.random.randn(n_samples, n_features)
    logit_pred = 1 / (1 + np.exp(-(X_pred @ par)))
    Y_pred = np.random.binomial(1, logit_pred, n_samples)

    # Compute moment function
    m_orig = logit_dsl_moment_orig(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_orig.shape == (n_samples, n_features)

    # Check that unlabeled observations have zero moment
    unlabeled_indices = np.where(labeled_ind == 0)[0]
    assert np.allclose(m_orig[unlabeled_indices], 0)


def test_logit_dsl_moment_pred():
    """Test the logit_dsl_moment_pred function"""
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
    logit_orig = 1 / (1 + np.exp(-(X_orig @ par)))
    Y_orig = np.random.binomial(1, logit_orig, n_samples)

    X_pred = np.random.randn(n_samples, n_features)
    logit_pred = 1 / (1 + np.exp(-(X_pred @ par)))
    Y_pred = np.random.binomial(1, logit_pred, n_samples)

    # Compute moment function
    m_pred = logit_dsl_moment_pred(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_pred.shape == (n_samples, n_features)


def test_logit_dsl_Jacobian():
    """Test the logit_dsl_Jacobian function"""
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
    logit_orig = 1 / (1 + np.exp(-(X_orig @ par)))
    Y_orig = np.random.binomial(1, logit_orig, n_samples)

    X_pred = np.random.randn(n_samples, n_features)
    logit_pred = 1 / (1 + np.exp(-(X_pred @ par)))
    Y_pred = np.random.binomial(1, logit_pred, n_samples)

    # Compute Jacobian
    J = logit_dsl_Jacobian(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert J.shape == (n_features, n_features)

    # Check that Jacobian is symmetric
    assert np.allclose(J, J.T)


def test_felm_dsl_moment_base():
    """Test the felm_dsl_moment_base function"""
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

    # Compute moment function
    m_dr = felm_dsl_moment_base(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, fe_Y, fe_X
    )

    # Check shape
    assert m_dr.shape == (n_samples, n_features)

    # Check that unlabeled observations have zero moment
    unlabeled_indices = np.where(labeled_ind == 0)[0]
    assert np.allclose(m_dr[unlabeled_indices], 0)


def test_demean_dsl():
    """Test the demean_dsl function"""
    # Create sample data
    n_samples = 100
    n_features = 5

    # Generate data
    data_base = pd.DataFrame(
        {"id": range(n_samples), "group": np.random.randint(0, 10, n_samples)}
    )

    # Generate outcome and features
    adj_Y = np.random.randn(n_samples)
    adj_X = np.random.randn(n_samples, n_features)

    # Compute demeaned data
    adj_Y_avg_exp, adj_X_avg_exp, adj_data = demean_dsl(
        data_base, adj_Y, adj_X, ["group"]
    )

    # Check shapes
    assert adj_Y_avg_exp.shape == (n_samples,)
    assert adj_X_avg_exp.shape == (n_samples, n_features)
    assert adj_data.shape == (n_samples, n_features + 2)  # +2 for id and group

    # Check that demeaned data has mean zero within each group
    for group in data_base["group"].unique():
        group_indices = data_base["group"] == group
        assert np.allclose(np.mean(adj_Y_avg_exp[group_indices]), 0, atol=1e-10)
        assert np.allclose(np.mean(adj_X_avg_exp[group_indices], axis=0), 0, atol=1e-10)
