"""
Functional tests for the summary and plot functions
"""

import numpy as np

from dsl.dsl import dsl, plot_power, power_dsl, summary, summary_power


def test_summary_dsl(sample_data, sample_prediction):
    """Test summary function for DSL results"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Run DSL
    dsl_result = dsl(
        model="lm",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        cluster="cluster",
        labeled="labeled",
        sample_prob="sample_prob",
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        seed=1234,
    )

    # Run summary
    summary_table = summary(dsl_result, ci=0.95, digits=4)

    # Check result
    assert summary_table is not None
    assert isinstance(summary_table, np.ndarray)
    assert summary_table.shape[0] == 6  # 5 features + intercept
    assert summary_table.shape[1] >= 4  # At least coefficients, SE, CI, p-value


def test_summary_power_dsl(sample_data, sample_prediction):
    """Test summary_power function for power_dsl results"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Run DSL first
    dsl_result = dsl(
        model="lm",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        cluster="cluster",
        labeled="labeled",
        sample_prob="sample_prob",
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        seed=1234,
    )

    # Run power_dsl
    labeled_sizes = [100, 200, 300]
    power_result = power_dsl(labeled_size=labeled_sizes, dsl_out=dsl_result)

    # Run summary_power
    summary_table = summary_power(power_result)

    # Check result
    assert summary_table is not None
    assert isinstance(summary_table, np.ndarray)
    assert summary_table.shape[0] == len(labeled_sizes)
    assert summary_table.shape[1] == 6  # 5 features + intercept


def test_plot_power(sample_data, sample_prediction):
    """Test plot_power function"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Run DSL first
    dsl_result = dsl(
        model="lm",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        cluster="cluster",
        labeled="labeled",
        sample_prob="sample_prob",
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        seed=1234,
    )

    # Run power_dsl
    labeled_sizes = [100, 200, 300]
    power_result = power_dsl(labeled_size=labeled_sizes, dsl_out=dsl_result)

    # Run plot_power for all coefficients
    plot_power(power_result)

    # Run plot_power for a specific coefficient
    plot_power(power_result, coef_name="x1")

    # Run plot_power for multiple coefficients
    plot_power(power_result, coef_name=["x1", "x2"])
