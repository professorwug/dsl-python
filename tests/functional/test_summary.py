import numpy as np
from dsl_kit import dsl, power_dsl, summary, summary_power


def test_summary_dsl(sample_data, sample_prediction):
    """Test summary_dsl function"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run DSL
    dsl_result = dsl(
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

    # Run summary
    result = summary(dsl_result)

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "t_values")
    assert hasattr(result, "p_values")

    # Check shapes
    assert result.coefficients.shape == (6,)  # 5 features + intercept
    assert result.standard_errors.shape == (6,)
    assert result.t_values.shape == (6,)
    assert result.p_values.shape == (6,)

    # Check values
    assert np.all(np.isfinite(result.coefficients))
    assert np.all(result.standard_errors >= 0)
    assert np.all(np.isfinite(result.t_values))
    assert np.all((result.p_values >= 0) & (result.p_values <= 1))


def test_summary_power(sample_data, sample_prediction):
    """Test summary_power function"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run power_dsl
    power_result = power_dsl(
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
        alpha=0.05,
        power=0.8,
        seed=1234,
    )

    # Run summary_power
    result = summary_power(power_result)

    # Check result
    assert result is not None
    assert hasattr(result, "power")
    assert hasattr(result, "predicted_se")
    assert hasattr(result, "critical_value")
    assert hasattr(result, "alpha")

    # Check values
    assert isinstance(result.power, float)
    assert isinstance(result.predicted_se, float)
    assert isinstance(result.critical_value, float)
    assert isinstance(result.alpha, float)
    assert 0 <= result.power <= 1
    assert result.predicted_se >= 0
    assert result.critical_value >= 0
    assert 0 <= result.alpha <= 1
