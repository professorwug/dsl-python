import matplotlib.pyplot as plt
from dsl_kit.plot import plot_power
from dsl_kit.power_dsl import power_dsl


def test_plot_power(sample_data, sample_prediction):
    """Test plot_power function"""
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

    # Run plot_power
    fig = plot_power(power_result)

    # Check result
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
