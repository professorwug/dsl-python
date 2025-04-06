import matplotlib.pyplot as plt


def plot_power(power_result):
    """Plot power analysis results.

    Parameters
    ----------
    power_result : PowerDSLResult
        Output from power_dsl function.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the power plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot predicted standard errors
    ax.plot(power_result.predicted_se, label="Predicted SE")

    # Add critical value line
    ax.axhline(
        y=power_result.critical_value, color="r", linestyle="--", label="Critical Value"
    )

    # Customize plot
    ax.set_xlabel("Coefficient Index")
    ax.set_ylabel("Standard Error")
    ax.set_title("Power Analysis Results")
    ax.legend()
    ax.grid(True)

    return fig
