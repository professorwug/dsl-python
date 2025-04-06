DSL (Double-Supervised Learning) Framework

A Python package for implementing double-supervised learning methods.

Installation
------------

You can install the package using pip:

    pip install dsl

Or install from source:

    git clone https://github.com/yourusername/dsl.git
    cd dsl
    pip install -e .

Usage
-----

Basic usage:

    from dsl import dsl

    # Run DSL estimation
    result = dsl(
        formula="y ~ x1 + x2",
        data=df,
        labeled_ind=labeled_indicator,
        sample_prob=sampling_probabilities
    )

    # Print summary
    print(summary(result))

Fixed effects regression:

    # Run DSL estimation with fixed effects
    result = dsl(
        formula="y ~ x1 + x2",
        data=df,
        labeled_ind=labeled_indicator,
        sample_prob=sampling_probabilities,
        fe="group"
    )

Power analysis:

    # Run power analysis
    power_result = power_dsl(
        formula="y ~ x1 + x2",
        data=df,
        labeled_ind=labeled_indicator,
        sample_prob=sampling_probabilities,
        n_samples=1000
    )

    # Plot power analysis results
    plot_power(power_result)

Functions
---------

- dsl: Main DSL estimation function
- power_dsl: Power analysis for DSL estimation
- summary: Summarize DSL estimation results
- summary_power: Summarize power analysis results
- plot_power: Plot power analysis results

References
----------

1. Your reference paper here
2. Additional references here 