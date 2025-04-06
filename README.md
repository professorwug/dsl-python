# DSL: Design-based Supervised Learning (Python)

## Repository Overview

This repository hosts parallel implementations of the Design-based Supervised Learning (DSL) framework in **Python**. Special thanks to Chandler L'Hommedieu for his help with the ideation and implementation of the Python version.

The primary goal of the Python implementation was to create a version that closely mirrors the statistical methodology and produces comparable results to the established **R** package, originally developed by Naoki Egami.

DSL combines supervised machine learning techniques with methods from survey statistics and econometrics to estimate regression models when outcome labels are only available for a non-random subset of the data (partially labeled data).

## Original R Package Documentation

For the theoretical background, detailed methodology, and original R package usage, please refer to the original package resources:

*   **Package Website & Vignettes:** [http://naokiegami.com/dsl](http://naokiegami.com/dsl)
*   **Original R Package Repository:** [https://github.com/naoki-egami/dsl](https://github.com/naoki-egami/dsl)

## Installation

### Prerequisites

*   Python 3.9+
*   pip (Python package installer)

### From PyPI

```bash
pip install dsl
```

### From Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Enan456/dsl-python.git
    cd dsl-python
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install in development mode:**
    ```bash
    pip install -e .
    ```

## Usage

The core estimation function is `dsl.dsl()`. Here's a basic example:

```python
import pandas as pd
from patsy import dmatrices
from dsl.dsl import dsl

# Prepare your data
# Your data should have:
# - outcome variable (y)
# - predictor variables (X)
# - labeled_ind: binary indicator for labeled data (1) or unlabeled data (0)
# - sample_prob: sampling probability for each observation

# Define your model formula
formula = "y ~ x1 + x2 + x3"

# Prepare design matrix (X) and response (y)
y, X = dmatrices(formula, data, return_type="dataframe")

# Run DSL estimation
result = dsl(
    X=X.values,
    y=y.values.flatten(),  # Ensure y is 1D
    labeled_ind=data["labeled"].values,
    sample_prob=data["sample_prob"].values,
    model="logit",  # Use "logit" for binary outcomes, "lm" for continuous
    method="logistic"  # Use "logistic" for logit, "linear" for lm
)

# Access results
print(f"Convergence: {result.success}")
print(f"Iterations: {result.niter}")
print(f"Coefficients: {result.coefficients}")
print(f"Standard Errors: {result.standard_errors}")
```

For a complete example using the PanChen dataset, see the tests directory.

## ELI5 

Imagine you have a large dataset of images, but only a few of them are labeled with their contents. DSL is like having a smart algorithm that can learn from the labeled images to predict the contents of the unlabeled ones. It uses patterns and features from the known data to make educated guesses about the unknown data, helping you understand the entire dataset better. DSL is particularly useful when working with synthetic data, where you can generate additional labeled examples to improve the model's performance.

When you have synthetic data, you can create more examples that mimic the real data. DSL can then use these synthetic examples to learn more about the patterns in your data, making it even better at predicting the contents of unlabeled images. This approach is especially helpful when you have limited real data but need a robust model.

DSL can also help you find the best way to split your data for training and testing. By analyzing how well the model performs on different parts of your data, DSL can identify effective splits that improve model accuracy. Additionally, DSL can detect biases in synthetic data, ensuring that your model is fair and representative of the real-world data it will encounter.

## Potential Applications

DSL can be used in various fields, such as:

- **Social Sciences:** Analyzing survey data where only a subset of responses are labeled.
- **Machine Learning:** Improving model performance when labeled data is limited.
- **Econometrics:** Estimating models with partially observed outcomes.
- **Healthcare:** Predicting patient outcomes with limited labeled data.
- **Synthetic Data Generation:** Creating and utilizing synthetic data to enhance model training and validation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License
