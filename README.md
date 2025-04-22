# DSL-Kit: Design-based Supervised Learning (Python)

## Repository Overview

This repository hosts parallel implementations of the Design-based Supervised Learning (DSL) framework in **Python**. Special thanks to Chandler L'Hommedieu for his help with the ideation and implementation of the Python version.

The primary goal of the Python implementation was to create a version that closely mirrors the statistical methodology and produces comparable results to the established **R** package, originally developed by Naoki Egami.

## What is DSL?

The DSL method (Design-based Supervised Learning) is a framework designed to correct for biases that occur when automated text annotations (from LLMs or supervised ML methods) are used as if they were error‐free in downstream statistical analyses.

### Key Components

#### Two-Step Approach
1. **Automated Annotation**: Use an automated method (e.g., an LLM) to label a large corpus of text.
2. **Expert Correction**: Randomly sample a subset of documents for expert coding. The known sampling probabilities allow researchers to estimate the average prediction error.

#### Bias Correction
DSL constructs a design-adjusted outcome by subtracting a bias-correction term (derived from the difference between automated predictions and expert labels, weighted by the sampling probability) from the predicted outcome. This adjustment ensures that prediction errors—especially when they are non-random—do not bias subsequent regression estimates.

#### Robust Inference
By leveraging a doubly robust estimation approach (similar to augmented inverse probability weighting), DSL provides consistent estimates and valid confidence intervals, even if the automated annotation errors are correlated with other variables in the analysis.

#### General Applicability
Although demonstrated in the context of text classification tasks, DSL is general enough to be applied to various types of statistical models (e.g., linear regression, logistic regression) and can accommodate different types of automated annotation methods.

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
pip install dsl_kit
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
import numpy as np
from patsy import dmatrices
from dsl_kit.dsl import dsl

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

### Function Parameters

The `dsl()` function has the following parameters:

```python
def dsl(
    X: np.ndarray,
    y: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob: np.ndarray,
    model: str = "logit",
    method: str = "linear",
    fe_Y: np.ndarray = None,
    fe_X: np.ndarray = None,
) -> DSLResult:
```

- **X**: Design matrix (numpy array)
- **y**: Response variable (numpy array)
- **labeled_ind**: Binary indicator for labeled data (1) or unlabeled data (0)
- **sample_prob**: Sampling probability for each observation
- **model**: Model type, options include:
  - "logit": For binary outcomes (default)
  - "lm": For continuous outcomes
  - "felm": For fixed effects models
- **method**: Estimation method, options include:
  - "linear": For linear models
  - "logistic": For logistic models
  - "fixed_effects": For fixed effects models
- **fe_Y**: Fixed effects for the outcome variable (required for fixed effects models)
- **fe_X**: Fixed effects for the predictor variables (required for fixed effects models)

### Return Value

The function returns a `DSLResult` object with the following attributes:

- **coefficients**: Estimated model coefficients
- **standard_errors**: Standard errors for the coefficients
- **vcov**: Variance-covariance matrix
- **objective**: Value of the objective function at the solution
- **success**: Boolean indicating if the estimation converged
- **message**: Convergence message
- **niter**: Number of iterations performed
- **model**: Model type used
- **labeled_size**: Number of labeled observations
- **total_size**: Total number of observations
- **predicted_values**: Predicted values for all observations
- **residuals**: Residuals for all observations

### Example with Fixed Effects

```python
# For fixed effects models, you need to provide fe_Y and fe_X
result = dsl(
    X=X.values,
    y=y.values.flatten(),
    labeled_ind=data["labeled"].values,
    sample_prob=data["sample_prob"].values,
    model="felm",
    method="fixed_effects",
    fe_Y=data["fixed_effect_y"].values,  # Fixed effects for outcome
    fe_X=data["fixed_effect_x"].values,  # Fixed effects for predictors
)

# Access predicted values and residuals
print(f"Predicted Values: {result.predicted_values}")
print(f"Residuals: {result.residuals}")
```

For a complete example using the PanChen dataset, see the tests directory.

## Applications

### LLM and Synthetic Data Applications
- Validate LLM outputs against expert annotations
- Correct biases in synthetic training data
- Scale annotation while maintaining statistical rigor
- Estimate true model performance with partially labeled data

### General Use Cases
- **Social Sciences:** Analyzing survey data where only a subset of responses are labeled
- **Machine Learning:** Improving model performance when labeled data is limited
- **Econometrics:** Estimating models with partially observed outcomes
- **Healthcare:** Predicting patient outcomes with limited labeled data
- **Synthetic Data Generation:** Creating and utilizing synthetic data to enhance model training and validation

## ELI5 

Imagine you have a large dataset of images, but only a few of them are labeled with their contents. DSL is like having a smart algorithm that can learn from the labeled images to predict the contents of the unlabeled ones. It uses patterns and features from the known data to make educated guesses about the unknown data, helping you understand the entire dataset better. DSL is particularly useful when working with synthetic data, where you can generate additional labeled examples to improve the model's performance.

When you have synthetic data, you can create more examples that mimic the real data. DSL can then use these synthetic examples to learn more about the patterns in your data, making it even better at predicting the contents of unlabeled images. This approach is especially helpful when you have limited real data but need a robust model.

DSL can also help you find the best way to split your data for training and testing. By analyzing how well the model performs on different parts of your data, DSL can identify effective splits that improve model accuracy. Additionally, DSL can detect biases in synthetic data, ensuring that your model is fair and representative of the real-world data it will encounter.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License
