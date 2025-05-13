"""
Core DSL (Double-Supervised Learning) module
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .helpers.dsl_general import dsl_general
from .helpers.estimate import estimate_power


@dataclass
class DSLResult:
    """Results from DSL estimation."""

    coefficients: np.ndarray
    standard_errors: np.ndarray
    vcov: np.ndarray
    objective: float
    success: bool
    message: str
    niter: int
    model: str
    labeled_size: int
    total_size: int
    predicted_values: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

    def __getitem__(self, key):
        """Allow indexing of DSLResult object."""
        if key == 0:
            return self.coefficients
        elif key == 1:
            return self.standard_errors
        elif key == 2:
            return self.vcov
        else:
            raise IndexError("DSLResult index out of range")


@dataclass
class PowerDSLResult:
    """Results from DSL power analysis."""

    power: np.ndarray
    predicted_se: np.ndarray
    critical_value: float
    alpha: float
    dsl_out: Optional[DSLResult] = None


def dsl(
    X: np.ndarray,
    y: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob: np.ndarray,
    model: str = "logit",
    method: str = "linear",
    fe_Y: Optional[np.ndarray] = None,
    fe_X: Optional[np.ndarray] = None,
) -> DSLResult:
    """
    Estimate DSL model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix
    y : np.ndarray
        Response variable
    labeled_ind : np.ndarray
        Labeled indicator
    sample_prob : np.ndarray
        Sampling probability
    model : str, optional
        Model type, by default "logit"
    method : str, optional
        Method for estimation, by default "linear"
    fe_Y : np.ndarray, optional
        Fixed effects for the outcome variable, by default None
    fe_X : np.ndarray, optional
        Fixed effects for the predictor variables, by default None

    Returns
    -------
    DSLResult
        Object containing estimation results
    """
    # Determine model type for dsl_general
    model_internal = "logit"  # Default
    if method == "linear":
        model_internal = "lm"
    elif method == "logistic":
        model_internal = "logit"
    elif method == "fixed_effects":
        model_internal = "felm"
        # Check if fixed effects are provided
        if fe_Y is None or fe_X is None:
            raise ValueError("Fixed effects models require fe_Y and fe_X parameters")
    # Keep original model name for result object
    model_name_for_result = model

    # Estimate parameters using the general function
    par, info = dsl_general(
        y,  # Pass y directly (will be flattened inside if needed)
        X,
        y,  # Pass y directly
        X,
        labeled_ind,
        sample_prob,
        model=model_internal,  # Use determined internal model type
        fe_Y=fe_Y,  # Pass fixed effects for outcome
        fe_X=fe_X,  # Pass fixed effects for predictors
    )

    # Note: dsl_vcov might be redundant if vcov is already computed in dsl_general
    # vcov = dsl_vcov(X, par, info["standard_errors"], model_internal)
    vcov = info["vcov"]  # Use vcov from dsl_general info dict

    # Calculate predicted values and residuals
    predicted_values = None
    residuals = None

    if model_internal == "lm":
        # For linear models
        predicted_values = np.dot(X, par)
        residuals = y - predicted_values
    elif model_internal == "logit":
        # For logistic models
        predicted_values = 1 / (1 + np.exp(-np.dot(X, par)))
        residuals = y - predicted_values
    elif model_internal == "felm":
        # For fixed effects models
        # Split parameters into main effects and fixed effects
        n_main = X.shape[1]  # Number of main effect parameters
        par_main = par[:n_main]
        par_fe = par[n_main:]

        # Compute main effects and fixed effects contributions
        predicted_values = np.dot(X, par_main)
        if fe_X is not None:
            predicted_values += np.dot(fe_X, par_fe).flatten()
        residuals = y - predicted_values

    # Populate and return DSLResult object
    return DSLResult(
        coefficients=par,
        standard_errors=info["standard_errors"],
        vcov=vcov,
        objective=info["objective"],
        success=info["convergence"],
        message=info["message"],
        niter=info["iterations"],
        model=model_name_for_result,  # Use original model name
        labeled_size=int(np.sum(labeled_ind)),
        total_size=X.shape[0],
        predicted_values=predicted_values,
        residuals=residuals,
    )


def power_dsl(
    formula: str,
    data: pd.DataFrame,
    labeled_ind: np.ndarray,
    sample_prob: Optional[np.ndarray] = None,
    model: str = "lm",
    fe: Optional[str] = None,
    method: str = "linear",
    n_samples: Optional[int] = None,
    alpha: float = 0.05,
    dsl_out: Optional[DSLResult] = None,
    **kwargs,
) -> PowerDSLResult:
    """
    Perform DSL power analysis.

    Parameters
    ----------
    formula : str
        Model formula
    data : pd.DataFrame
        Data frame
    labeled_ind : np.ndarray
        Labeled indicator
    sample_prob : Optional[np.ndarray], optional
        Sampling probability, by default None
    model : str, optional
        Model type, by default "lm"
    fe : Optional[str], optional
        Fixed effects variable, by default None
    method : str, optional
        Supervised learning method, by default "linear"
    n_samples : Optional[int], optional
        Number of samples for power analysis, by default None
    alpha : float, optional
        Significance level, by default 0.05
    dsl_out : Optional[DSLResult], optional
        DSL estimation results, by default None
    **kwargs : dict
        Additional arguments for the estimator

    Returns
    -------
    PowerDSLResult
        DSL power analysis results
    """
    # Estimate DSL model if not provided
    if dsl_out is None:
        dsl_out = dsl(
            data.values,
            data.values[:, 0],
            data.values[:, 0],
            data.values[:, 1],
            model,
            method,
        )

    # Parse formula
    from patsy import dmatrices

    _, X = dmatrices(formula, data, return_type="dataframe")
    X = X.values

    # Set default number of samples
    if n_samples is None:
        n_samples = len(data)

    # Estimate power
    power_results = estimate_power(
        X,
        dsl_out.coefficients,
        dsl_out.standard_errors,
        n_samples,
        alpha,
    )

    # Return results
    return PowerDSLResult(
        power=power_results["power"],
        predicted_se=power_results["predicted_se"],
        critical_value=power_results["critical_value"],
        alpha=power_results["alpha"],
        dsl_out=dsl_out,
    )


def summary(result: DSLResult) -> pd.DataFrame:
    """
    Summarize DSL estimation results.

    Parameters
    ----------
    result : DSLResult
        DSL estimation results

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    # Create summary table
    summary = pd.DataFrame(
        {
            "Estimate": result.coefficients,
            "Std. Error": result.standard_errors,
            "t value": result.coefficients / result.standard_errors,
            "Pr(>|t|)": 2
            * (
                1
                - stats.t.cdf(
                    np.abs(result.coefficients / result.standard_errors),
                    len(result.residuals) - len(result.coefficients),
                )
            ),
        }
    )

    return summary


def summary_power(result: PowerDSLResult) -> pd.DataFrame:
    """
    Summarize DSL power analysis results.

    Parameters
    ----------
    result : PowerDSLResult
        DSL power analysis results

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    # Create summary table
    summary = pd.DataFrame(
        {
            "Power": result.power,
            "Predicted SE": result.predicted_se,
        }
    )

    return summary


def plot_power(
    result: PowerDSLResult,
    coefficients: Optional[Union[str, List[str]]] = None,
) -> None:
    """
    Plot DSL power analysis results.

    Parameters
    ----------
    result : PowerDSLResult
        DSL power analysis results
    coefficients : Optional[Union[str, List[str]]], optional
        Coefficients to plot, by default None
    """
    import matplotlib.pyplot as plt

    # Get coefficient names
    if result.dsl_out is not None:
        from patsy import dmatrices

        _, X = dmatrices(
            result.dsl_out.formula, result.dsl_out.data, return_type="dataframe"
        )
        coef_names = X.columns
    else:
        coef_names = [f"beta_{i}" for i in range(len(result.power))]

    # Select coefficients to plot
    if coefficients is None:
        coefficients = coef_names
    elif isinstance(coefficients, str):
        coefficients = [coefficients]

    # Create plot
    plt.figure(figsize=(10, 6))
    for coef in coefficients:
        idx = coef_names.index(coef)
        plt.plot(
            result.predicted_se[idx],
            result.power[idx],
            label=coef,
            marker="o",
        )

    plt.xlabel("Predicted Standard Error")
    plt.ylabel("Power")
    plt.title("DSL Power Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()
