"""
General DSL helper functions
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm  # Import statsmodels
from scipy.optimize import minimize

from .fixed_effects import demean_dsl, felm_dsl_Jacobian, felm_dsl_moment_base
from .moment import (
    lm_dsl_Jacobian,
    lm_dsl_moment_base,
    logit_dsl_Jacobian,
    logit_dsl_moment_base,
)


def dsl_general(
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    model: str = "lm",
    fe_Y: Optional[np.ndarray] = None,
    fe_X: Optional[np.ndarray] = None,
    moment_fn: Optional[callable] = None,
    jac_fn: Optional[callable] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    General DSL estimation function.

    Parameters
    ----------
    Y_orig : np.ndarray
        Original outcome
    X_orig : np.ndarray
        Original features
    Y_pred : np.ndarray
        Predicted outcome
    X_pred : np.ndarray
        Predicted features
    labeled_ind : np.ndarray
        Labeled indicator
    sample_prob_use : np.ndarray
        Sampling probability
    model : str, optional
        Model type, by default "lm"
    fe_Y : Optional[np.ndarray], optional
        Fixed effects outcome, by default None
    fe_X : Optional[np.ndarray], optional
        Fixed effects features, by default None
    moment_fn : Optional[callable], optional
        Moment function, by default None
    jac_fn : Optional[callable], optional
        Jacobian function, by default None

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        Estimated parameters and additional information
    """
    if moment_fn is None:
        if model == "lm":
            moment_fn = lm_dsl_moment_base
        elif model == "logit":
            moment_fn = logit_dsl_moment_base
        elif model == "felm":
            moment_fn = felm_dsl_moment_base
            # For fixed effects, we need to demean the data
            if fe_Y is not None and fe_X is not None:
                # Create a temporary DataFrame for demeaning
                temp_df = pd.DataFrame(
                    {
                        "id": range(len(Y_orig)),
                        "fe": fe_Y.argmax(axis=1) if fe_Y.ndim > 1 else fe_Y,
                    }
                )
                Y_orig, X_orig = demean_dsl(temp_df, Y_orig, X_orig, ["fe"])[:2]
                Y_pred, X_pred = demean_dsl(temp_df, Y_pred, X_pred, ["fe"])[:2]

    if jac_fn is None:
        if model == "lm":
            jac_fn = lm_dsl_Jacobian
        elif model == "logit":
            jac_fn = logit_dsl_Jacobian
        elif model == "felm":
            jac_fn = felm_dsl_Jacobian

    # Initial parameter estimate
    n_main_orig = X_orig.shape[1]  # Store original number of main parameters
    n_params = n_main_orig
    if model == "felm" and fe_X is not None:
        n_fe_params = fe_X.shape[1]
        n_params += n_fe_params
    else:
        n_fe_params = 0
    par_init = np.zeros(n_params)

    # Define objective function
    def objective(par):
        return dsl_general_moment(
            par,
            labeled_ind,
            sample_prob_use,
            Y_orig,
            X_orig,  # Pass standardized X
            Y_pred,
            X_pred,  # Pass standardized X
            fe_Y=fe_Y,
            fe_X=fe_X,
            model=model,
            n_main_orig=n_main_orig,  # Pass original n_main
        )

    # Optimize
    result = minimize(
        objective,
        par_init,
        method="BFGS",
        options={"maxiter": 1000, "disp": False},
    )

    # Compute standard errors
    J = dsl_general_Jacobian(
        result.x,
        labeled_ind,
        sample_prob_use,
        Y_orig,
        X_orig,  # Pass standardized X
        Y_pred,
        X_pred,  # Pass standardized X
        model=model,
        fe_Y=fe_Y,
        fe_X=fe_X,
        n_main_orig=n_main_orig,  # Pass original n_main
    )

    # Compute variance-covariance matrix
    # Need to calculate the 'meat' (Omega) using moment contributions
    # Omega = E[m_i * m_i.T]
    moments = dsl_general_moment_contributions(
        result.x,
        labeled_ind,
        sample_prob_use,
        Y_orig,
        X_orig,
        Y_pred,
        X_pred,
        fe_Y=fe_Y,
        fe_X=fe_X,
        model=model,
        n_main_orig=n_main_orig,
    )
    n_obs = moments.shape[0]
    omega = (moments.T @ moments) / n_obs

    # Sandwich variance: (J^-1) * Omega * (J^-1).T / n
    # Note: R code seems to calculate J slightly differently sometimes,
    # and the scaling might differ. This follows the standard GMM formula.
    try:
        J_inv = np.linalg.inv(J)
        vcov = (J_inv @ omega @ J_inv.T) / n_obs
    except np.linalg.LinAlgError:
        print("Jacobian is singular, using pseudo-inverse")
        J_inv = np.linalg.pinv(J)
        vcov = (J_inv @ omega @ J_inv.T) / n_obs

    # Ensure vcov is positive semi-definite for SE calculation
    vcov = (vcov + vcov.T) / 2  # Symmetrize
    min_eig = np.min(np.real(np.linalg.eigvals(vcov)))
    if min_eig < 0:
        # Add regularization if not PSD
        vcov += (-min_eig + 1e-8) * np.eye(vcov.shape[0])

    try:
        se = np.sqrt(np.diag(vcov))
    except RuntimeWarning as e:
        print(f"Warning calculating standard errors: {e}")
        se = np.full(vcov.shape[0], np.nan)

    # Predict values and calculate residuals
    predicted_values = dsl_predict_internal(
        result.x, X_orig, X_pred, model, fe_Y, fe_X, n_main_orig, n_fe_params
    )
    residuals = Y_orig - predicted_values

    # Return results
    return result.x, {
        "standard_errors": se,
        "vcov": vcov,
        "objective": result.fun,
        "convergence": result.success,
        "message": result.message,
        "iterations": result.nit,
        "jacobian": J,
        "omega": omega,
        "n_obs": n_obs,
        "n_main_params": n_main_orig,
        "n_fe_params": n_fe_params,
        "predicted_values": predicted_values,
        "residuals": residuals,
    }


def dsl_predict(X: np.ndarray, se: np.ndarray, model: str = "linear") -> np.ndarray:
    """
    Predict using DSL estimates.
    NOTE: This function might need revision. Using standard errors (se)
    as initial parameters (beta) is likely incorrect for prediction.
    It should probably use the estimated coefficients.

    Parameters
    ----------
    X : np.ndarray
        Features
    se : np.ndarray
        Standard errors from labeled data estimation
    model : str, optional
        Model type, by default "linear"

    Returns
    -------
    np.ndarray
        Predicted values
    """
    if model in ["linear", "felm"]:
        # Add small regularization for numerical stability
        reg_param = 1e-6
        # X_dot_X = X.T @ X # Unused
        # X_dot_X_inv = np.linalg.inv(X_dot_X + reg_param * np.eye(X_dot_X.shape[0]))
        # Using standard errors as beta is likely wrong for prediction
        # Should use estimated coefficients from dsl_general
        beta = se
        return X @ beta
    elif model in ["logit", "logistic"]:  # Support both logit and logistic
        # Initialize parameters with standard errors
        beta = se
        max_iter = 100
        tol = 1e-6

        for i in range(max_iter):
            # Calculate probabilities
            z = X @ beta
            p = 1 / (1 + np.exp(-z))

            # Calculate gradient and Hessian
            W = np.diag(p * (1 - p))
            hessian = X.T @ W @ X

            # Add small regularization for numerical stability
            reg_param = 1e-6
            hessian += reg_param * np.eye(hessian.shape[0])

            # Update parameters
            beta_new = beta + np.linalg.solve(
                hessian, X.T @ (p - p)
            )  # Zero gradient since we're predicting

            # Check convergence
            if np.all(np.abs(beta_new - beta) < tol):
                break
            beta = beta_new

        # Calculate predictions
        z = X @ beta
        return 1 / (1 + np.exp(-z))
    else:
        raise ValueError(f"Unknown model type: {model}")


def dsl_residuals(
    Y: np.ndarray,
    X: np.ndarray,
    par: np.ndarray,
    model: str = "lm",
) -> np.ndarray:
    """
    Compute residuals using DSL estimates.

    Parameters
    ----------
    Y : np.ndarray
        Outcomes
    X : np.ndarray
        Features
    par : np.ndarray
        Estimated parameters
    model : str, optional
        Model type, by default "lm"

    Returns
    -------
    np.ndarray
        Residuals
    """
    if model == "lm":
        return Y - X @ par
    elif model == "logit":
        return Y - 1 / (1 + np.exp(-X @ par))
    else:
        raise ValueError(f"Unknown model type: {model}")


def dsl_vcov(
    X: np.ndarray,
    par: np.ndarray,
    se: np.ndarray,
    model: str = "lm",
) -> np.ndarray:
    """
    Compute variance-covariance matrix using DSL estimates.

    Parameters
    ----------
    X : np.ndarray
        Features
    par : np.ndarray
        Estimated parameters
    se : np.ndarray
        Standard errors
    model : str, optional
        Model type, by default "lm"

    Returns
    -------
    np.ndarray
        Variance-covariance matrix
    """
    # Compute predicted values
    if model == "lm":
        pred = X @ par
    elif model == "logit":
        pred = 1 / (1 + np.exp(-X @ par))
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Compute weights
    if model == "lm":
        w = np.ones_like(pred)
    elif model == "logit":
        w = pred * (1 - pred)

    # Compute variance-covariance matrix
    X_w = X * w[:, np.newaxis]
    V = np.linalg.inv(X_w.T @ X_w)
    V = V * (se**2)[:, np.newaxis]

    return V


def dsl_general_Jacobian(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    model: str,
    fe_Y: Optional[np.ndarray] = None,
    fe_X: Optional[np.ndarray] = None,
    n_main_orig: Optional[int] = None,  # Add original n_main
) -> np.ndarray:
    """
    Calculate the average Jacobian matrix J = E[dm_i/dpar].
    Returns a (n_params x n_params) matrix.
    """
    if model == "lm":
        jac_fn = lm_dsl_Jacobian
        args = (labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, model)
    elif model == "logit":
        jac_fn = logit_dsl_Jacobian
        args = (labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, model)
    elif model == "felm":
        jac_fn = felm_dsl_Jacobian
        if fe_Y is None or fe_X is None:
            raise ValueError("fe_Y and fe_X must be provided for felm model")
        if n_main_orig is None:
            # Infer n_main_orig if not passed (shouldn't happen in normal flow)
            n_main_orig = X_orig.shape[1] - (1 if np.all(X_orig[:, 0] == 1) else 0)
        args = (
            labeled_ind,
            sample_prob_use,
            Y_orig,
            X_orig,
            Y_pred,
            X_pred,
            model,
            fe_Y,
            fe_X,
            n_main_orig,  # Pass n_main_orig
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    J = jac_fn(par, *args)
    return J


def dsl_general_moment(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    model: str,
    fe_Y: Optional[np.ndarray] = None,
    fe_X: Optional[np.ndarray] = None,
    n_main_orig: Optional[int] = None,  # Add original n_main
) -> float:
    """
    GMM Objective Function: Sum of squared average moments.
    Q(par) = m_bar(par).T @ m_bar(par)
    where m_bar(par) = (1/n) * sum(m_i(par))

    Parameters
    ----------
    par : np.ndarray
        Parameter vector
    labeled_ind : np.ndarray
        Labeled indicator
    sample_prob_use : np.ndarray
        Sampling probability
    Y_orig : np.ndarray
        Original outcome
    X_orig : np.ndarray
        Original features (standardized)
    Y_pred : np.ndarray
        Predicted outcome
    X_pred : np.ndarray
        Predicted features (standardized)
    model : str
        Model type
    fe_Y : Optional[np.ndarray], optional
        Fixed effects outcome, by default None
    fe_X : Optional[np.ndarray], optional
        Fixed effects features, by default None
    n_main_orig : Optional[int], optional
        Original number of main parameters, by default None

    Returns
    -------
    float
        Objective function value
    """
    moments = dsl_general_moment_contributions(
        par,
        labeled_ind,
        sample_prob_use,
        Y_orig,
        X_orig,
        Y_pred,
        X_pred,
        fe_Y,
        fe_X,
        model,
        n_main_orig,  # Pass through
    )
    m_bar = np.mean(moments, axis=0)
    return m_bar @ m_bar  # Sum of squared average moments


def dsl_general_moment_contributions(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    fe_Y: Optional[np.ndarray],
    fe_X: Optional[np.ndarray],
    model: str,
    n_main_orig: Optional[int],  # Add original n_main
) -> np.ndarray:
    """
    Calculate the moment contributions m_i(par) for each observation.
    Returns an (n_obs x n_params) matrix.
    """
    if model == "lm":
        moment_fn = lm_dsl_moment_base
        args = (labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred)
    elif model == "logit":
        moment_fn = logit_dsl_moment_base
        args = (labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred)
    elif model == "felm":
        moment_fn = felm_dsl_moment_base
        if fe_Y is None or fe_X is None:
            raise ValueError("fe_Y and fe_X must be provided for felm model")
        if n_main_orig is None:
            # Infer n_main_orig if not passed (shouldn't happen in normal flow)
            n_main_orig = X_orig.shape[1] - (1 if np.all(X_orig[:, 0] == 1) else 0)
        args = (
            labeled_ind,
            sample_prob_use,
            Y_orig,
            X_orig,
            Y_pred,
            X_pred,
            fe_Y,
            fe_X,
            n_main_orig,  # Pass n_main_orig
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    m = moment_fn(par, *args)
    return m


def dsl_general_moment_base_decomp(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    model: str = "lm",
    clustered: bool = False,
    cluster: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the base moment decomposition for DSL estimation.

    Parameters
    ----------
    par : np.ndarray
        Parameters at which to evaluate the moment
    labeled_ind : np.ndarray
        Labeled indicator
    sample_prob_use : np.ndarray
        Sampling probability
    Y_orig : np.ndarray
        Original outcome
    X_orig : np.ndarray
        Original features
    Y_pred : np.ndarray
        Predicted outcome
    X_pred : np.ndarray
        Predicted features
    model : str, optional
        Model type, by default "lm"
    clustered : bool, optional
        Whether to use clustered standard errors, by default False
    cluster : Optional[np.ndarray], optional
        Cluster indicator, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Main components of the moment decomposition
    """
    # Import moment functions
    from .moment import (
        felm_dsl_moment_base_decomp,
        lm_dsl_moment_base_decomp,
        logit_dsl_moment_base_decomp,
    )

    # Select appropriate moment decomposition function
    if model == "lm":
        moment_decomp_fn = lm_dsl_moment_base_decomp
    elif model == "logit":
        moment_decomp_fn = logit_dsl_moment_base_decomp
    elif model == "felm":
        moment_decomp_fn = felm_dsl_moment_base_decomp
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Compute moment decomposition
    main_1, main_23 = moment_decomp_fn(
        par,
        labeled_ind,
        sample_prob_use,
        Y_orig,
        X_orig,
        Y_pred,
        X_pred,
        clustered,
        cluster,
    )

    return main_1, main_23


def dsl_general_moment_est(
    model: str,
    formula: str,
    labeled: str,
    sample_prob: str,
    predicted_var: List[str],
    data_orig: pd.DataFrame,
    data_pred: pd.DataFrame,
    index: Optional[List[str]] = None,
    fixed_effect: Optional[str] = None,
    clustered: bool = False,
    cluster: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate DSL model using moment conditions (Original Interface - Deprecated?).
    This function seems to use a different approach than dsl_general
    and might be outdated or serve a different purpose (e.g., decomposition).

    Parameters
    ----------
    model : str
        Model type ("lm", "logit", or "felm")
    formula : str
        Formula for the model
    labeled : str
        Name of labeled indicator column
    sample_prob : str
        Name of sampling probability column
    predicted_var : List[str]
        List of predicted variable names
    data_orig : pd.DataFrame
        Original data
    data_pred : pd.DataFrame
        Predicted data
    index : Optional[List[str]], optional
        List of index variables for fixed effects, by default None
    fixed_effect : Optional[str], optional
        Type of fixed effects, by default None
    clustered : bool, optional
        Whether to use clustered standard errors, by default False
    cluster : Optional[str], optional
        Name of cluster variable, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing estimation results
    """
    # Import patsy for formula parsing
    import patsy

    # Parse formula
    y_orig, X_orig = patsy.dmatrices(formula, data_orig, return_type="dataframe")
    y_pred, X_pred = patsy.dmatrices(formula, data_pred, return_type="dataframe")

    # Get labeled indicator and sampling probability
    labeled_ind = data_orig[labeled].values
    sample_prob_use = data_orig[sample_prob].values

    # Get fixed effects if needed
    fe_Y = None
    fe_X = None
    if fixed_effect is not None and index is not None:
        fe_Y = data_orig[index].values
        fe_X = data_pred[index].values

    # Get cluster variable if needed
    cluster_var = None
    if clustered and cluster is not None:
        cluster_var = data_orig[cluster].values

    # Initial parameter estimate
    par_init = np.zeros(X_orig.shape[1])

    # Estimate model
    par_est, info = dsl_general(
        Y_orig=y_orig.values.flatten(),
        X_orig=X_orig.values,
        Y_pred=y_pred.values.flatten(),
        X_pred=X_pred.values,
        labeled_ind=labeled_ind,
        sample_prob_use=sample_prob_use,
        model=model,
        fe_Y=fe_Y,
        fe_X=fe_X,
    )

    # Compute moment decomposition
    main_1, main_23 = dsl_general_moment_base_decomp(
        par=par_est,
        labeled_ind=labeled_ind,
        sample_prob_use=sample_prob_use,
        Y_orig=y_orig.values.flatten(),
        X_orig=X_orig.values,
        Y_pred=y_pred.values.flatten(),
        X_pred=X_pred.values,
        model=model,
        clustered=clustered,
        cluster=cluster_var,
    )

    # Compute Jacobian
    J = dsl_general_Jacobian(
        par=par_est,
        labeled_ind=labeled_ind,
        sample_prob_use=sample_prob_use,
        Y_orig=y_orig.values.flatten(),
        X_orig=X_orig.values,
        Y_pred=y_pred.values.flatten(),
        X_pred=X_pred.values,
        model=model,
        fe_Y=fe_Y,
        fe_X=fe_X,
    )

    # Compute variance-covariance matrices
    # Note: J here is likely the (n x k) Jacobian from the moment function,
    # not the (k x k) average Jacobian used in dsl_general.
    D = np.linalg.inv(J.T @ J)
    Meat = main_1 + main_23
    vcov = D @ Meat @ D
    vcov0 = D @ main_1 @ D

    # Get column names for output
    coef_names = X_orig.columns

    # Return results
    return {
        "coefficients": pd.Series(par_est, index=coef_names),
        "standard_errors": pd.Series(np.sqrt(np.diag(vcov)), index=coef_names),
        "vcov": pd.DataFrame(vcov, index=coef_names, columns=coef_names),
        "Meat": pd.DataFrame(Meat, index=coef_names, columns=coef_names),
        "Meat_decomp": {
            "main_1": pd.DataFrame(main_1, index=coef_names, columns=coef_names),
            "main_23": pd.DataFrame(main_23, index=coef_names, columns=coef_names),
        },
        "J": pd.DataFrame(J),  # J here might be (n x k)
        "D": pd.DataFrame(D, index=coef_names, columns=coef_names),
        "vcov0": pd.DataFrame(vcov0, index=coef_names, columns=coef_names),
    }


def stable_inverse(A: np.ndarray) -> np.ndarray:
    """
    Compute stable matrix inverse using QR decomposition.
    """
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R.T @ R, R.T @ Q.T)


def compute_sandwich_var(J, m, n_obs):
    """
    Compute sandwich variance estimator (scaled version).
    Assumes J is (k,k) average Jacobian and m is (n,k) moments.
    vcov = (J^-1) @ Meat @ (J^-1) / n
    Meat = (1/n) * sum(m_i @ m_i.T) (should be k x k)
    """
    n_features = J.shape[0]

    # Compute J inverse (Bread)
    try:
        bread = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        print("Warning: Jacobian is singular, using pseudo-inverse.")
        bread = np.linalg.pinv(J)

    # Compute Meat = (1/n) * sum(m_i @ m_i.T)
    # m has shape (n, k)
    if m.shape[0] != n_obs or m.shape[1] != n_features:
        raise ValueError(
            f"Moment shape mismatch in vcov: Expected ({n_obs},{n_features}), "
            f"got {m.shape}"
        )

    meat = (m.T @ m) / n_obs  # Efficient calculation: (k,n) @ (n,k) -> (k,k)

    # Compute variance-covariance matrix (scaled)
    vcov_scaled = (bread @ meat @ bread.T) / n_obs

    return vcov_scaled


def dsl_predict_internal(
    par: np.ndarray,
    X_orig_std: np.ndarray,  # Standardized X for original data
    X_pred_std: np.ndarray,  # Standardized X for prediction data
    model: str,
    fe_Y: Optional[np.ndarray],
    fe_X: Optional[np.ndarray],
    n_main_orig: int,
    n_fe_params: int,
) -> np.ndarray:
    """
    Internal prediction function used within dsl_general.
    Uses estimated parameters and standardized data.
    """
    if model in ["lm", "linear"]:
        # Use the original (standardized) data for prediction
        # Assuming par includes intercept if standardized with intercept
        return X_orig_std @ par
    elif model == "logit":
        z = X_orig_std @ par
        return 1 / (1 + np.exp(-z))
    elif model == "felm":
        par_main = par[:n_main_orig]  # Use original n_main
        par_fe = par[n_main_orig:]
        if len(par_fe) != n_fe_params:
            raise ValueError(
                f"Fixed effect parameter mismatch: expected {n_fe_params}, got {len(par_fe)}"
            )
        if fe_Y is None or fe_X is None:
            raise ValueError("fe_Y and fe_X required for felm prediction")

        # Calculate fixed effect component
        fe_use = fe_X @ par_fe
        # Prediction using standardized X and estimated main + fixed effects
        return X_orig_std @ par_main + fe_use.flatten()
    else:
        raise ValueError(f"Unknown model type for prediction: {model}")


def standardize_data(
    X: np.ndarray,
    ref_X: np.ndarray,
    intercept: bool = True,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize data based on reference data means and standard deviations.

    Parameters
    ----------
    X : np.ndarray
        Data to standardize.
    ref_X : np.ndarray
        Reference data for calculating means and std devs.
    intercept : bool, optional
        Whether to add an intercept term, by default True.
    epsilon : float, optional
        Small value to add to std dev denominator for numerical stability.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Standardized data, means, standard deviations.
    """
    if intercept:
        X_adj = sm.add_constant(X, has_constant="add")
        ref_X_adj = sm.add_constant(ref_X, has_constant="add")
    else:
        X_adj = X.copy()
        ref_X_adj = ref_X.copy()

    means = np.mean(ref_X_adj, axis=0)
    stds = np.std(ref_X_adj, axis=0)

    # Don't scale the constant term if it exists
    if intercept:
        stds[0] = 1.0  # Set std dev of constant to 1
        means[0] = 0.0  # Set mean of constant to 0 for centering

    # Add epsilon to avoid division by zero
    stds_reg = stds + epsilon

    # Standardize (center and scale if intercept, just scale otherwise)
    if intercept:
        X_standardized = (X_adj - means) / stds_reg
    else:
        X_standardized = X_adj / stds_reg

    return X_standardized, means, stds


def rescale_params(
    par_scaled: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    vcov_scaled: np.ndarray,
    intercept: bool = True,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rescale standardized parameters and their variance-covariance matrix
    back to the original scale.

    Parameters
    ----------
    par_scaled : np.ndarray
        Estimated parameters on the standardized scale.
    means : np.ndarray
        Means used for standardization.
    stds : np.ndarray
        Standard deviations used for standardization.
    vcov_scaled : np.ndarray
        Variance-covariance matrix for standardized parameters.
    intercept : bool, optional
        Whether an intercept was included during standardization.
    epsilon : float, optional
        Small value added to stds during standardization.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Rescaled parameters, Rescaled variance-covariance matrix.
    """
    n_params = len(par_scaled)
    stds_reg = stds + epsilon

    # Create the rescaling transformation matrix (Jacobian D)
    D = np.zeros((n_params, n_params))

    if intercept:
        if len(means) != n_params or len(stds_reg) != n_params:
            raise ValueError(
                f"Shape mismatch: par({n_params}), means({len(means)}), stds({len(stds_reg)})"
            )
        # First parameter is the intercept
        D[0, 0] = 1.0
        # Adjust intercept based on other scaled parameters and means/stds
        D[0, 1:] = -means[1:] / stds_reg[1:]
        # Scale other parameters
        np.fill_diagonal(D[1:, 1:], 1.0 / stds_reg[1:])
    else:
        if len(stds_reg) != n_params:
            raise ValueError(f"Shape mismatch: par({n_params}), stds({len(stds_reg)})")
        # Only scaling, D is diagonal
        np.fill_diagonal(D, 1.0 / stds_reg)

    # Rescale parameters: par_orig = D @ par_scaled
    par_orig = D @ par_scaled

    # Rescale variance-covariance matrix: V_orig = D @ V_scaled @ D.T
    vcov_orig = D @ vcov_scaled @ D.T

    return par_orig, vcov_orig
