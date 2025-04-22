"""
Fixed effects implementation for DSL framework.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def demean_dsl(
    data_base: pd.DataFrame,
    adj_Y: np.ndarray,
    adj_X: np.ndarray,
    index: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Demean data for fixed effects regression.

    Parameters
    ----------
    data_base : pd.DataFrame
        Base data frame
    adj_Y : np.ndarray
        Adjusted outcome
    adj_X : np.ndarray
        Adjusted features
    index : List[str]
        List of fixed effect variables

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Demeaned outcome, features, and data frame
    """
    # Create fixed effect matrix
    fixed_effect_use = pd.get_dummies(data_base[index])

    # Add small regularization for numerical stability
    reg_param = 1e-6
    fixed_effect_matrix = fixed_effect_use.values
    FTF = fixed_effect_matrix.T @ fixed_effect_matrix
    FTF_reg = FTF + reg_param * np.eye(FTF.shape[0])

    # Demean outcome and features
    fixed_effect_Y = (
        fixed_effect_matrix @ np.linalg.inv(FTF_reg) @ fixed_effect_matrix.T @ adj_Y
    )

    fixed_effect_X = (
        fixed_effect_matrix @ np.linalg.inv(FTF_reg) @ fixed_effect_matrix.T @ adj_X
    )

    # Create adjusted data frame
    adj_data = pd.DataFrame(
        np.column_stack([data_base[["id"] + index], adj_X]),
        columns=["id"] + index + [f"x{i+1}" for i in range(adj_X.shape[1])],
    )

    return fixed_effect_Y, fixed_effect_X, adj_data


def felm_dsl_moment_base(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    fe_Y: np.ndarray,
    fe_X: np.ndarray,
    n_main_orig: int,
) -> np.ndarray:
    """
    Base moment function for fixed effects regression.

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
        Original features
    Y_pred : np.ndarray
        Predicted outcome
    X_pred : np.ndarray
        Predicted features
    fe_Y : np.ndarray
        Fixed effects outcome
    fe_X : np.ndarray
        Fixed effects features
    n_main_orig : int
        Original number of main effects

    Returns
    -------
    np.ndarray
        Moment function value
    """
    # Split parameters into main effects and fixed effects using n_main_orig
    par_main = par[:n_main_orig]
    par_fe = par[n_main_orig:]

    # Compute fixed effects
    # Ensure par_fe is 2D for matrix multiplication
    par_fe = par_fe.reshape(-1, 1)
    # Compute fixed effects contribution - this will be a column vector
    fe_use = fe_X @ par_fe

    # Ensure all arrays are properly shaped
    Y_orig = Y_orig.reshape(-1)  # Flatten to 1D
    Y_pred = Y_pred.reshape(-1)  # Flatten to 1D
    fe_use = fe_use.reshape(-1)  # Flatten to 1D
    labeled_ind = labeled_ind.reshape(-1)  # Flatten to 1D
    sample_prob_use = sample_prob_use.reshape(-1)  # Flatten to 1D

    # Calculate residuals with fixed effects
    residuals_orig = Y_orig - X_orig @ par_main - fe_use
    residuals_pred = Y_pred - X_pred @ par_main - fe_use

    # Original moment with fixed effects - use broadcasting
    m_orig = np.column_stack(
        [
            X_orig * residuals_orig.reshape(-1, 1),  # Main effects
            fe_X * residuals_orig.reshape(-1, 1),  # Fixed effects
        ]
    )
    m_orig = m_orig * labeled_ind.reshape(-1, 1)  # Reshape labeled_ind for broadcasting

    # Predicted moment with fixed effects - use broadcasting
    m_pred = np.column_stack(
        [
            X_pred * residuals_pred.reshape(-1, 1),  # Main effects
            fe_X * residuals_pred.reshape(-1, 1),  # Fixed effects
        ]
    )

    # Combined moment
    weights = (labeled_ind / sample_prob_use).reshape(
        -1, 1
    )  # Reshape weights for broadcasting
    m_dr = m_pred + (m_orig - m_pred) * weights

    # Zero out unlabeled observations
    m_dr = m_dr * labeled_ind.reshape(-1, 1)  # Reshape labeled_ind for broadcasting

    return m_dr


def felm_dsl_Jacobian(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    model: str,
    fe_Y: np.ndarray,
    fe_X: np.ndarray,
    n_main_orig: int,
) -> np.ndarray:
    """
    Jacobian for fixed effects regression.

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
        Original features
    Y_pred : np.ndarray
        Predicted outcome
    X_pred : np.ndarray
        Predicted features
    model : str
        Model type
    fe_Y : np.ndarray
        Fixed effects outcome
    fe_X : np.ndarray
        Fixed effects features
    n_main_orig : int
        Original number of main effects

    Returns
    -------
    np.ndarray
        Jacobian matrix
    """
    # Split parameters into main effects and fixed effects using n_main_orig
    n_fe = fe_X.shape[1]  # Number of fixed effects parameters

    # Original moment with fixed effects
    X_o = X_orig.copy()
    X_o = X_o * labeled_ind.reshape(-1, 1)

    # For fixed effects, use sparse matrices
    X_o = csr_matrix(X_o)
    X_p = csr_matrix(X_pred)
    fe_X_sparse = csr_matrix(fe_X)

    # Compute diagonal matrices
    d1 = np.diag(labeled_ind / sample_prob_use)
    d2 = np.diag(1 - labeled_ind / sample_prob_use)

    # Compute Jacobian for main effects
    J_main = (X_o.T @ d1 @ X_o + X_p.T @ d2 @ X_p) / len(X_orig)
    # Convert to dense array if sparse
    if hasattr(J_main, "toarray"):
        J_main = J_main.toarray()

    # Compute Jacobian for fixed effects
    J_fe = fe_X_sparse.T @ fe_X_sparse / len(X_orig)
    # Convert to dense array if sparse
    if hasattr(J_fe, "toarray"):
        J_fe = J_fe.toarray()

    # Add small regularization to avoid singularity
    reg_param = 1e-6
    J_main = J_main + reg_param * np.eye(n_main_orig)
    J_fe = J_fe + reg_param * np.eye(n_fe)

    # Combine Jacobians
    J = np.block(
        [[J_main, np.zeros((n_main_orig, n_fe))], [np.zeros((n_fe, n_main_orig)), J_fe]]
    )

    return J
