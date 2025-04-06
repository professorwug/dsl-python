# Mathematical Implementation Summary: Python DSL Implementation

This document summarizes the core mathematical steps involved in the Python `dsl` implementation, primarily based on the functions within `dsl/helpers/dsl_general.py` and `dsl/helpers/moment.py`.

## I. Estimation (`dsl_general`)

The main estimation workflow follows these steps:

1.  **Data Standardization:**
    *   Predictor variables (`X_orig`, `X_pred`) are standardized based on the means (`mean_X`) and standard deviations (`sd_X`) derived from `X_pred`.
    *   If an intercept is present: `X_use = (X - mean_X) / sd_X` (Intercept column remains 1).
    *   If no intercept: `X_use = X / sd_X`.
    *   Means and standard deviations are stored for later rescaling.

2.  **Initial Parameter Estimation:**
    *   Before the main optimization, an initial estimate (`par_init`) is obtained using `statsmodels` on the *labeled* portion of the *standardized* data (`X_orig_labeled_use`, `Y_orig_labeled`).
    *   `sm.Logit` (with BFGS solver) is used for `model='logit'`. `sm.OLS` is used for `model='lm'`.
    *   This provides a better starting point than zeros for the main optimization.

3.  **Point Estimation (GMM):**
    *   The core parameters are estimated by minimizing a GMM objective function using `scipy.optimize.minimize(method="BFGS")`.
    *   **Objective Function (`objective` nested function):** Minimizes the sum of squared *average* moment conditions:
        \[ Q(\beta) = \mathbf{\bar{m}}(\beta)^T \mathbf{\bar{m}}(\beta) = \sum_{j=1}^k \left( \frac{1}{n} \sum_{i=1}^n m_{ij}(\beta) \right)^2 \]
        where \( m_{ij}(\beta) \) are the doubly robust moment contributions calculated by the `moment_fn`.
    *   **Gradient (`gradient` nested function):** An analytical gradient is provided to the optimizer:
        \[ \frac{\partial Q(\beta)}{\partial \beta} = 2 \mathbf{J}(\beta)^T \mathbf{\bar{m}}(\beta) \]
        where \( \mathbf{J}(\beta) \) is the average Jacobian and \( \mathbf{\bar{m}}(\beta) \) is the vector of average moments.

4.  **Moment Calculation (`lm_dsl_moment_base`, `logit_dsl_moment_base`):**
    *   Calculates the \( n \times k \) moment matrix \( \mathbf{M} \) using the same doubly robust structure as R:
        \[ \mathbf{m}_{\text{dr}, i}(\beta) = \mathbf{m}_{\text{pred}, i}(\beta) + \frac{I_i}{\pi_i} (\mathbf{m}_{\text{orig}, i}(\beta) - \mathbf{m}_{\text{pred}, i}(\beta)) \]
    *   **LM:** \( \mathbf{m}_{\text{orig/pred}, i}(\beta) = \mathbf{x}_i (y_i^{\text{orig/pred}} - \mathbf{x}_i^T \beta) \)
    *   **Logit:** \( \mathbf{m}_{\text{orig/pred}, i}(\beta) = \mathbf{x}_i (y_i^{\text{orig/pred}} - p_i(\beta)) \)
    *   *Difference from R:* Explicit flattening (`Y.flatten()`) is used when calculating residuals to ensure correct NumPy broadcasting.

## II. Variance Estimation (`dsl_general` post-optimization, `compute_sandwich_var`)

Uses the standard GMM sandwich variance estimator formula, calculated *after* optimization using the optimal scaled parameters \( \hat{\beta}_{\text{scaled}} \).

\[ \text{Var}(\hat{\beta}_{\text{scaled}}) = \frac{1}{n} (\mathbf{J}^{-1}) \mathbf{\Omega} (\mathbf{J}^{-T}) \]

1.  **Jacobian (`lm_dsl_Jacobian`, `logit_dsl_Jacobian`):**
    *   Calculates the \( k \times k \) average Jacobian matrix \( \mathbf{J} \) evaluated at \( \hat{\beta}_{\text{scaled}} \).
        \[ \mathbf{J} = E \left[ \frac{\partial \mathbf{m}_i(\beta)}{\partial \beta^T} \right] \approx \frac{1}{n} \sum_{i=1}^n \frac{\partial \mathbf{m}_i(\hat{\beta}_{\text{scaled}})}{\partial \beta^T} \]
    *   **LM:** \( \mathbf{J}_{lm} \approx -\frac{1}{n} \mathbf{X}_{\text{use}}^T \mathbf{X}_{\text{use}} \) (using standardized X).
    *   **Logit:** \( \mathbf{J}_{logit} \approx -\frac{1}{n} \mathbf{X}_{\text{use}}^T \mathbf{W}_{\text{dr}} \mathbf{X}_{\text{use}} \), where \( \mathbf{W}_{\text{dr}} \) uses doubly robust weights \( w_{\text{dr}, i} = w_{\text{pred}, i} + \frac{I_i}{\pi_i} (w_{\text{orig}, i} - w_{\text{pred}, i}) \) with \( w_i = p_i(1-p_i) \), evaluated at \( \hat{\beta}_{\text{scaled}} \).

2.  **Meat Matrix (`compute_sandwich_var`):**
    *   Estimates \( \mathbf{\Omega} = E[\mathbf{m}_i(\beta) \mathbf{m}_i(\beta)^T] \) using moments \( \mathbf{M} \) calculated at \( \hat{\beta}_{\text{scaled}} \).
    *   \( \hat{\mathbf{\Omega}} = \frac{1}{n} \sum_{i=1}^n \mathbf{m}_{\text{dr}, i}(\hat{\beta}_{\text{scaled}}) \mathbf{m}_{\text{dr}, i}(\hat{\beta}_{\text{scaled}})^T = \frac{1}{n} \mathbf{M}^T \mathbf{M} \) (calculated efficiently as `(m.T @ m) / n_obs`).

3.  **Bread (`compute_sandwich_var`):**
    *   Inverse Jacobian: \( \mathbf{J}^{-1} \). Calculated using `np.linalg.inv(J)` with fallback to `pinv` for singular matrices.

4.  **Scaled Variance (`compute_sandwich_var`):**
    *   \( \mathbf{V}_{\text{scaled}} = \frac{1}{n} (\text{bread}) \hat{\mathbf{\Omega}} (\text{bread}^T) \)

## III. Rescaling (`dsl_general`)

The estimated coefficients and their variance-covariance matrix are rescaled back to the original variable scale.

1.  **Rescaling Coefficients:**
    *   Calculates \( \hat{\beta}_{\text{orig}} \) from \( \hat{\beta}_{\text{scaled}} \) by reversing the standardization.
    *   **With Intercept:**
        *   \( \beta_{j, \text{orig}} = \beta_{j, \text{scaled}} / \sigma_j \) for \( j \ge 1 \)
        *   \( \beta_{0, \text{orig}} = \beta_{0, \text{scaled}} - \sum_{j=1}^{k-1} \beta_{j, \text{orig}} \mu_j \)
    *   **Without Intercept:** \( \beta_{j, \text{orig}} = \beta_{j, \text{scaled}} / \sigma_j \) for all \( j \).

2.  **Rescaling Jacobian (`D_rescale_jacobian`):**
    *   Calculates \( \mathbf{D} = \frac{\partial \beta_{\text{orig}}}{\partial \beta_{\text{scaled}}} \) using the same formulas as R based on \( \mu_j, \sigma_j \).

3.  **Final Variance:**
    *   \( \mathbf{V}_{\text{orig}} = \mathbf{D} \mathbf{V}_{\text{scaled}} \mathbf{D}^T \)

## IV. Standard Errors and P-Values (`dsl`, potentially formatters/summaries)

*   Standard errors are \( \sqrt{\text{diag}(\mathbf{V}_{\text{orig}})} \).
*   P-values (calculated in helper/test scripts like `compare_panchen.py`) use the Wald statistic (Estimate / Std.Error) compared against a Student's t-distribution (`scipy.stats.t`) with \( n-k \) degrees of freedom.
*   *Difference from R:* R typically uses the standard normal (z) distribution for p-values in logit/GMM contexts. However, for large \( n \), the t-distribution is numerically very close to the normal distribution, making the practical difference negligible.

## Summary of Key Differences from R

*   **Optimization:** Uses `scipy.optimize.minimize` (BFGS) vs. R's `stats::optim`. Minor implementation differences (e.g., line search, convergence checks) can lead to slightly different results.
*   **Initialization:** Uses `statsmodels` vs. R's base `glm` for initial parameter guesses.
*   **P-Value Distribution:** Test/comparison scripts use t-distribution vs. R's standard z-distribution for logit p-values (negligible impact for large n).
*   **Numerical Libraries:** Relies on NumPy/SciPy vs. R's base math/matrix libraries, potentially using different BLAS/LAPACK versions, allowing for tiny floating-point variations.
*   **Formula Parsing:** Uses `patsy` vs. R's native formula handling for design matrix creation.
*   **Numerical Stability/Edge Cases:** Minor differences in handling edge cases (e.g., division by zero) or numerical stability additions.
*   **Code Structure:** Python uses explicit GMM objective/gradient for the main optimization, while R's structure might differ slightly internally. 