# Mathematical Implementation Summary: Python DSL Package

This document summarizes the core mathematical steps involved in the Python `dsl` package, primarily based on the functions within `dsl/helpers/dsl_general.py`, `dsl/helpers/moment.py`, and `dsl/helpers/fixed_effects.py`.

## I. Estimation (`dsl_general`)

The main estimation workflow follows these steps:

1.  **Data Standardization (`standardize_data`):** (Optional, currently not the default)
    *   Predictor variables (X) *can* be standardized based on reference means and standard deviations.
    *   If standardized *with* intercept: `X_std = (sm.add_constant(X) - means) / (stds + epsilon)`
    *   If standardized *without* intercept: `X_std = X / (stds + epsilon)`
    *   Means and standard deviations are stored for potential rescaling (`rescale_params`). *Note: Current primary workflow in `dsl.dsl` does not apply this standardization by default.*

2.  **Point Estimation (GMM):**
    *   The core parameters \( \beta \) (potentially including main effects \( \beta_{\text{main}} \) and fixed effects \( \beta_{\text{fe}} \)) are estimated by minimizing a GMM objective function using `scipy.optimize.minimize` (BFGS method).
    *   **Objective Function (`objective` within `dsl_general`, calls `dsl_general_moment`):** The function minimizes the sum of squared *average* moment conditions:
        \[ Q(\beta) = \mathbf{\bar{m}}(\beta)^T \mathbf{\bar{m}}(\beta) \]
        where \( \mathbf{\bar{m}}(\beta) = \frac{1}{n} \sum_{i=1}^n \mathbf{m}_{\text{dr}, i}(\beta) \), and \( \mathbf{m}_{\text{dr}, i}(\beta) \) is the vector of doubly robust moment contributions for observation \( i \).
    *   **Gradient:** BFGS uses the objective function values to approximate the gradient numerically.

3.  **Moment Calculation (`dsl_general_moment_contributions`, calls specific moment functions like `lm_dsl_moment_base`, `logit_dsl_moment_base`, `felm_dsl_moment_base`):**
    *   Calculates the \( n \times k \) matrix \( \mathbf{M} \) where each row \( \mathbf{m}_{\text{dr}, i}(\beta) \) is the moment contribution vector for observation \( i \).
    *   Uses the doubly robust (DR) moment structure:
        \[ \mathbf{m}_{\text{dr}, i}(\beta) = \mathbf{m}_{\text{pred}, i}(\beta) + \frac{I_i}{\pi_i} (\mathbf{m}_{\text{orig}, i}(\beta) - \mathbf{m}_{\text{pred}, i}(\beta)) \]
        where \( I_i \) is the labeled indicator (1 if labeled, 0 otherwise) and \( \pi_i \) is the sampling probability.
    *   **Linear Model (LM - `lm_dsl_moment_base`):**
        *   Residuals: \( \epsilon_i(\beta) = y_i - \mathbf{x}_i^T \beta \)
        *   \( \mathbf{m}_{\text{orig}, i}(\beta) = \mathbf{x}_i \epsilon_{\text{orig}, i}(\beta) \) (set to 0 if \( I_i=0 \))
        *   \( \mathbf{m}_{\text{pred}, i}(\beta) = \mathbf{x}_i \epsilon_{\text{pred}, i}(\beta) \) (using \( \hat{y}_i \) in residuals)
    *   **Logistic Model (Logit - `logit_dsl_moment_base`):**
        *   Probability: \( p_i(\beta) = \frac{1}{1 + e^{-\mathbf{x}_i^T \beta}} \)
        *   Residuals: \( \epsilon_i(\beta) = y_i - p_i(\beta) \)
        *   \( \mathbf{m}_{\text{orig}, i}(\beta) = \mathbf{x}_i \epsilon_{\text{orig}, i}(\beta) \) (set to 0 if \( I_i=0 \))
        *   \( \mathbf{m}_{\text{pred}, i}(\beta) = \mathbf{x}_i \epsilon_{\text{pred}, i}(\beta) \) (using \( \hat{y}_i \) in residuals)
    *   **Fixed Effects Model (FELM - `felm_dsl_moment_base`):**
        *   Parameters are split: \( \beta = [\beta_{\text{main}}, \beta_{\text{fe}}]^T \)
        *   Fixed effect component for observation \( i \): \( \text{fe}_{\text{use}, i} = \mathbf{fe}_{\text{X}, i}^T \beta_{\text{fe}} \) (Note: `fe_Y` provided to the function is *not* used in the moment calculation itself, only `fe_X`.)
        *   Residuals: \( \epsilon_i(\beta) = y_i - (\mathbf{x}_{\text{main}, i}^T \beta_{\text{main}} + \text{fe}_{\text{use}, i}) \)
        *   Moment contributions are calculated for *all* parameters (main and fixed effects):
            *   \( \mathbf{m}_{\text{orig}, i}(\beta) = [\mathbf{x}_{\text{main}, i}, \mathbf{fe}_{\text{X}, i}] \epsilon_{\text{orig}, i}(\beta) \) (set to 0 if \( I_i=0 \))
            *   \( \mathbf{m}_{\text{pred}, i}(\beta) = [\mathbf{x}_{\text{main}, i}, \mathbf{fe}_{\text{X}, i}] \epsilon_{\text{pred}, i}(\beta) \) (using \( \hat{y}_i \) in residuals)

## II. Variance Estimation (`dsl_general` post-optimization)

Uses the standard GMM sandwich variance estimator formula:

\[ \text{Var}(\hat{\beta}) = \frac{1}{n} (\mathbf{J}^{-1}) \mathbf{\Omega} (\mathbf{J}^{-T}) \]

where \( \hat{\beta} \) are the estimated parameters from the optimization.

1.  **Jacobian (`dsl_general_Jacobian`, calls specific Jacobian functions like `lm_dsl_Jacobian`, `logit_dsl_Jacobian`, `felm_dsl_Jacobian`):**
    *   Calculates the \( k \times k \) average Jacobian matrix \( \mathbf{J} \) evaluated at the estimated parameters \( \hat{\beta} \).
    *   \( \mathbf{J} = E \left[ \frac{\partial \mathbf{m}_i(\beta)}{\partial \beta^T} \right] \approx \frac{1}{n} \sum_{i=1}^n \frac{\partial \mathbf{m}_{\text{dr}, i}(\hat{\beta})}{\partial \beta^T} \)
    *   **LM (`lm_dsl_Jacobian`):** Uses sparse matrices and diagonal weight matrices \( \mathbf{D}_1 = \text{diag}(I_i / \pi_i) \) and \( \mathbf{D}_2 = \text{diag}(1 - I_i / \pi_i) \).
        \[ \mathbf{J}_{lm} \approx \frac{1}{n} (\mathbf{X}_{\text{orig}}^T \mathbf{D}_1 \mathbf{X}_{\text{orig}} + \mathbf{X}_{\text{pred}}^T \mathbf{D}_2 \mathbf{X}_{\text{pred}}) \]
        *Note: The sign differs from the typical OLS Jacobian \(-E[X^TX]\) due to the moment definition.*
    *   **Logit (`logit_dsl_Jacobian`):** Calculates doubly robust weights \( w_{\text{dr}, i} = w_{\text{pred}, i} + \frac{I_i}{\pi_i} (w_{\text{orig}, i} - w_{\text{pred}, i}) \) with \( w_i = p_i(1-p_i) \).
        \[ \mathbf{J}_{logit} \approx -\frac{1}{n} \mathbf{X}_{\text{pred}}^T \mathbf{W}_{\text{dr}} \mathbf{X}_{\text{pred}} \]
        where \( \mathbf{W}_{\text{dr}} \) is the diagonal matrix of \( w_{\text{dr}, i} \).
    *   **FELM (`felm_dsl_Jacobian`):** Constructs a block matrix Jacobian.
        *   Calculates \( \mathbf{J}_{\text{main}} \) similar to the LM Jacobian using \( \mathbf{X}_{\text{main}} \).
        *   Calculates \( \mathbf{J}_{\text{fe}} = \frac{1}{n} \mathbf{FE}_{\text{X}}^T \mathbf{FE}_{\text{X}} \).
        *   Combines into a block matrix:
            \[ \mathbf{J}_{\text{felm}} = \begin{pmatrix} \mathbf{J}_{\text{main}} & \mathbf{0} \\ \mathbf{0} & \mathbf{J}_{\text{fe}} \end{pmatrix} \]
        *Small regularization is added to the diagonal blocks for stability.*

2.  **Meat Matrix (`dsl_general` using `dsl_general_moment_contributions`):**
    *   Estimates the \( k \times k \) variance of the moment conditions \( \mathbf{\Omega} = E[\mathbf{m}_{\text{dr}, i}(\beta) \mathbf{m}_{\text{dr}, i}(\beta)^T] \).
    *   Calculated directly from the moment contributions matrix \( \mathbf{M} \) (output of `dsl_general_moment_contributions`):
        \[ \hat{\mathbf{\Omega}} = \frac{1}{n} \sum_{i=1}^n \mathbf{m}_{\text{dr}, i}(\hat{\beta}) \mathbf{m}_{\text{dr}, i}(\hat{\beta})^T = \frac{1}{n} \mathbf{M}^T \mathbf{M} \]

3.  **Bread:**
    *   The "bread" is the inverse Jacobian: \( \mathbf{J}^{-1} \). Computed using `np.linalg.inv(J)`, with a fallback to `np.linalg.pinv(J)` if `J` is singular.

4.  **Final Variance:**
    *   The final variance-covariance matrix is computed:
        \[ \mathbf{V} = \frac{1}{n} (\mathbf{J}^{-1}) \hat{\mathbf{\Omega}} (\mathbf{J}^{-T}) \]
    *   The matrix is symmetrized and adjusted slightly if not positive semi-definite to ensure valid standard errors.

## III. Rescaling (Not currently used by default)

*   If standardization *were* applied, the `rescale_params` function would be used to transform \( \hat{\beta}_{\text{scaled}} \) and \( \mathbf{V}_{\text{scaled}} \) back to the original scale using the transformation Jacobian \( \mathbf{D} \).
*   \( \hat{\beta}_{\text{orig}} = \mathbf{D} \hat{\beta}_{\text{scaled}} \)
*   \( \mathbf{V}_{\text{orig}} = \mathbf{D} \mathbf{V}_{\text{scaled}} \mathbf{D}^T \)

## IV. Fixed Effects Summary (`felm` model)

*   **Input:** Requires `fe_X` (n x p matrix of fixed effect dummies/indicators) be passed to `dsl`. `fe_Y` is also required but *not* used in the core moment/Jacobian calculations, only potentially during initial demeaning if implemented that way (currently not the default).
*   **Estimation:** Parameters are estimated jointly \( \beta = [\beta_{\text{main}}, \beta_{\text{fe}}] \).
*   **Moments:** Include terms for both main effects and fixed effects, derived from residuals \( \epsilon_i = y_i - (\mathbf{x}_{\text{main}, i}^T \beta_{\text{main}} + \mathbf{fe}_{\text{X}, i}^T \beta_{\text{fe}}) \).
*   **Jacobian:** Block diagonal structure separating main effects and fixed effects.
*   **Prediction (`dsl_predict_internal`):** \( \hat{y}_i = \mathbf{x}_{\text{main}, i}^T \hat{\beta}_{\text{main}} + \mathbf{fe}_{\text{X}, i}^T \hat{\beta}_{\text{fe}} \).
*   **Demeaning (`demean_dsl`):** A helper function exists for demeaning data based on index variables, but it's *not* currently used within the primary `dsl_general` workflow for FELM. The fixed effects are estimated directly via the extended parameter vector.

## V. Standard Errors and P-Values (`dsl` result formatting)

*   Standard errors are the square root of the diagonal elements of the final \( \mathbf{V} \).
*   P-values are calculated using a z-test (comparing `Estimate / Std.Error` to a standard normal distribution `scipy.stats.norm.cdf`), consistent with asymptotic theory for GMM.

## VI. Comparison with R (PanChen Dataset Example)

*   A comparison using the PanChen dataset (logistic model) was performed between this Python implementation and target R results (`tests/PanChen_test/target_r_output_panchen.txt`).
*   **Results:** The estimated standard errors were found to be very closely aligned between the two implementations. Coefficients were generally similar, though a notable difference was observed for the `countyWrong` predictor. This suggests a high degree of consistency overall, but potentially minor discrepancies in specific coefficient estimates likely due to subtle differences in optimization or numerical precision.