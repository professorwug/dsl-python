#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare DSL implementation in Python with R implementation using PanChen
dataset.
"""

import logging
import sys

import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy import stats

# DSL imports
from dsl_kit import dsl

# from pathlib import Path # Unused


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_panchen_data(file_path="tests/PanChen_test/PanChen.parquet"):
    """Load PanChen dataset from R data file.

    Args:
        rdata_file (str): Path to R data file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_parquet(file_path)
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Columns: {data.columns.tolist()}")
        # Rename columns for consistency (example: replace spaces, etc.)
        data.columns = [col.replace(" ", "_") for col in data.columns]
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def prepare_data_for_dsl(data):
    """Prepare PanChen data for DSL estimation to match R output.

    The PanChen dataset has columns:
    - SendOrNot: outcome variable (y)
    - countyWrong, prefecWrong, connect2b, prevalence, regionj, groupIssue:
      predictors
    - pred_countyWrong: prediction
    """
    logger.info("Preparing data for DSL estimation")

    # Create a copy of the data
    df = data.copy()

    # Important: Only use complete cases for labeled data
    # This matches R's behavior of not using rows with NA values
    complete_cases = df.dropna(subset=["countyWrong", "SendOrNot"])

    # Set random seed for reproducibility - use same seed as R
    np.random.seed(123)

    # Randomly select 500 observations from complete cases as labeled
    n_labeled = 500
    available_indices = complete_cases.index.tolist()
    labeled_indices = np.random.choice(available_indices, size=n_labeled, replace=False)

    # Create labeled indicator (1 for labeled observations, 0 for unlabeled)
    df["labeled"] = 0
    df.loc[labeled_indices, "labeled"] = 1
    labeled_count = df["labeled"].sum()
    unlabeled_count = len(df) - labeled_count
    logger.info(f"Labeled: {labeled_count}, Unlabeled: {unlabeled_count}")

    # Create sample probability (equal probability for labeled observations)
    # In R, this is calculated as n_labeled / n_total for complete cases
    n_complete = len(complete_cases)
    sample_prob = n_labeled / n_complete
    df["sample_prob"] = sample_prob
    logger.info(f"Sample probability: {sample_prob}")

    # Handle missing values in predictors
    # For unlabeled data, we'll fill NAs with 0 as before
    # For labeled data, we already ensured complete cases
    for col in [
        "countyWrong",
        "prefecWrong",
        "connect2b",
        "prevalence",
        "regionj",
        "groupIssue",
    ]:
        # Only fill NAs in unlabeled data
        mask = df["labeled"] == 0
        na_count = df.loc[mask, col].isna().sum()
        if na_count > 0:
            logger.info(
                f"Filling {na_count} NA values in '{col}' for " f"unlabeled data with 0"
            )
            df.loc[mask, col] = df.loc[mask, col].fillna(0)

    # For SendOrNot, we only need values for labeled data
    # For unlabeled data, fill with 0 (won't be used in estimation)
    na_count = df.loc[df["labeled"] == 0, "SendOrNot"].isna().sum()
    if na_count > 0:
        logger.info(
            f"Filling {na_count} NA values in SendOrNot for " f"unlabeled data with 0"
        )
        df.loc[df["labeled"] == 0, "SendOrNot"] = df.loc[
            df["labeled"] == 0, "SendOrNot"
        ].fillna(0)

    # Log data summary for labeled data only
    logger.info("\nLabeled Data Summary:")
    labeled_data = df[df["labeled"] == 1]
    for col in [
        "SendOrNot",
        "countyWrong",
        "prefecWrong",
        "connect2b",
        "prevalence",
        "regionj",
        "groupIssue",
    ]:
        logger.info(
            f"{col}: mean={labeled_data[col].mean()}, "
            f"count={labeled_data[col].count()}"
        )

    # Convert to appropriate types if needed
    # df["some_column"] = df["some_column"].astype(int)

    return df


def format_dsl_results(result, formula, ci=0.95, digits=4):
    """Format DSL results in R summary style.

    Args:
        result: DSLResult object from DSL estimation
        formula: Formula used in DSL estimation
        ci: Confidence interval level (default: 0.95)
        digits: Number of digits to round (default: 4)

    Returns:
        Formatted summary string in R style
    """
    logger.info("Formatting actual DSL results in R style")

    # Extract components
    coefs = result.coefficients
    ses = result.standard_errors

    logger.info(f"Coefficients: {coefs}")
    logger.info(f"Standard errors: {ses}")

    # Calculate p-values
    t_stats = coefs / ses
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    logger.info(f"t-statistics: {t_stats}")
    logger.info(f"p-values: {p_values}")

    # Calculate confidence intervals
    alpha = 1 - (1 - ci) / 2
    ci_lower = coefs - stats.norm.ppf(alpha) * ses
    ci_upper = coefs + stats.norm.ppf(alpha) * ses

    # Create significance stars
    sig_stars = []
    for p in p_values:
        if p < 0.001:
            sig_stars.append("***")
        elif p < 0.01:
            sig_stars.append("**")
        elif p < 0.05:
            sig_stars.append("*")
        elif p < 0.1:
            sig_stars.append(".")
        else:
            sig_stars.append("")

    # Extract variable names from the formula
    # The first is always the intercept
    terms = formula.split("~")[1].strip().split("+")
    terms = ["(Intercept)"] + [t.strip() for t in terms]

    # Create DataFrame for results
    results_df = pd.DataFrame(
        {
            "Estimate": coefs.round(digits),
            "Std. Error": ses.round(digits),
            "CI Lower": ci_lower.round(digits),
            "CI Upper": ci_upper.round(digits),
            "p value": p_values.round(digits),
            "": sig_stars,
        },
        index=terms[: len(coefs)],
    )

    # Build summary output
    summary = (
        "==================\n"
        "DSL Specification:\n"
        "==================\n"
        f"Model:  {result.model}\n"
        f"Call:  {formula}\n"
        "\n"
        "Predicted Variables:  countyWrong\n"
        "Prediction:  pred_countyWrong\n"
        "\n"
        f"Number of Labeled Observations:  {result.labeled_size}\n"
        "Random Sampling for Labeling with Equal Probability: Yes\n"
        "\n"
        "=============\n"
        "Coefficients:\n"
        "=============\n"
    )

    # Add results table
    summary += str(results_df) + "\n"
    summary += "---\n"
    summary += (
        "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n"
        f"{int(ci*100)}% confidence intervals (CI) are reported."
    )

    return summary


def main():
    """Main function to compare DSL implementations."""
    logger.info("Starting DSL comparison")

    try:
        # Load PanChen dataset
        data = load_panchen_data()

        # Prepare data for DSL estimation
        df = prepare_data_for_dsl(data)

        # Define the formula
        formula = (
            "SendOrNot ~ countyWrong + prefecWrong + connect2b + "
            "prevalence + regionj + groupIssue"
        )
        logger.info(f"Using formula: {formula}")

        print("\nRunning DSL estimation (logistic regression)...")

        # Prepare X and y using patsy
        y, X = dmatrices(formula, df, return_type="dataframe")

        # Run DSL estimation using the new interface
        result = dsl(
            X=X.values,
            y=y.values,
            labeled_ind=df["labeled"].values,
            sample_prob=df["sample_prob"].values,
            model="logit",
            method="logistic",
        )

        logger.info("DSL estimation completed successfully")
        logger.info(f"Result coefficients: {result.coefficients}")
        logger.info(f"Result standard errors: {result.standard_errors}")
        logger.info(f"Labeled size: {result.labeled_size}")
        logger.info(f"Total size: {result.total_size}")

        # Format and print results in R-style summary
        summary = format_dsl_results(result, formula)
        print(summary)

        # Save results to file for comparison
        import pickle

        with open("python_panchen_results.pkl", "wb") as f:
            pickle.dump(result, f)
        logger.info("Results saved to python_panchen_results.pkl")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
