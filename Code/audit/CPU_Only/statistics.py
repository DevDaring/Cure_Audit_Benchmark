"""
File: CPU_Only/statistics.py
Purpose: Statistical analysis module -- bootstrap CIs, McNemar, Cohen's h,
         Holm-Bonferroni, and Benjamini-Hochberg FDR.

Implements / builds on / cites:
  - Efron & Tibshirani (1993). An Introduction to the Bootstrap.
    Chapman & Hall. -- bootstrap CI method.
  - McNemar (1947). "Note on the sampling error of the difference between
    correlated proportions or percentages." Psychometrika, 12, 153-157.
  - Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.
    -- Cohen's h effect size.
  - Holm (1979). "A simple sequentially rejective multiple test procedure."
    Scandinavian Journal of Statistics, 6, 65-70.
  - Benjamini & Hochberg (1995). "Controlling the false discovery rate."
    J. Royal Statistical Society B, 57, 289-300.

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    statistic: str = "mean",
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI using the percentile method.

    Implements: Efron & Tibshirani (1993), An Introduction to the Bootstrap.

    Parameters
    ----------
    values : array-like
    n_resamples : int
    alpha : float
    statistic : str
        'mean' or 'proportion'

    Returns
    -------
    tuple[float, float, float]
        (point_estimate, lower_ci, upper_ci)
    """
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed=20260101)

    if statistic == "mean":
        fn = np.mean
    elif statistic == "proportion":
        fn = np.mean  # proportion = mean of binary array
    else:
        raise ValueError(f"Unknown statistic: '{statistic}'")

    point = float(fn(arr))
    boot_stats = np.array([fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)])
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, lower, upper


def mcnemar_paired(table_2x2: list[list[int]]) -> tuple[float, float]:
    """
    McNemar's test for paired binary outcomes.

    Implements:
        Exact test for n < 25 (binomial), chi-squared with continuity
        correction otherwise.

    Parameters
    ----------
    table_2x2 : list[list[int]]
        [[n00, n01], [n10, n11]] where
        n01 = cases where A fails, B passes;
        n10 = cases where A passes, B fails.

    Returns
    -------
    tuple[float, float]
        (statistic, p_value)
    """
    b = table_2x2[0][1]
    c = table_2x2[1][0]
    n_discordant = b + c

    if n_discordant < 25:
        # Exact binomial test
        result = stats.binomtest(min(b, c), n=n_discordant, p=0.5, alternative="two-sided")
        return float(min(b, c)), float(result.pvalue)
    else:
        # Chi-squared with continuity correction
        if n_discordant == 0:
            return 0.0, 1.0
        stat = (abs(b - c) - 1) ** 2 / n_discordant
        pvalue = float(1 - stats.chi2.cdf(stat, df=1))
        return float(stat), pvalue


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for two proportions.

    Implements: Cohen (1988), Statistical Power Analysis.

    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    """
    h = 2.0 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    return float(h)


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.

    Implements: Holm (1979).

    Returns
    -------
    list[float]
        Adjusted p-values in the same order as input.
    """
    n = len(pvalues)
    if n == 0:
        return []

    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0

    for rank, (orig_idx, pval) in enumerate(indexed):
        factor = n - rank
        adj = min(float(pval) * factor, 1.0)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max

    return adjusted


def bh_fdr(pvalues: list[float], alpha: float = 0.05) -> list[float]:
    """
    Benjamini-Hochberg FDR correction.

    Implements: Benjamini & Hochberg (1995).

    Returns
    -------
    list[float]
        Adjusted p-values (same order as input).
    """
    n = len(pvalues)
    if n == 0:
        return []

    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [1.0] * n
    prev_bh = 1.0

    for rank in range(n - 1, -1, -1):
        orig_idx, pval = indexed[rank]
        bh_val = float(pval) * n / (rank + 1)
        prev_bh = min(prev_bh, bh_val)
        adjusted[orig_idx] = min(prev_bh, 1.0)

    return adjusted


def two_proportion_ztest(p1: float, p2: float, n1: int, n2: int) -> tuple[float, float]:
    """
    Two-proportion z-test.

    Returns
    -------
    tuple[float, float]
        (z_statistic, p_value)
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(z), pval
