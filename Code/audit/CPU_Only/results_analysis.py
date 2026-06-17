"""
File: CPU_Only/results_analysis.py
Purpose: Generate final tables and figures from scored results.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - matplotlib / seaborn visualisation.

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_DIR, OSM_MODELS, RESULTS_DIR, ensure_dirs

_OSM_NAMES = {m["name"] for m in OSM_MODELS}

logger = logging.getLogger(__name__)


def _load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all result parquet files. Returns (behavioral, cdva, scored, leaderboard)."""
    behavioral_path = RESULTS_DIR / "behavioral_results.parquet"
    cdva_path = RESULTS_DIR / "cdva_results.parquet"
    scored_path = RESULTS_DIR / "scored_results.parquet"
    leaderboard_path = RESULTS_DIR / "leaderboard.parquet"

    behavioral = pd.read_parquet(behavioral_path) if behavioral_path.exists() else pd.DataFrame()
    cdva = pd.read_parquet(cdva_path) if cdva_path.exists() else pd.DataFrame()
    scored = pd.read_parquet(scored_path) if scored_path.exists() else pd.DataFrame()
    leaderboard = pd.read_parquet(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()
    return behavioral, cdva, scored, leaderboard


def plot_mirage_b_pass_rates(scored_df: pd.DataFrame) -> None:
    """Bar chart: MIRAGE-B pass rate per model per benchmark."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if scored_df.empty:
            logger.warning("scored_df is empty; skipping MIRAGE-B plot.")
            return

        summary = (
            scored_df.groupby(["seed_source", "model_name"])["mirage_b_pass"]
            .mean()
            .reset_index()
            .rename(columns={"mirage_b_pass": "pass_rate"})
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=summary, x="model_name", y="pass_rate", hue="seed_source", ax=ax
        )
        ax.set_title("MIRAGE-B Pass Rate by Model and Benchmark")
        ax.set_xlabel("Model")
        ax.set_ylabel("Pass Rate")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out_path = FIGURES_DIR / "mirage_b_pass_rates.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved: %s", out_path)
    except Exception as exc:
        logger.warning("MIRAGE-B plot failed: %s", exc)


def plot_cdva_distribution(cdva_df: pd.DataFrame) -> None:
    """Violin plot: CDVA seed score distribution per model."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if cdva_df.empty:
            logger.warning("cdva_df is empty; skipping CDVA distribution plot.")
            return

        # Aggregate per (seed, model)
        agg = (
            cdva_df[cdva_df["success_flag"] == True]  # noqa: E712
            .groupby(["seed_id", "model_name"])["cdva_pair_score"]
            .mean()
            .reset_index()
            .rename(columns={"cdva_pair_score": "cdva_seed_score"})
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=agg, x="model_name", y="cdva_seed_score", ax=ax)
        ax.set_title("CDVA Seed Score Distribution per Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("CDVA Seed Score")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out_path = FIGURES_DIR / "cdva_distribution.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved: %s", out_path)
    except Exception as exc:
        logger.warning("CDVA distribution plot failed: %s", exc)


def plot_leaderboard_heatmap(leaderboard_df: pd.DataFrame) -> None:
    """Heatmap: 4x5 failure-mode validity matrix."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if leaderboard_df.empty:
            logger.warning("leaderboard_df is empty; skipping heatmap.")
            return

        fm_cols = ["FM1", "FM2", "FM3", "FM4", "FM5"]
        plot_df = leaderboard_df[[c for c in fm_cols if c in leaderboard_df.columns]]

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            plot_df.astype(float),
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title("MIRAGE Validity Leaderboard (Failure Mode Rates)")
        plt.tight_layout()
        out_path = FIGURES_DIR / "leaderboard_heatmap.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved: %s", out_path)
    except Exception as exc:
        logger.warning("Leaderboard heatmap failed: %s", exc)


def run_results_analysis() -> None:
    """Master function to generate all figures and print summary tables."""
    ensure_dirs()
    behavioral, cdva, scored, leaderboard = _load_results()

    logger.info("--- MIRAGE Results Summary ---")
    if not scored.empty:
        logger.info(
            "Overall MIRAGE-B pass rate: %.3f",
            scored["mirage_b_pass"].mean(),
        )
        osm_scored = scored[scored["model_name"].isin(_OSM_NAMES)]
        if len(osm_scored) > 0:
            logger.info(
                "Overall MIRAGE-Full pass rate (OSM only): %.3f",
                osm_scored["mirage_full_pass"].dropna().mean(),
            )

    plot_mirage_b_pass_rates(scored)
    plot_cdva_distribution(cdva)
    plot_leaderboard_heatmap(leaderboard)

    logger.info("Results analysis complete. Figures in: %s", FIGURES_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_results_analysis()
