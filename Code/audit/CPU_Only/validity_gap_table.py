"""
Build public native vs MIRAGE validity-gap table per source benchmark.

Native pass  = slot-a answer matches gold (what the source benchmark alone tests).
MIRAGE-Full  = mirage_full_pass from scored_results.parquet.

Outputs:
  results/validity_gap_leaderboard.md   (public markdown table)
  results/validity_gap_leaderboard.parquet

Usage (after scoring):
    python -m CPU_Only.validity_gap_table
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, ensure_dirs
from CPU_Only.scoring import _answers_match

_OSM_NAMES = {m["name"] for m in OSM_MODELS}

logger = logging.getLogger(__name__)

_BENCHMARKS = ["bbq", "crows_pairs", "stereoset"]
_MD_PATH = RESULTS_DIR / "validity_gap_leaderboard.md"
_PARQUET_PATH = RESULTS_DIR / "validity_gap_leaderboard.parquet"


def _native_pass(behavioral_df: pd.DataFrame, seed_id: str, model_name: str) -> bool:
    rows = behavioral_df[
        (behavioral_df["seed_id"] == seed_id)
        & (behavioral_df["model_name"] == model_name)
        & (behavioral_df["slot"] == "a")
        & (behavioral_df["subvariant"] == "surface")
        & (behavioral_df["sample_index"] == 0)
        & (behavioral_df["success_flag"] == True)  # noqa: E712
    ]
    if rows.empty or "gold_answer" not in rows.columns:
        return False
    return _answers_match(
        str(rows.iloc[0]["parsed_answer"]),
        str(rows.iloc[0]["gold_answer"]),
        str(rows.iloc[0].get("seed_source", "")),
    )


def build_validity_gap_table(
    behavioral_df: pd.DataFrame,
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per (benchmark, model): native pass rate, MIRAGE-Full pass rate, gap."""
    records: list[dict] = []
    for benchmark in _BENCHMARKS:
        b_beh = behavioral_df[behavioral_df["seed_source"] == benchmark]
        b_scored = scored_df.merge(
            b_beh[["seed_id", "seed_source"]].drop_duplicates(),
            on="seed_id",
            how="inner",
        )
        if b_beh.empty:
            continue
        for model_name in sorted(b_scored["model_name"].unique()):
            seeds = b_scored[b_scored["model_name"] == model_name]["seed_id"].unique()
            native = [
                _native_pass(b_beh, sid, model_name)
                for sid in seeds
            ]
            n = len(seeds)
            native_rate = sum(native) / n if n else 0.0

            if model_name in _OSM_NAMES:
                mirage = b_scored[
                    (b_scored["model_name"] == model_name)
                ].set_index("seed_id")["mirage_full_pass"].reindex(seeds)
                mirage_valid = mirage.dropna().astype(bool)
                mirage_rate = float(mirage_valid.sum()) / n if n else 0.0
                gap = native_rate - mirage_rate
            else:
                # API models: MIRAGE-Full not defined (no CDVA).
                mirage_rate = float("nan")
                gap = float("nan")

            records.append(
                {
                    "benchmark": benchmark,
                    "model_name": model_name,
                    "n_seeds": n,
                    "native_pass_rate": native_rate,
                    "mirage_full_pass_rate": mirage_rate,
                    "validity_gap": gap,
                }
            )
    return pd.DataFrame(records)


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# MIRAGE Validity Gap Leaderboard",
        "",
        "Native pass rate = correct on slot-(a) only (source benchmark surface test).",
        "MIRAGE-Full = behavioural + CDVA validity instrument pass.",
        "Validity gap = native − MIRAGE-Full (hidden invalidity the source benchmark misses).",
        "",
        "## Per benchmark (macro-averaged over models)",
        "",
        "| Benchmark | N seeds | Native pass | MIRAGE-Full pass | Validity gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for benchmark in _BENCHMARKS:
        sub = df[df["benchmark"] == benchmark]
        if sub.empty:
            continue
        n = int(sub["n_seeds"].iloc[0])
        native = sub["native_pass_rate"].mean()
        mirage = sub["mirage_full_pass_rate"].mean()
        gap = sub["validity_gap"].mean()
        label = {"bbq": "BBQ", "crows_pairs": "CrowS-Pairs", "stereoset": "StereoSet"}[benchmark]
        lines.append(
            f"| {label} | {n} | {native:.1%} | {mirage:.1%} | **{gap:.1%}** |"
        )

    lines.extend(
        [
            "",
            "## Per model × benchmark",
            "",
            "| Model | Benchmark | Native | MIRAGE-Full | Gap |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in df.sort_values(["model_name", "benchmark"]).iterrows():
        label = {"bbq": "BBQ", "crows_pairs": "CrowS", "stereoset": "StereoSet"}[
            row["benchmark"]
        ]
        lines.append(
            f"| {row['model_name']} | {label} | {row['native_pass_rate']:.1%} | "
            f"{row['mirage_full_pass_rate']:.1%} | {row['validity_gap']:.1%} |"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ensure_dirs()
    beh_path = RESULTS_DIR / "behavioral_results.parquet"
    scored_path = RESULTS_DIR / "scored_results.parquet"
    if not beh_path.exists() or not scored_path.exists():
        logger.error("Need behavioral_results.parquet and scored_results.parquet first.")
        sys.exit(1)

    behavioral = pd.read_parquet(beh_path)
    scored = pd.read_parquet(scored_path)
    df = build_validity_gap_table(behavioral, scored)
    df.to_parquet(_PARQUET_PATH, index=False)
    _write_markdown(df, _MD_PATH)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
