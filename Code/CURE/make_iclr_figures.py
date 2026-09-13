"""
File: Code/CURE/make_iclr_figures.py
Purpose: Regenerate the results figure for the ICLR 2027 submission from the saved result
         files, so that no number in the figure is typed by hand and the figure reads in
         grayscale as well as in colour.

Figure produced
  fig_frontier.pdf   four panels, one per model: causal signal removed (x) against utility
                     cost (y) for audit-guided erasure and the published comparison methods
                     of the head-to-head table. Markers replace the per-point text labels of
                     the earlier PNG, which overlapped on two panels. Self-Debias (removed
                     0, cost 0 on every model) sits at the origin and is left off the plotted
                     range; the caption says so.

Comparison set. Exactly the nine rows of the head-to-head table: audit-guided erasure and
the eight published methods in BASELINES. The 'patchscopes' row present in the result
files belongs to a different study and is excluded here, as it is from the tables.

Source of every number: Code/CURE/results/cure_final_<model>.parquet
  columns causal_residual_removed and utility_cost, identical to tab:h2h.

Usage:
  python Code/CURE/make_iclr_figures.py
  python Code/CURE/make_iclr_figures.py --out Submission2/iclr2027/figures
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd               # noqa: E402

log = logging.getLogger("iclr_figures")

ROOT = Path(__file__).resolve().parents[2]
CURE_R = ROOT / "Code" / "CURE" / "results"

MODELS = ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct",
          "gemma-2-2b-it", "phi-4-mini-instruct"]
DISPLAY = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "qwen2.5-7b-instruct": "Qwen2.5-7B",
           "gemma-2-2b-it": "Gemma-2-2B", "phi-4-mini-instruct": "Phi-4-mini"}

# Same names and order as tab:h2h (make_ml_tables.py).
BASELINES = [
    ("prompt_debias", "Self-Debias"),
    ("generic_erase", "GenericErase"),
    ("meandiff_steer", "MeanDiff steering"),
    ("fairsteer", "FairSteer"),
    ("biasgym", "BiasGym"),
    ("sae_debias", "SAE-Debias"),
    ("hsal", "H-SAL"),
    ("nofreelunch", "LogitSteer"),
]
# marker, size, face colour: distinct shapes so the panel reads without colour
STYLE = {
    "cure": ("*", 190, "black"),
    "prompt_debias": (">", 44, "white"),
    "generic_erase": ("o", 40, "white"),
    "meandiff_steer": ("P", 48, "white"),
    "fairsteer": ("s", 40, "white"),
    "biasgym": ("v", 46, "white"),
    "sae_debias": ("^", 46, "white"),
    "hsal": ("X", 48, "white"),
    "nofreelunch": ("D", 36, "white"),
}
UTILITY_BUDGET = 0.15   # the rank-selection budget stated in Section 3.2

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "font.family": "DejaVu Sans", "axes.linewidth": 0.9,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(model):
    d = pd.read_parquet(CURE_R / f"cure_final_{model}.parquet")
    keep = ["cure"] + [k for k, _ in BASELINES]
    d = d[d["method"].isin(keep)].copy()
    d["utility_cost"] = d["utility_cost"].astype(float)
    d["causal_residual_removed"] = d["causal_residual_removed"].astype(float)
    return d.set_index("method")


def fig_frontier(out_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9), sharey=True)
    handles, labels = {}, {}
    names = {"cure": "Audit-guided erasure"}
    names.update(dict(BASELINES))
    for ax, model in zip(axes, MODELS):
        d = load(model)
        for method in ["cure"] + [k for k, _ in BASELINES]:
            if method not in d.index:
                log.warning("%s: method %s missing", model, method)
                continue
            if method == "prompt_debias":
                # removes nothing and costs nothing: the point sits at the origin and would
                # compress the x axis, so the caption states it and the plot omits it
                continue
            mk, sz, face = STYLE[method]
            h = ax.scatter(d.loc[method, "causal_residual_removed"],
                           d.loc[method, "utility_cost"],
                           marker=mk, s=sz, facecolor=face, edgecolor="black",
                           linewidth=0.9, zorder=4)
            handles.setdefault(method, h)
            labels.setdefault(method, names[method])
        ax.axhline(UTILITY_BUDGET, color="0.45", lw=0.9, ls="--", zorder=1)
        ax.axhline(0.0, color="0.75", lw=0.7, ls=":", zorder=1)
        shown = d.drop(index="prompt_debias", errors="ignore")
        lo, hi = shown["causal_residual_removed"].min(), shown["causal_residual_removed"].max()
        pad = max(0.02, 0.15 * (hi - lo))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(DISPLAY[model])
        ax.set_xlabel("Causal signal removed")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, lw=0.4, color="0.9", zorder=0)
    axes[0].set_ylabel("Utility cost\n(accuracy drop, lower is better)")
    axes[0].text(0.02, 0.965, "dashed: 0.15 utility budget", transform=axes[0].transAxes,
                 fontsize=7.5, color="0.35", va="top")
    order = ["cure"] + [k for k, _ in BASELINES]
    fig.legend([handles[k] for k in order if k in handles],
               [labels[k] for k in order if k in labels],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.16),
               handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_frontier.pdf"
    fig.savefig(path)
    plt.close(fig)
    log.info("wrote %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "Submission2" / "iclr2027" / "figures"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fig_frontier(Path(args.out))


if __name__ == "__main__":
    main()
