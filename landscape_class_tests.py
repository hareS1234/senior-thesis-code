#!/usr/bin/env python
"""
landscape_class_tests.py

Statistical tests for kinetic differences between disconnectivity-graph
landscape classes (single-funnel, moderately frustrated, multi-funnel).

Tests performed:
    1. Kruskal-Wallis H-test across three classes (non-parametric one-way ANOVA)
    2. Pairwise Mann-Whitney U tests with Bonferroni correction
    3. Effect sizes (rank-biserial correlation for each pair)

Targets: log10(MFPT_AB), log10(MFPT_BA), log10(t1), t1/t2

Outputs
-------
  {out_dir}/kruskal_wallis_results.csv        — H statistic, p-value per target
  {out_dir}/mann_whitney_pairwise.csv          — all pairwise comparisons
  {out_dir}/landscape_class_descriptive.csv    — per-class descriptive stats
  {out_dir}/fig_landscape_class_boxplots.pdf   — box + strip plots per target

Usage:
    python landscape_class_tests.py \
        --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
        --out-dir landscape_class_tests
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
#  Landscape class assignments (from Table 4.2 in the thesis)
# ======================================================================

LANDSCAPE_CLASSES = {
    # Single-funnel
    "aaaaaa": "single-funnel",
    "aaggaa": "single-funnel",
    "eeeeee": "single-funnel",
    "gggggg": "single-funnel",
    "kkkkkk": "single-funnel",
    "rrrrrr": "single-funnel",
    "vvvvvv": "single-funnel",
    # Moderately frustrated
    "ffggff": "moderate",
    "flgglf": "moderate",
    "lfggfl": "moderate",
    "llggll": "moderate",
    "mvggvv": "moderate",
    "klvffa": "moderate",
    "svsssy": "moderate",
    "keggek": "moderate",
    "regger": "moderate",
    "kyggyk": "moderate",
    "ryggyr": "moderate",
    "ykggky": "moderate",
    "yrggry": "moderate",
    "yyggyy": "moderate",
    # Multi-funnel
    "gaiigl": "multi-funnel",
    "gailss": "multi-funnel",
    "ggvvia": "multi-funnel",
    "gyviik": "multi-funnel",
    "nfgail": "multi-funnel",
    "snqnnf": "multi-funnel",
    "ssqvtq": "multi-funnel",
    "sstnvg": "multi-funnel",
    "vqivyk": "multi-funnel",
}

CLASS_ORDER = ["single-funnel", "moderate", "multi-funnel"]
CLASS_DISPLAY = {
    "single-funnel": "single-funnel",
    "moderate": "moderately frustrated",
    "multi-funnel": "multi-funnel",
}


# ======================================================================
#  Effect size: rank-biserial correlation
# ======================================================================

def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """
    Rank-biserial correlation r_rb as effect size for Mann-Whitney U.

        r_rb = 1 - 2U / (n1 * n2)

    Interpretation:
        |r_rb| < 0.3  : small
        0.3-0.5       : medium
        > 0.5         : large
    """
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    n1, n2 = len(x), len(y)
    return 1.0 - (2.0 * res.statistic) / (n1 * n2)


# ======================================================================
#  Data loading and class assignment
# ======================================================================

def load_and_classify(targets_csv: Path) -> pd.DataFrame:
    """Load GT validation CSV and assign landscape classes."""
    df = pd.read_csv(targets_csv)

    # Extract sequence from dps_dir (e.g., "aaaaaa_nocap/aaaaaa_99idps_nocap" -> "aaaaaa")
    df["dps_dir"] = df["dps_dir"].astype(str).str.rstrip("/")
    df["sequence"] = df["dps_dir"].apply(
        lambda p: Path(p).parent.name.replace("_nocap", "")
        if "/" in p else Path(p).name.split("_")[0]
    )

    # Assign landscape class
    df["landscape_class"] = df["sequence"].map(LANDSCAPE_CLASSES)

    # Add log-transformed targets
    for col, log_col in [
        ("MFPT_coarse_AB", "log_MFPT_AB"),
        ("MFPT_coarse_BA", "log_MFPT_BA"),
        ("t1", "log_t1"),
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                df[log_col] = np.log10(vals.values)

    # Keep only rows with a valid class and at least one finite target
    df = df[df["landscape_class"].notna()].copy()

    return df


def finite_analysis_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to the networks used for the class-level kinetic comparison.

    The main-text table groups the 36 systems with complete coarse observables,
    so the statistical tests should operate on the same subset rather than on
    whichever rows happen to be finite target-by-target.
    """
    required = ["log_MFPT_AB", "log_MFPT_BA", "log_t1", "t1_over_t2"]
    keep = np.ones(len(df), dtype=bool)
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required target '{col}' missing after preprocessing.")
        keep &= np.isfinite(pd.to_numeric(df[col], errors="coerce"))
    return df.loc[keep].copy()


# ======================================================================
#  Main analysis
# ======================================================================

TARGET_COLS = {
    "log_MFPT_AB": "$\\log_{10}\\mathrm{MFPT}_{A \\to B}$",
    "log_MFPT_BA": "$\\log_{10}\\mathrm{MFPT}_{B \\to A}$",
    "log_t1": "$\\log_{10} t_1$",
    "t1_over_t2": "$t_1 / t_2$",
}


def main():
    parser = argparse.ArgumentParser(
        description="Statistical tests for landscape class kinetic differences."
    )
    parser.add_argument("--targets-csv", type=Path,
                        default=Path("GTcheck_micro_vs_coarse_T300K_full.csv"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("landscape_class_tests"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and classify ──────────────────────────────────────────────
    df = load_and_classify(args.targets_csv)
    analysis_df = finite_analysis_subset(df)
    print(f"[landscape_tests] {len(df)} classified networks loaded")
    print(f"[landscape_tests] {len(analysis_df)} networks retained with complete coarse observables")
    for cls in CLASS_ORDER:
        n = (analysis_df["landscape_class"] == cls).sum()
        print(f"  {CLASS_DISPLAY[cls]}: n = {n}")

    analysis_df.to_csv(args.out_dir / "analysis_networks.csv", index=False)

    # ── Descriptive statistics ─────────────────────────────────────────
    desc_rows = []
    for target in TARGET_COLS:
        if target not in analysis_df.columns:
            continue
        for cls in CLASS_ORDER:
            vals = analysis_df.loc[
                analysis_df["landscape_class"] == cls, target
            ].dropna().values
            if len(vals) == 0:
                continue
            desc_rows.append({
                "target": target,
                "class": cls,
                "class_display": CLASS_DISPLAY[cls],
                "n": len(vals),
                "mean": np.mean(vals),
                "median": np.median(vals),
                "std": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                "min": np.min(vals),
                "max": np.max(vals),
                "IQR": np.percentile(vals, 75) - np.percentile(vals, 25),
            })

    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(args.out_dir / "landscape_class_descriptive.csv", index=False)
    print("\n[landscape_tests] Descriptive statistics saved.")

    # ── Kruskal-Wallis tests ───────────────────────────────────────────
    kw_rows = []
    for target in TARGET_COLS:
        if target not in analysis_df.columns:
            continue

        groups = []
        for cls in CLASS_ORDER:
            vals = analysis_df.loc[
                analysis_df["landscape_class"] == cls, target
            ].dropna().values
            if len(vals) > 0:
                groups.append(vals)

        if len(groups) < 2:
            continue

        H, p = stats.kruskal(*groups)
        # Effect size: eta-squared (H) = (H - k + 1) / (N - k)
        N_total = sum(len(g) for g in groups)
        k = len(groups)
        eta_sq_H = (H - k + 1) / (N_total - k) if N_total > k else np.nan
        if np.isfinite(eta_sq_H):
            eta_sq_H = max(0.0, eta_sq_H)

        kw_rows.append({
            "target": target,
            "H_statistic": H,
            "p_value": p,
            "eta_sq_H": eta_sq_H,
            "n_total": N_total,
            "n_groups": k,
            "significant_005": p < 0.05,
            "significant_001": p < 0.01,
        })

        print(f"\n  Kruskal-Wallis for {target}:")
        print(f"    H = {H:.4f}, p = {p:.4f}, eta²_H = {eta_sq_H:.4f}")

    kw_df = pd.DataFrame(kw_rows)
    kw_df.to_csv(args.out_dir / "kruskal_wallis_results.csv", index=False)

    # ── Pairwise Mann-Whitney U with Bonferroni correction ─────────────
    mw_rows = []
    n_comparisons = 3  # C(3,2) = 3 pairs

    for target in TARGET_COLS:
        if target not in analysis_df.columns:
            continue

        class_data = {}
        for cls in CLASS_ORDER:
            vals = analysis_df.loc[
                analysis_df["landscape_class"] == cls, target
            ].dropna().values
            if len(vals) > 0:
                class_data[cls] = vals

        for cls_a, cls_b in combinations(CLASS_ORDER, 2):
            if cls_a not in class_data or cls_b not in class_data:
                continue

            x, y = class_data[cls_a], class_data[cls_b]
            U, p_raw = stats.mannwhitneyu(x, y, alternative="two-sided")
            p_bonf = min(p_raw * n_comparisons, 1.0)
            r_rb = rank_biserial(x, y)

            mw_rows.append({
                "target": target,
                "group_A": cls_a,
                "group_B": cls_b,
                "group_A_display": CLASS_DISPLAY[cls_a],
                "group_B_display": CLASS_DISPLAY[cls_b],
                "n_A": len(x),
                "n_B": len(y),
                "U_statistic": U,
                "p_raw": p_raw,
                "p_bonferroni": p_bonf,
                "bonferroni_alpha": 0.05 / n_comparisons,
                "rank_biserial_r": r_rb,
                "significant_005_bonf": p_bonf < 0.05,
                "significant_001_bonf": p_bonf < 0.01,
                "effect_size": (
                    "large" if abs(r_rb) > 0.5 else
                    "medium" if abs(r_rb) > 0.3 else "small"
                ),
            })

            print(f"    {cls_a} vs {cls_b}: U={U:.1f}, "
                  f"p_raw={p_raw:.4f}, p_bonf={p_bonf:.4f}, "
                  f"r_rb={r_rb:.3f} ({mw_rows[-1]['effect_size']})")

    mw_df = pd.DataFrame(mw_rows)
    mw_df.to_csv(args.out_dir / "mann_whitney_pairwise.csv", index=False)

    # ── Box + strip plots ──────────────────────────────────────────────
    available_targets = [t for t in TARGET_COLS if t in df.columns]
    n_targets = len(available_targets)

    if n_targets > 0:
        fig, axes = plt.subplots(1, n_targets, figsize=(4.5 * n_targets, 5))
        if n_targets == 1:
            axes = [axes]

        colors = {"single-funnel": "#4C72B0", "moderate": "#DD8452", "multi-funnel": "#55A868"}

        for ax, target in zip(axes, available_targets):
            plot_data = []
            positions = []
            labels = []
            for i, cls in enumerate(CLASS_ORDER):
                vals = analysis_df.loc[
                    analysis_df["landscape_class"] == cls, target
                ].dropna().values
                if len(vals) > 0:
                    plot_data.append(vals)
                    positions.append(i)
                    labels.append(f"{CLASS_DISPLAY[cls]}\n(n={len(vals)})")

            bp = ax.boxplot(
                plot_data, positions=positions, widths=0.5,
                patch_artist=True, showfliers=False,
            )
            for patch, pos in zip(bp["boxes"], positions):
                cls = CLASS_ORDER[pos]
                patch.set_facecolor(colors[cls])
                patch.set_alpha(0.4)
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)

            # Strip (jitter) overlay
            for pos_i, vals in zip(positions, plot_data):
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
                cls = CLASS_ORDER[pos_i]
                ax.scatter(pos_i + jitter, vals, s=25, alpha=0.7,
                           color=colors[cls], edgecolors="white", linewidths=0.3,
                           zorder=3)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel(TARGET_COLS[target], fontsize=11)
            ax.grid(True, axis="y", alpha=0.3)

            # Add significance bars from Kruskal-Wallis
            kw_row = kw_df[kw_df["target"] == target]
            if len(kw_row) > 0:
                p = kw_row.iloc[0]["p_value"]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                ax.set_title(f"KW: $p$ = {p:.3f} ({stars})", fontsize=10)

            sig_rows = mw_df[
                (mw_df["target"] == target) &
                (mw_df["significant_005_bonf"])
            ]
            if len(sig_rows) > 0:
                y_min = min(np.min(vals) for vals in plot_data)
                y_max = max(np.max(vals) for vals in plot_data)
                y_span = y_max - y_min if y_max > y_min else 1.0
                bar_height = y_max + 0.08 * y_span
                bar_step = 0.10 * y_span
                pos_lookup = {cls: pos for cls, pos in zip(CLASS_ORDER, positions)}
                for i_sig, row in enumerate(sig_rows.itertuples()):
                    x1 = pos_lookup[row.group_A]
                    x2 = pos_lookup[row.group_B]
                    y = bar_height + i_sig * bar_step
                    ax.plot([x1, x1, x2, x2],
                            [y, y + 0.02 * y_span, y + 0.02 * y_span, y],
                            color="black", linewidth=1.0)
                    stars = (
                        "***" if row.p_bonferroni < 0.001 else
                        "**" if row.p_bonferroni < 0.01 else
                        "*"
                    )
                    ax.text((x1 + x2) / 2, y + 0.025 * y_span, stars,
                            ha="center", va="bottom", fontsize=10)
                ax.set_ylim(top=bar_height + (len(sig_rows) + 1) * bar_step)

        fig.suptitle(
            "Kinetic observables by landscape class",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(
            args.out_dir / "fig_landscape_class_boxplots.pdf",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"\n[landscape_tests] Box plots saved.")

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  KRUSKAL-WALLIS SUMMARY")
    print(f"{'='*60}")
    print(kw_df[["target", "H_statistic", "p_value", "eta_sq_H",
                  "significant_005"]].to_string(index=False))

    print(f"\n  SIGNIFICANT PAIRWISE COMPARISONS (Bonferroni-corrected)")
    print(f"{'='*60}")
    sig = mw_df[mw_df["significant_005_bonf"]]
    if len(sig) > 0:
        print(sig[["target", "group_A", "group_B", "p_bonferroni",
                    "rank_biserial_r", "effect_size"]].to_string(index=False))
    else:
        print("  None at alpha = 0.05 after Bonferroni correction.")

    print(f"\nAll outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
