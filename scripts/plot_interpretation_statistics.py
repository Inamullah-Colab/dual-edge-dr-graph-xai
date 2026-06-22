from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LESION_PREFIXES = (
    "x2_microaneurysm",
    "x2_hemorrhage",
    "x2_hard_exudate",
    "x2_cotton_wool_spot",
    "x2_neovascularization",
    "x2_total_evidence",
)


def savefig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=320, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def pretty_label(name: str, max_len: int = 46) -> str:
    label = name
    label = label.replace("x2_", "")
    label = label.replace("x4_macula__", "")
    label = label.replace("x1_", "")
    label = label.replace("x12_", "X12 ")
    label = label.replace("x34_", "X34 ")
    label = label.replace("graph_", "graph ")
    label = label.replace("_", " ")
    label = re.sub(r"\s+", " ", label).strip()
    label = label.replace("crve", "CRVE").replace("crae", "CRAE").replace("avr", "AVR")
    label = label.replace("CRVE hubbard", "CRVE Hubbard")
    label = label.replace("CRAE hubbard", "CRAE Hubbard")
    label = label.replace("AVR hubbard", "AVR Hubbard")
    if len(label) <= max_len:
        return label
    return label[: max_len - 1].rstrip() + "..."


def is_clinical_lesion(feature: str) -> bool:
    return feature.startswith(LESION_PREFIXES)


def signed_log_fdr(p: pd.Series) -> pd.Series:
    return -np.log10(pd.to_numeric(p, errors="coerce").fillna(1.0).clip(lower=1e-300))


def plot_forest(tests: pd.DataFrame, out_dir: Path, top_n: int) -> pd.DataFrame:
    df = tests.copy()
    df["abs_rho"] = df["grade_spearman_rho"].abs()
    keep = (
        df["feature"].str.startswith(("graph_", "x2_", "x4_", "x12_", "x34_"))
        & ~df["feature"].str.startswith("x2_map_embed_")
    )
    df = df[keep].sort_values(["significant_fdr_0_05", "abs_rho"], ascending=[False, False]).head(top_n)
    df = df.sort_values("grade_spearman_rho")

    colors = np.where(df["grade_spearman_rho"] >= 0, "#B3472E", "#2F6F9F")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.34 * len(df))))
    ax.axvline(0, color="#20242A", lw=1.0)
    ax.hlines(y, 0, df["grade_spearman_rho"], color=colors, lw=2.0, alpha=0.75)
    ax.scatter(df["grade_spearman_rho"], y, c=colors, s=48, zorder=3, edgecolor="white", linewidth=0.6)
    for yi, row in enumerate(df.itertuples(index=False)):
        txt = f"FDR={row.fdr_p_value:.1e}" if row.fdr_p_value > 0 else "FDR<1e-300"
        ha = "left" if row.grade_spearman_rho >= 0 else "right"
        xoff = 0.018 if row.grade_spearman_rho >= 0 else -0.018
        ax.text(row.grade_spearman_rho + xoff, yi, txt, va="center", ha=ha, fontsize=7, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_label(v, 54) for v in df["feature"]], fontsize=8)
    ax.set_xlabel("Spearman rho with DR grade")
    ax.set_title("FDR-corrected grade trends in graph, lesion, and biomarker evidence")
    ax.grid(axis="x", color="#D8DCE2", lw=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.8, 0.8)
    savefig(fig, out_dir / "forest_interpretable_grade_trends.png")
    return df


def plot_bubble(linkage: pd.DataFrame, out_dir: Path, top_n: int) -> pd.DataFrame:
    df = linkage.copy()
    clinical = df[df["lesion_feature"].map(is_clinical_lesion)].copy()
    if clinical.empty:
        clinical = df.copy()
    clinical["abs_rho"] = clinical["spearman_rho"].abs()
    clinical["log_fdr"] = signed_log_fdr(clinical["fdr_p_value"])
    clinical = clinical.sort_values(["abs_rho", "log_fdr"], ascending=False).head(top_n)

    lesion_order = list(dict.fromkeys(clinical["lesion_feature"]))
    biomarker_order = list(dict.fromkeys(clinical["biomarker"]))
    x = clinical["lesion_feature"].map({v: i for i, v in enumerate(lesion_order)})
    y = clinical["biomarker"].map({v: i for i, v in enumerate(biomarker_order)})
    sizes = 45 + 9 * clinical["log_fdr"].clip(upper=80)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(lesion_order)), max(5.6, 0.28 * len(biomarker_order))))
    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=clinical["spearman_rho"],
        cmap="RdBu_r",
        vmin=-0.7,
        vmax=0.7,
        edgecolors="#1F2933",
        linewidths=0.35,
        alpha=0.88,
    )
    ax.set_xticks(range(len(lesion_order)))
    ax.set_xticklabels([pretty_label(v, 26) for v in lesion_order], rotation=35, ha="right")
    ax.set_yticks(range(len(biomarker_order)))
    ax.set_yticklabels([pretty_label(v, 48) for v in biomarker_order], fontsize=8)
    ax.set_xlabel("Lesion evidence feature")
    ax.set_ylabel("AutoMorph biomarker")
    ax.set_title("FDR-corrected lesion-biomarker associations")
    ax.grid(color="#E4E7EB", lw=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Spearman rho")
    savefig(fig, out_dir / "bubble_clinical_lesion_biomarker_links.png")
    return clinical


def zscore(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce")
    std = arr.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return arr * 0
    return (arr - arr.mean()) / std


def plot_boxwhisker(features: pd.DataFrame, top_links: pd.DataFrame, out_dir: Path, n_pairs: int) -> list[str]:
    candidates: list[tuple[str, str, float]] = []
    for row in top_links.itertuples(index=False):
        if row.lesion_feature in features.columns and row.biomarker in features.columns:
            candidates.append((row.lesion_feature, row.biomarker, row.spearman_rho))
        if len(candidates) >= n_pairs:
            break

    if not candidates:
        return []

    grades = sorted(pd.to_numeric(features["diagnosis"], errors="coerce").dropna().astype(int).unique())
    fig, axes = plt.subplots(len(candidates), 2, figsize=(10.8, max(6.0, 2.35 * len(candidates))), sharex=True)
    if len(candidates) == 1:
        axes = np.asarray([axes])

    plotted: list[str] = []
    for r, (lesion, biomarker, rho) in enumerate(candidates):
        for c, feature in enumerate((lesion, biomarker)):
            ax = axes[r, c]
            tmp = pd.DataFrame({"grade": features["diagnosis"].astype(int), "value": zscore(features[feature])})
            values = [tmp.loc[tmp["grade"] == g, "value"].dropna().to_numpy() for g in grades]
            bp = ax.boxplot(
                values,
                positions=np.arange(len(grades)),
                widths=0.62,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111827", "lw": 1.4},
                whiskerprops={"color": "#4B5563", "lw": 1.0},
                capprops={"color": "#4B5563", "lw": 1.0},
            )
            fill = "#E9F0F7" if c == 0 else "#F7ECE6"
            edge = "#2F6F9F" if c == 0 else "#B3472E"
            for patch in bp["boxes"]:
                patch.set_facecolor(fill)
                patch.set_edgecolor(edge)
                patch.set_linewidth(1.1)
            means = [np.nanmean(v) if len(v) else np.nan for v in values]
            ax.plot(np.arange(len(grades)), means, color=edge, marker="o", ms=3.8, lw=1.4)
            ax.axhline(0, color="#CBD2D9", lw=0.8)
            ax.set_title(pretty_label(feature, 48), fontsize=10)
            ax.set_xticks(np.arange(len(grades)))
            ax.set_xticklabels(grades)
            ax.grid(axis="y", color="#E4E7EB", lw=0.6, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel("z-scored value")
                ax.text(
                    -0.42,
                    0.92,
                    f"link rho={rho:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#333333",
                    va="top",
                )
            if r == len(candidates) - 1:
                ax.set_xlabel("DR grade")
        plotted.append(f"{lesion} -> {biomarker} (rho={rho:.3f})")

    fig.suptitle("Grade-wise box/whisker distributions for top lesion-biomarker links", y=1.01, fontsize=12)
    savefig(fig, out_dir / "boxwhisker_top_lesion_biomarker_grade_trends.png")
    return plotted


def write_summary(out_dir: Path, forest_df: pd.DataFrame, bubble_df: pd.DataFrame, box_pairs: list[str]) -> None:
    lines = [
        "# Interpretation Plot Summary",
        "",
        "Generated publication-style interpretation plots from the APTOS non-augmented full fusion graph outputs.",
        "",
        "## Files",
        "",
        "- `forest_interpretable_grade_trends.png/.pdf`: FDR-corrected feature trends across DR grade.",
        "- `bubble_clinical_lesion_biomarker_links.png/.pdf`: clinical lesion evidence versus AutoMorph biomarker links.",
        "- `boxwhisker_top_lesion_biomarker_grade_trends.png/.pdf`: grade-wise distributions for the strongest readable links.",
        "",
        "## Top Forest Features",
        "",
    ]
    for row in forest_df.head(12).itertuples(index=False):
        lines.append(
            f"- `{row.feature}`: rho={row.grade_spearman_rho:.3f}, FDR={row.fdr_p_value:.2e}, {row.grade_trend_direction}"
        )
    lines.extend(["", "## Top Lesion-Biomarker Links", ""])
    for row in bubble_df.head(12).itertuples(index=False):
        lines.append(
            f"- `{row.lesion_feature}` vs `{row.biomarker}`: rho={row.spearman_rho:.3f}, FDR={row.fdr_p_value:.2e}"
        )
    lines.extend(["", "## Box/Whisker Pairs", ""])
    for pair in box_pairs:
        lines.append(f"- {pair}")
    (out_dir / "INTERPRETATION_PLOTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make readable interpretation plots for lesion-biomarker statistical tests.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--hypothesis-tests", type=Path, required=True)
    parser.add_argument("--linkage", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-forest", type=int, default=22)
    parser.add_argument("--top-bubble", type=int, default=35)
    parser.add_argument("--box-pairs", type=int, default=4)
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 140,
            "savefig.facecolor": "white",
        }
    )

    features = pd.read_csv(args.features)
    tests = pd.read_csv(args.hypothesis_tests)
    linkage = pd.read_csv(args.linkage)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    forest_df = plot_forest(tests, args.out_dir, args.top_forest)
    bubble_df = plot_bubble(linkage, args.out_dir, args.top_bubble)
    box_pairs = plot_boxwhisker(features, bubble_df, args.out_dir, args.box_pairs)
    write_summary(args.out_dir, forest_df, bubble_df, box_pairs)

    print(f"Wrote interpretation plots to {args.out_dir}")


if __name__ == "__main__":
    main()
