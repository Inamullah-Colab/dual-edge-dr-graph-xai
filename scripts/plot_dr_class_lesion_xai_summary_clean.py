from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rxg.dr_xai_evidence import DRXAIPreprocessor, LESION_CHANNELS, lesion_evidence_maps, norm01


GRADE_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "PDR"}
CHANNEL_LABELS = {
    "microaneurysm": "Microaneurysm",
    "hemorrhage": "Hemorrhage",
    "hard_exudate": "Hard exudate",
    "cotton_wool_spot": "Cotton-wool spot",
    "neovascularization": "Neovascularization",
}


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.34) -> np.ndarray:
    heat = norm01(heat)
    h8 = (heat * 255).astype(np.uint8)
    cam = cv2.cvtColor(cv2.applyColorMap(h8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * rgb + alpha * cam, 0, 255).astype(np.uint8)


def short_label(name: str) -> str:
    repl = {
        "x2_neovascularization_std": "NV variability",
        "x2_neovascularization_mean": "NV mean",
        "x2_hemorrhage_std": "Hemorrhage variability",
        "x2_hemorrhage_area_035": "Hemorrhage area >0.35",
        "x2_hemorrhage_area_050": "Hemorrhage area >0.50",
        "x4_macula__CRAE_Hubbard_zone_b": "CRAE zone B",
        "x4_macula__CRAE_Hubbard_zone_c": "CRAE zone C",
        "x4_macula__CRVE_Hubbard_zone_c": "CRVE zone C",
        "x4_macula__Artery_Vessel_density_zone_c": "Artery density zone C",
        "x4_macula__Average_width_zone_c": "Average width zone C",
        "x4_macula__Vein_Average_width_zone_c": "Vein width zone C",
    }
    if name in repl:
        return repl[name]
    return name.replace("x2_", "").replace("x4_macula__", "").replace("_", " ")[:34]


def plot_assoc(ax: plt.Axes, linkage: pd.DataFrame, top_n: int = 7) -> None:
    clinical = linkage[
        linkage["lesion_feature"].str.startswith(("x2_neovascularization", "x2_hemorrhage", "x2_microaneurysm"))
    ].copy()
    if clinical.empty:
        ax.axis("off")
        return
    clinical["abs_rho"] = clinical["spearman_rho"].abs()
    clinical = clinical.sort_values("abs_rho", ascending=False).head(top_n).iloc[::-1]
    colors = np.where(clinical["spearman_rho"] >= 0, "#B3472E", "#2F6F9F")
    labels = [
        f"{short_label(a)}\n→ {short_label(b)}"
        for a, b in zip(clinical["lesion_feature"], clinical["biomarker"])
    ]
    y = np.arange(len(clinical))
    ax.barh(y, clinical["spearman_rho"], color=colors, height=0.58, alpha=0.9)
    ax.axvline(0, color="#1F2933", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.82, 0.82)
    ax.set_xlabel("Spearman rho", fontsize=8)
    ax.set_title("FDR-tested lesion-biomarker associations", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for yi, (label, row) in enumerate(zip(labels, clinical.itertuples(index=False))):
        x = row.spearman_rho
        label_x = -0.78 if x >= 0 else 0.78
        ax.text(
            label_x,
            yi,
            label,
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=6.8,
            color="#111827",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.4},
        )
        ax.text(
            x + (0.025 if x >= 0 else -0.025),
            yi,
            f"{x:+.2f}",
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="#111827",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean publisher figure for class-wise lesion/XAI interpretation.")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--linkage", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    examples = pd.read_csv(args.examples)
    features = pd.read_csv(args.features)
    linkage = pd.read_csv(args.linkage)
    feat_cols = [
        "id_code",
        "attention_pred_grade",
        "attn_x12_spatial",
        "attn_x34_jacobian",
        *[f"attention_prob_grade_{i}" for i in range(5)],
    ]
    examples = examples.merge(features[feat_cols], on="id_code", how="left")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess = DRXAIPreprocessor(image_size=args.image_size)

    n_rows = len(examples)
    fig = plt.figure(figsize=(18.4, 3.10 * n_rows + 1.65))
    gs = fig.add_gridspec(
        n_rows,
        8,
        width_ratios=[1.0, 1.0, 1.05, 0.82, 0.82, 0.82, 0.22, 1.85],
        wspace=0.12,
        hspace=0.26,
    )
    headers = [
        "Raw fundus",
        "DR-XAI green",
        "Combined XAI/CAM",
        "Hemorrhage",
        "Exudate",
        "Neovascularization",
    ]
    for c, title in enumerate(headers):
        ax = fig.add_subplot(gs[0, c])
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.axis("off")

    for r, row in enumerate(examples.itertuples(index=False)):
        raw_bgr = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
        raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        rgb = preprocess(row.image_path)
        maps = lesion_evidence_maps(rgb)
        total = norm01(np.stack([maps[k] for k in LESION_CHANNELS]).max(axis=0))
        panels = [
            raw,
            rgb,
            overlay(rgb, total),
            maps["hemorrhage"],
            maps["hard_exudate"],
            maps["neovascularization"],
        ]
        for c, panel in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            if c < 3:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap="magma", vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                pred = int(row.attention_pred_grade)
                true = int(row.diagnosis)
                correct = "correct" if pred == true else "wrong"
                prob = getattr(row, f"attention_prob_grade_{pred}")
                ax.text(
                    -0.08,
                    0.52,
                    f"Grade {true}\n{GRADE_NAMES.get(true, '')}\npred {pred} ({correct})\np={prob:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.2,
                    fontweight="bold",
                )
            if c == 5:
                ax.text(
                    0.02,
                    0.96,
                    "conservative NV\nfine/disordered vessels",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.8,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.42, "edgecolor": "none", "pad": 2},
                )

    ax_gap = fig.add_subplot(gs[:, 6])
    ax_gap.axis("off")
    ax_assoc = fig.add_subplot(gs[:, 7])
    plot_assoc(ax_assoc, linkage)

    fig.suptitle(
        "Representative correctly classified DR grades with lesion evidence and biomarker associations",
        fontsize=14.5,
        fontweight="bold",
        y=0.988,
    )
    fig.text(
        0.01,
        0.01,
        "X2 is lesion evidence maps/statistics; X3 is the 128-D image/lesion embedding. "
        "Lesion channels are weak DR-XAI-style evidence, not expert lesion segmentation. "
        "Association bars summarize FDR-corrected lesion-biomarker Spearman links.",
        fontsize=8.5,
        color="#374151",
    )
    png = out_dir / "dr_all_classes_lesion_xai_association_summary_clean.png"
    pdf = out_dir / "dr_all_classes_lesion_xai_association_summary_clean.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
