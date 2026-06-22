from __future__ import annotations

# Warning: This code is for research and educational purposes only. Any clinical deployment requires IRB approval and prospective field validation.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import DataLoader

from rxg.config import PipelineConfig
from rxg.jacobian import X34JacobianBuilder
from rxg.graph import SimilarityGraphBuilder
from rxg.metrics import ClassificationEvaluator
from rxg.models import NumpyTensorDataset, TwoTokenAttentionClassifier
from rxg.stats import HypothesisTester


def prefixed_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if c.startswith(prefixes)]



def select_x2_map_columns(cols: list[str]) -> list[str]:
    """Prefer X2 image/map embedding columns for X12 spatial fusion."""
    map_cols = [c for c in cols if c.startswith("x2_map_embed_")]
    return map_cols if map_cols else cols


def select_x4_columns(cols: list[str], scope: str) -> list[str]:
    """Select X4 biomarkers for the X3->X4 Jacobian branch.

    `zone_b_c_no_knudtson` keeps localized macular zone B/C biomarkers and removes
    Knudtson formula columns. This avoids feeding the full verbose Macular_Features table
    into X34 when the experiment is focused on zone B/C biomarker sensitivity.
    """
    audit = {"x4_missing_fraction_before_impute", "x4_has_any_before_impute", "x4_source_file"}
    selected = [c for c in cols if c not in audit and not c.endswith("_source_file")]
    if scope == "all":
        return selected
    if scope == "zone_b_c":
        return [c for c in selected if c.endswith("_zone_b") or c.endswith("_zone_c")]
    if scope == "zone_b_c_no_knudtson":
        return [
            c for c in selected
            if (c.endswith("_zone_b") or c.endswith("_zone_c")) and "Knudtson" not in c
        ]
    raise ValueError(f"Unknown x4 scope: {scope}")


def select_x3_columns(cols: list[str]) -> list[str]:
    """Return the canonical 128-D X3 image/lesion embedding columns.

    X3 must be the image/lesion embedding used by the X34 Jacobian branch.
    The canonical release schema is x3_image_embed_000 ... x3_image_embed_127.
    Other x3_/embed_ columns are rejected to avoid accidentally feeding X2 map
    descriptors or unrelated features into the Jacobian branch.
    """
    canonical = [f"x3_image_embed_{i:03d}" for i in range(128)]
    if all(c in cols for c in canonical):
        return canonical
    raise ValueError(
        "X3 CSV must contain exactly the canonical 128-D columns "
        "x3_image_embed_000 ... x3_image_embed_127. "
        "If using Huang et al. lesion-based contrastive learning, export its 128-D "
        "features into this schema before running the graph builder."
    )

def numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def merge_required(base: pd.DataFrame, path: str | Path, prefixes: tuple[str, ...], name: str) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} CSV not found: {path}")
    df = pd.read_csv(path)
    if "id_code" not in df.columns:
        raise ValueError(f"{name} CSV must contain id_code")
    df["id_code"] = df["id_code"].astype(str)
    cols = prefixed_columns(df, prefixes)
    if not cols:
        raise ValueError(f"{name} CSV has no columns with prefixes {prefixes}")
    keep = ["id_code"] + cols
    merged = base.merge(df[keep], on="id_code", how="left", validate="one_to_one")
    missing_rows = int(merged[cols].isna().all(axis=1).sum())
    if missing_rows:
        raise ValueError(f"{name} does not match {missing_rows} rows from the matched manifest")
    return merged, cols


def build_x12(df: pd.DataFrame, x1_cols: list[str], x2_cols: list[str], dim: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    x1 = StandardScaler().fit_transform(numeric_matrix(df, x1_cols))
    x2 = StandardScaler().fit_transform(numeric_matrix(df, x2_cols))
    d = min(x1.shape[1], x2.shape[1], 64)
    x1r = PCA(n_components=d, random_state=seed).fit_transform(x1) if x1.shape[1] > d else x1[:, :d]
    x2r = PCA(n_components=d, random_state=seed).fit_transform(x2) if x2.shape[1] > d else x2[:, :d]
    mixed = np.concatenate([x1, x2, x1r * x2r, np.abs(x1r - x2r)], axis=1)
    n_comp = min(dim, mixed.shape[1], mixed.shape[0] - 1)
    emb = PCA(n_components=n_comp, random_state=seed).fit_transform(mixed)
    emb = normalize(StandardScaler().fit_transform(emb)).astype(np.float32)
    if emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1]))).astype(np.float32)
    cols = [f"x12_{i:03d}" for i in range(dim)]
    return pd.DataFrame(emb[:, :dim], columns=cols), emb[:, :dim]


def build_x34(df: pd.DataFrame, x3_cols: list[str], x4_cols: list[str], cfg: PipelineConfig, epochs: int) -> tuple[pd.DataFrame, np.ndarray]:
    x3 = numeric_matrix(df, x3_cols)
    x4 = numeric_matrix(df, x4_cols)
    builder = X34JacobianBuilder(x3_dim=cfg.x3_embedding_dim, x4_dim=cfg.x4_reduced_dim, random_state=cfg.random_state)
    x3r, x4r = builder.reduce(x3, x4)
    builder.fit_mapper(x3r, x4r, epochs=epochs, batch_size=cfg.batch_size)
    frob, diag, out_sens, in_sens = builder.jacobian_summary(x3r)
    emb = np.concatenate([x3r, x4r, frob[:, None], diag[:, None], out_sens, in_sens[:, :32]], axis=1)
    emb = StandardScaler().fit_transform(emb).astype(np.float32)
    cols = [f"x34_{i:03d}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, columns=cols), emb


def train_attention(df: pd.DataFrame, x12: np.ndarray, x34: np.ndarray, cfg: PipelineConfig, epochs: int) -> tuple[pd.DataFrame, np.ndarray, dict]:
    y = df["diagnosis"].astype(int).to_numpy()
    groups = df[cfg.group_col] if cfg.group_col in df.columns else df["source_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=cfg.random_state)
    tr, te = next(splitter.split(np.arange(len(df)), y, groups.astype(str)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTokenAttentionClassifier(x12.shape[1], x34.shape[1], hidden_dim=cfg.hidden_dim, n_classes=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(NumpyTensorDataset(x12[tr], x34[tr], y[tr]), batch_size=cfg.batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb12, xb34, yb in train_loader:
            xb12, xb34, yb = xb12.to(device), xb34.to(device), yb.to(device)
            logits, _, _ = model(xb12, xb34)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits, attn, fused = model(
            torch.tensor(x12, dtype=torch.float32, device=device),
            torch.tensor(x34, dtype=torch.float32, device=device),
        )
        prob = F.softmax(logits, dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
        attn_np = attn.cpu().numpy()
        fused_np = fused.cpu().numpy().astype(np.float32)
    evaluator = ClassificationEvaluator(n_classes=5)
    metrics = evaluator.evaluate(y[te], pred[te], prob[te])
    metrics.update({"model": "x12_x34_attention", "n_train": int(len(tr)), "n_test": int(len(te)), "n_features": int(x12.shape[1] + x34.shape[1])})
    out = pd.DataFrame({
        "attn_x12_spatial": attn_np[:, 0],
        "attn_x34_jacobian": attn_np[:, 1],
        "attention_pred_grade": pred,
    })
    for i in range(prob.shape[1]):
        out[f"attention_prob_grade_{i}"] = prob[:, i]
    return out, fused_np, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Spatial-Jacobian Attention Graph from matched X1, X2, X3, and X4 evidence files")
    parser.add_argument("--config", default="configs/non_augmented.yaml")
    parser.add_argument("--base-x1-x4", required=True, help="Matched non-augmented X1+X4 CSV")
    parser.add_argument("--x2-csv", required=True, help="Matched X2 lesion/XAI feature CSV with id_code and x2_ columns")
    parser.add_argument("--x3-csv", required=True, help="Matched X3 embedding CSV with id_code and x3_ or embed_ columns")
    parser.add_argument("--output", default=None, help="Final full feature CSV")
    parser.add_argument("--jacobian-epochs", type=int, default=40)
    parser.add_argument("--attention-epochs", type=int, default=80)
    parser.add_argument("--x4-scope", choices=["zone_b_c_no_knudtson", "zone_b_c", "all"], default="zone_b_c_no_knudtson")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    out_dir = cfg.out / "full_fusion_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.base_x1_x4)
    base["id_code"] = base["id_code"].astype(str)
    if base["id_code"].duplicated().any():
        raise ValueError("base X1/X4 table has duplicate id_code values")
    df, x2_cols_all = merge_required(base, args.x2_csv, ("x2_",), "X2")
    df, x3_cols_all = merge_required(df, args.x3_csv, ("x3_", "embed_"), "X3")
    x3_cols = select_x3_columns(x3_cols_all)
    x1_cols = prefixed_columns(df, ("x1_",))
    x2_cols = select_x2_map_columns(x2_cols_all)
    x4_cols_all = prefixed_columns(df, ("x4_macula__", "x4_"))
    x4_cols = select_x4_columns(x4_cols_all, args.x4_scope)
    if not x1_cols or not x4_cols:
        raise ValueError("base X1/X4 table must contain x1_ and selected x4_ biomarker columns")
    unselected_x4_cols = [c for c in x4_cols_all if c not in x4_cols and c not in {"x4_missing_fraction_before_impute", "x4_has_any_before_impute"}]
    if unselected_x4_cols:
        df = df.drop(columns=unselected_x4_cols)

    x12_df, x12 = build_x12(df, x1_cols, x2_cols, cfg.x12_embedding_dim, cfg.random_state)
    x34_df, x34 = build_x34(df, x3_cols, x4_cols, cfg, args.jacobian_epochs)
    attn_df, fused, attn_metrics = train_attention(df, x12, x34, cfg, args.attention_epochs)

    fused_cols = [f"fused_attention_{i:03d}" for i in range(fused.shape[1])]
    fused_df = pd.DataFrame(fused, columns=fused_cols)
    full = pd.concat([df.reset_index(drop=True), x12_df, x34_df, attn_df, fused_df], axis=1)

    graph_result = SimilarityGraphBuilder(k=cfg.graph_k).build(full[["id_code", "source_id", "diagnosis"]], x12, x34, fused)
    SimilarityGraphBuilder.save(graph_result, out_dir, prefix="x12_x34_attention_graph")
    full = full.merge(graph_result.nodes.drop(columns=["source_id", "diagnosis"]), on="id_code", how="left", validate="one_to_one")

    output = Path(args.output) if args.output else out_dir / "x1_x2_x3_x4_x12_x34_attention_graph_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(output, index=False)
    pd.DataFrame([attn_metrics]).to_csv(out_dir / "attention_metrics.csv", index=False)

    tester = HypothesisTester()
    feature_cols = [
        c for c in full.columns
        if c.startswith(("x1_", "x2_", "x3_", "x4_", "x12_", "x34_", "graph_"))
    ]
    tests = tester.test_by_grade(full, feature_cols)
    tests.to_csv(out_dir / "hypothesis_tests_all_features.csv", index=False)
    lesion_cols = [c for c in full.columns if c.startswith("x2_")]
    biomarker_cols = [c for c in full.columns if c.startswith("x4_")]
    linkage = tester.lesion_biomarker_linkage(full, lesion_cols, biomarker_cols)
    linkage.to_csv(out_dir / "lesion_biomarker_linkage.csv", index=False)

    report = {
        "rows": int(len(full)),
        "x1_features": len(x1_cols),
        "x2_features_used_for_x12": len(x2_cols),
        "x2_features_total": len(x2_cols_all),
        "x3_features": len(x3_cols),
        "x3_schema": "x3_image_embed_000..x3_image_embed_127",
        "x4_features_used_for_x34": len(x4_cols),
        "x4_features_total": len(x4_cols_all),
        "x4_scope": args.x4_scope,
        "x12_features": int(x12.shape[1]),
        "x34_features": int(x34.shape[1]),
        "graph_nodes": int(len(graph_result.nodes)),
        "graph_edges": int(len(graph_result.edges)),
        "attention_metrics": attn_metrics,
        "hypothesis_tests": int(len(tests)),
        "lesion_biomarker_links": int(len(linkage)),
        "output_csv": str(output),
    }
    (out_dir / "full_fusion_graph.report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
