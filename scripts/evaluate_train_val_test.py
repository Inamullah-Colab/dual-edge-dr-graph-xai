from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

STAGES = [
    ("x1_vessel", ("x1_",)),
    ("x2_xai_lesion", ("x2_",)),
    ("x3_embedding", ("x3_", "embed_")),
    ("x4_biomarker", ("x4_", "m3_")),
    ("x12_spatial", ("x12_",)),
    ("x34_jacobian", ("x34_",)),
    ("x1234_all", ("x1_", "x2_", "x3_", "embed_", "x4_", "m3_", "x12_", "x34_")),
]


def cols_for(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if c.startswith(prefixes)]


def matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)


def split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.40, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, stratify=y[temp_idx])
    return train_idx, val_idx, test_idx


def five_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray | None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae_grade": float(mean_absolute_error(y_true, y_pred)),
        "adjacent_grade_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
    }
    if prob is not None and len(np.unique(y_true)) == 5:
        onehot = np.eye(5)[y_true.astype(int)]
        out["macro_auc_ovr"] = float(roc_auc_score(onehot, prob, average="macro", multi_class="ovr"))
        out["weighted_auc_ovr"] = float(roc_auc_score(onehot, prob, average="weighted", multi_class="ovr"))
    return out


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision_ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_predict(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray, seed: int):
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed, early_stopping=True)),
    ])
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[eval_idx])
    prob = clf.predict_proba(x[eval_idx]) if hasattr(clf, "predict_proba") else None
    return pred, prob


def save_confusion(out_dir: Path, task: str, stage: str, split: str, y_true: np.ndarray, pred: np.ndarray, labels: list[int]) -> None:
    d = out_dir / task / stage
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(confusion_matrix(y_true, pred, labels=labels)).to_csv(d / f"{split}_confusion_matrix.csv", index=False)
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": s}).to_csv(d / f"{split}_per_class_metrics.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate five-class and binary DR with train/val/test splits")
    ap.add_argument("--feature-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.feature_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y5 = df["diagnosis"].astype(int).to_numpy()
    train5, val5, test5 = split_indices(y5, args.seed)
    yb = (y5 >= 2).astype(int)
    trainb, valb, testb = split_indices(yb, args.seed)

    split_report = {
        "rows": int(len(df)),
        "five_class": {
            "definition": "grades 0,1,2,3,4",
            "train": int(len(train5)), "val": int(len(val5)), "test": int(len(test5)),
            "train_counts": pd.Series(y5[train5]).value_counts().sort_index().astype(int).to_dict(),
            "val_counts": pd.Series(y5[val5]).value_counts().sort_index().astype(int).to_dict(),
            "test_counts": pd.Series(y5[test5]).value_counts().sort_index().astype(int).to_dict(),
        },
        "binary_referable": {
            "definition": "non_referable=0-1; referable=2-4",
            "train": int(len(trainb)), "val": int(len(valb)), "test": int(len(testb)),
            "train_counts": pd.Series(yb[trainb]).value_counts().sort_index().astype(int).to_dict(),
            "val_counts": pd.Series(yb[valb]).value_counts().sort_index().astype(int).to_dict(),
            "test_counts": pd.Series(yb[testb]).value_counts().sort_index().astype(int).to_dict(),
        },
    }
    (out_dir / "train_val_test_split_report.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")

    rows5, rowsb = [], []
    for stage, prefixes in STAGES:
        cols = cols_for(df, prefixes)
        if not cols:
            continue
        x = matrix(df, cols)
        for split_name, eval_idx in [("val", val5), ("test", test5)]:
            pred, prob = fit_predict(x, y5, train5, eval_idx, args.seed)
            m = five_metrics(y5[eval_idx], pred, prob)
            m.update({"task": "five_class", "split": split_name, "model": stage, "n_features": len(cols), "n_train": len(train5), "n_eval": len(eval_idx)})
            rows5.append(m)
            save_confusion(out_dir, "five_class", stage, split_name, y5[eval_idx], pred, [0, 1, 2, 3, 4])
        for split_name, eval_idx in [("val", valb), ("test", testb)]:
            pred, prob = fit_predict(x, yb, trainb, eval_idx, args.seed)
            p1 = prob[:, 1]
            m = binary_metrics(yb[eval_idx], pred, p1)
            m.update({"task": "binary_referable", "split": split_name, "model": stage, "n_features": len(cols), "n_train": len(trainb), "n_eval": len(eval_idx)})
            rowsb.append(m)
            save_confusion(out_dir, "binary_referable", stage, split_name, yb[eval_idx], pred, [0, 1])

    five = pd.DataFrame(rows5)
    binary = pd.DataFrame(rowsb)
    five.to_csv(out_dir / "five_class_train_val_test_summary.csv", index=False)
    binary.to_csv(out_dir / "binary_referable_train_val_test_summary.csv", index=False)
    print("FIVE CLASS")
    print(five.to_string(index=False))
    print("\nBINARY REFERABLE")
    print(binary.to_string(index=False))
    print(json.dumps(split_report, indent=2))


if __name__ == "__main__":
    main()
