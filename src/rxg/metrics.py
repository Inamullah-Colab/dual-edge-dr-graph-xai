from __future__ import annotations

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


class ClassificationEvaluator:
    """Common metric suite for each stage: X1, X2, X3, X4, X12, X34, and fused."""

    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "mean_absolute_grade_error": float(mean_absolute_error(y_true, y_pred)),
            "adjacent_grade_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        }
        if y_prob is not None:
            onehot = np.eye(self.n_classes)[y_true.astype(int)]
            result["macro_auc_ovr"] = float(roc_auc_score(onehot, y_prob, average="macro", multi_class="ovr"))
            result["weighted_auc_ovr"] = float(roc_auc_score(onehot, y_prob, average="weighted", multi_class="ovr"))
        return result

    def per_class(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(self.n_classes)), zero_division=0
        )
        return pd.DataFrame({"grade": range(self.n_classes), "precision": p, "recall": r, "f1": f, "support": s})

    def confusion(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=list(range(self.n_classes))))


def binary_threshold_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, target_spec: float = 0.95) -> dict:
    """DR-XAI-style screening metric for referable/non-referable experiments."""
    thresholds = np.linspace(0, 1, 1001)
    best = {"threshold": 0.5, "specificity": 0.0, "sensitivity": 0.0, "diff": float("inf")}
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)
        diff = abs(spec - target_spec)
        if diff < best["diff"]:
            best = {"threshold": float(t), "specificity": float(spec), "sensitivity": float(sens), "diff": float(diff)}
    best["auroc"] = float(roc_auc_score(y_true, y_prob))
    best["auprc"] = float(average_precision_score(y_true, y_prob))
    return best
