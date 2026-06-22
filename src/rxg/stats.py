from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    out = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        out[order[i]] = min(prev, 1.0)
    return out


def direction_label(rho: float) -> str:
    if rho > 0.05:
        return "increases_with_dr_grade"
    if rho < -0.05:
        return "decreases_with_dr_grade"
    return "weak_or_flat_grade_trend"


class HypothesisTester:
    """Kruskal-Wallis + Spearman + FDR testing for biomarkers and graph metrics."""

    def test_by_grade(self, df: pd.DataFrame, feature_cols: list[str], grade_col: str = "diagnosis") -> pd.DataFrame:
        labels = df[grade_col].astype(int)
        rows = []
        for col in feature_cols:
            y = pd.to_numeric(df[col], errors="coerce")
            groups = [y[labels == g].dropna().to_numpy() for g in sorted(labels.unique())]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2 or y.nunique(dropna=True) < 2:
                continue
            h, p = stats.kruskal(*groups)
            rho, rp = stats.spearmanr(labels, y, nan_policy="omit")
            rows.append({
                "feature": col,
                "kruskal_h": float(h),
                "p_value": float(p),
                "grade_spearman_rho": float(rho),
                "grade_spearman_p_value": float(rp),
            })
        out = pd.DataFrame(rows).sort_values("p_value")
        if not out.empty:
            out["fdr_p_value"] = benjamini_hochberg(out.p_value.to_numpy())
            out["grade_trend_direction"] = out.grade_spearman_rho.apply(direction_label)
            out["significant_fdr_0_05"] = out.fdr_p_value < 0.05
        return out

    def lesion_biomarker_linkage(
        self,
        df: pd.DataFrame,
        lesion_cols: list[str],
        biomarker_cols: list[str],
        min_abs_rho: float = 0.10,
    ) -> pd.DataFrame:
        rows = []
        for lesion in lesion_cols:
            lx = pd.to_numeric(df[lesion], errors="coerce")
            for biom in biomarker_cols:
                bx = pd.to_numeric(df[biom], errors="coerce")
                valid = lx.notna() & bx.notna()
                if valid.sum() < 20 or lx[valid].nunique() < 2 or bx[valid].nunique() < 2:
                    continue
                rho, p = stats.spearmanr(lx[valid], bx[valid])
                if abs(rho) >= min_abs_rho:
                    rows.append({"lesion_feature": lesion, "biomarker": biom, "spearman_rho": float(rho), "p_value": float(p), "n": int(valid.sum())})
        out = pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame()
        if not out.empty:
            out["fdr_p_value"] = benjamini_hochberg(out.p_value.to_numpy())
        return out
