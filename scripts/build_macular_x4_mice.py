from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from rxg.config import PipelineConfig

parser = argparse.ArgumentParser(description="Build X4 from full Macular_Features.csv with negative-to-NaN and MICE imputation")
parser.add_argument("--config", default="configs/non_augmented.yaml")
parser.add_argument("--restrict-manifest", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

cfg = PipelineConfig.from_yaml(args.config)
restrict = pd.read_csv(args.restrict_manifest)
ids = restrict[["id_code", "source_id", "stream", "diagnosis"]].copy()
ids["id_code"] = ids["id_code"].astype(str)
path = Path(cfg.macular_features_csv)
if not path.exists():
    raise FileNotFoundError(f"Macular_Features.csv not found: {path}")
raw_df = pd.read_csv(path)
if "Name" not in raw_df.columns:
    raise ValueError("Macular_Features.csv must contain Name")
raw_df["id_code"] = raw_df["Name"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
raw_df = ids.merge(raw_df.drop(columns=["Name"]), on="id_code", how="left")
feature_cols = [c for c in raw_df.columns if c not in {"id_code", "source_id", "stream", "diagnosis"}]
raw = raw_df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
negative_values = int((raw < 0).sum().sum())
missing_before = int(raw.isna().sum().sum())
raw = raw.mask(raw < 0, np.nan)
missing_after_negative = int(raw.isna().sum().sum())
missing_fraction = raw.isna().mean(axis=1)
has_any = raw.notna().any(axis=1).astype(int)
imputer = IterativeImputer(max_iter=15, random_state=42, sample_posterior=False)
imputed = pd.DataFrame(imputer.fit_transform(raw), columns=[f"x4_macula__{c}" for c in feature_cols])
post_impute_negative_values = int((imputed < 0).sum().sum())
imputed = imputed.clip(lower=0.0)
out_df = pd.concat([ids.reset_index(drop=True), imputed.reset_index(drop=True)], axis=1)
out_df["x4_missing_fraction_before_impute"] = missing_fraction.to_numpy()
out_df["x4_has_any_before_impute"] = has_any.to_numpy()
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out, index=False)
report = {
    "rows": int(len(out_df)),
    "x4_feature_count": int(len([c for c in out_df.columns if c.startswith("x4_macula__")])),
    "negative_values_replaced": negative_values,
    "post_impute_negative_values_clipped": post_impute_negative_values,
    "missing_before_impute": missing_before,
    "missing_after_negative_to_nan": missing_after_negative,
    "imputation_method": "mice_iterative_imputer_max_iter_15",
    "source_file": str(path),
    "output_csv": str(out),
}
out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
