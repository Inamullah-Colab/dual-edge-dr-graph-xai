from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rxg.config import PipelineConfig
from rxg.validate_ids import SameDatasetValidator

parser = argparse.ArgumentParser(description="Build strict non-augmented manifest for rows present in raw images, X1, X4, and optional X2/X3")
parser.add_argument("--config", default="configs/non_augmented.yaml")
parser.add_argument("--x2-csv", default=None)
parser.add_argument("--x3-csv", default=None)
args = parser.parse_args()

cfg = PipelineConfig.from_yaml(args.config)
labels = pd.read_csv(cfg.labels_csv).rename(columns={cfg.id_col: "id_code", cfg.label_col: "diagnosis"})
labels["id_code"] = labels["id_code"].astype(str)
validator = SameDatasetValidator(cfg)
ids = set(labels["id_code"])
raw = validator.image_ids()
x1 = validator.x1_ids()
x4 = validator.x4_ids()
keep = ids & raw & x1 & x4
x2 = validator.table_ids(args.x2_csv)
x3 = validator.table_ids(args.x3_csv)
if x2 is not None:
    keep &= x2
if x3 is not None:
    keep &= x3
out = labels[labels["id_code"].isin(keep)].copy().sort_values("id_code")
out["source_id"] = out["id_code"]
out["stream"] = "original"
out["image_path"] = out["id_code"].map(lambda x: str(Path(cfg.image_dir) / f"{x}.png"))
out_dir = Path(cfg.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "non_augmented_strict_intersection_manifest.csv"
out.to_csv(out_csv, index=False)
summary = {
    "labels_rows": int(len(labels)),
    "raw_image_matches": int(len(ids & raw)),
    "x1_matches": int(len(ids & x1)),
    "x4_matches": int(len(ids & x4)),
    "x2_matches": None if x2 is None else int(len(ids & x2)),
    "x3_matches": None if x3 is None else int(len(ids & x3)),
    "strict_intersection_rows": int(len(out)),
    "output_csv": str(out_csv),
}
(out_dir / "non_augmented_strict_intersection_manifest.report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
