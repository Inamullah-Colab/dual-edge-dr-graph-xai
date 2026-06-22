from __future__ import annotations

import argparse
import json
from pathlib import Path

from rxg.config import PipelineConfig
from rxg.validate_ids import SameDatasetValidator

parser = argparse.ArgumentParser(description="Validate raw images, X1, X4, and optional X2/X3 share the same non-augmented id_code set")
parser.add_argument("--config", default="configs/non_augmented.yaml")
parser.add_argument("--x2-csv", default=None)
parser.add_argument("--x3-csv", default=None)
args = parser.parse_args()

cfg = PipelineConfig.from_yaml(args.config)
report, missing = SameDatasetValidator(cfg).validate(args.x2_csv, args.x3_csv)
out = Path(cfg.output_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "same_dataset_validation.json").write_text(json.dumps({"report": report.to_dict(), "missing": missing}, indent=2), encoding="utf-8")
print(json.dumps({"report": report.to_dict(), "missing": missing}, indent=2))
