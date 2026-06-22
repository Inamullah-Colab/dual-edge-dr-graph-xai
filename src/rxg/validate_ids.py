from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .automorph_paths import M2_MAP_FOLDERS


@dataclass
class IDCoverageReport:
    dataset_rows: int
    raw_image_matches: int
    x1_any_matches: int
    x4_matches: int
    optional_x2_matches: int | None = None
    optional_x3_matches: int | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class SameDatasetValidator:
    """Validates that raw images, X1, X4, and optional X2/X3 use the same id_code space."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.ids = pd.read_csv(cfg.labels_csv)[cfg.id_col].astype(str)
        self.id_set = set(self.ids)

    def image_ids(self) -> set[str]:
        root = Path(self.cfg.image_dir)
        ids = set()
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            ids.update(p.stem for p in root.glob(ext))
        return ids

    def x1_ids(self) -> set[str]:
        root = Path(self.cfg.automorph_results_dir)
        ids = set()
        for rel, _ in M2_MAP_FOLDERS.values():
            folder = root / rel
            if folder.exists():
                for p in folder.glob("*"):
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                        ids.add(p.stem)
        return ids

    def x4_ids(self) -> set[str]:
        path = Path(getattr(self.cfg, "macular_features_csv", "") or "")
        if not path.exists():
            return set()
        df = pd.read_csv(path, usecols=["Name"])
        return set(df["Name"].astype(str).str.replace(r"\.[^.]+$", "", regex=True))

    @staticmethod
    def table_ids(path: str | Path | None, id_col: str = "id_code") -> set[str] | None:
        if not path:
            return None
        path = Path(path)
        if not path.exists():
            return None
        df = pd.read_csv(path, usecols=[id_col])
        return set(df[id_col].astype(str))

    def validate(self, x2_csv: str | Path | None = None, x3_csv: str | Path | None = None) -> tuple[IDCoverageReport, dict[str, list[str]]]:
        raw = self.image_ids()
        x1 = self.x1_ids()
        x4 = self.x4_ids()
        x2 = self.table_ids(x2_csv)
        x3 = self.table_ids(x3_csv)
        report = IDCoverageReport(
            dataset_rows=len(self.ids),
            raw_image_matches=len(self.id_set & raw),
            x1_any_matches=len(self.id_set & x1),
            x4_matches=len(self.id_set & x4),
            optional_x2_matches=None if x2 is None else len(self.id_set & x2),
            optional_x3_matches=None if x3 is None else len(self.id_set & x3),
        )
        missing = {
            "raw_images_missing_first20": sorted(self.id_set - raw)[:20],
            "x1_missing_first20": sorted(self.id_set - x1)[:20],
            "x4_missing_first20": sorted(self.id_set - x4)[:20],
        }
        if x2 is not None:
            missing["x2_missing_first20"] = sorted(self.id_set - x2)[:20]
        if x3 is not None:
            missing["x3_missing_first20"] = sorted(self.id_set - x3)[:20]
        return report, missing
