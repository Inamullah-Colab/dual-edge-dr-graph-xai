from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    """Configuration shared by the command-line pipeline.

    The defaults are intentionally conservative and point to project-local outputs.
    Override paths in YAML when transferring the code to a new machine or GitHub runner.
    """

    labels_csv: str = "/home/i1n23/datasets/APTOS2019_balanced_augmented/train.csv"
    image_dir: str = "/home/i1n23/datasets/APTOS2019_balanced_augmented/images"
    automorph_results_dir: str = "/home/i1n23/automorph_augmented_data/Results"
    mice_biomarker_manifest: str = "/home/i1n23/retina_graph_dr/outputs/raw_image_biomarker_attention/full_augmented_mice_biomarker_manifest.csv"
    macular_features_csv: str = ""
    output_dir: str = "/home/i1n23/retina_graph_dr/github_ready_xai_graph_dr/outputs"
    id_col: str = "id_code"
    label_col: str = "diagnosis"
    group_col: str = "source_id"
    image_size: int = 224
    x12_embedding_dim: int = 128
    x3_embedding_dim: int = 128
    x4_reduced_dim: int = 32
    x34_embedding_dim: int = 226
    graph_k: int = 8
    random_state: int = 42
    batch_size: int = 192
    hidden_dim: int = 192
    lesion_classes: list[str] = field(default_factory=lambda: [
        "microaneurysm",
        "hemorrhage",
        "hard_exudate",
        "cotton_wool_spot",
        "neovascularization",
    ])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @property
    def out(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
