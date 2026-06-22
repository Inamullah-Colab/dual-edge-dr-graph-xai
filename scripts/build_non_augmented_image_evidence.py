from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rxg.dr_xai_evidence import (
    DRXAIPreprocessor,
    LESION_CHANNELS,
    image_embedding,
    lesion_evidence_maps,
    lesion_map_embedding,
    map_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build matched DR-XAI-style X2 lesion evidence and X3 128-D image embeddings"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--x2-output", required=True)
    parser.add_argument("--x3-output", required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--no-green", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    preprocess = DRXAIPreprocessor(
        image_size=args.image_size,
        use_mask=not args.no_mask,
        use_clahe=not args.no_clahe,
        use_green=not args.no_green,
    )
    rows_x2 = []
    rows_x3 = []
    failed = []
    nv_area_035: list[float] = []
    other_area_035: list[float] = []

    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        try:
            rgb = preprocess(row.image_path)
            maps = lesion_evidence_maps(rgb)
            x2 = {
                "id_code": row.id_code,
                "diagnosis": int(row.diagnosis),
                "source_id": row.source_id,
                "stream": row.stream,
            }
            for name in LESION_CHANNELS:
                x2.update(map_stats(maps[name], f"x2_{name}"))
                area = x2[f"x2_{name}_area_035"]
                if name == "neovascularization":
                    nv_area_035.append(area)
                else:
                    other_area_035.append(area)

            stacked = np.stack([maps[n] for n in LESION_CHANNELS], axis=0)
            x2["x2_total_evidence_mean"] = float(stacked.mean())
            x2["x2_total_evidence_max"] = float(stacked.max())
            x2_emb = lesion_map_embedding(maps, dim=128)
            x2.update({f"x2_map_embed_{j:03d}": float(v) for j, v in enumerate(x2_emb)})
            rows_x2.append(x2)

            emb = image_embedding(rgb, dim=128)
            x3 = {
                "id_code": row.id_code,
                "diagnosis": int(row.diagnosis),
                "source_id": row.source_id,
                "stream": row.stream,
            }
            x3.update({f"x3_image_embed_{j:03d}": float(v) for j, v in enumerate(emb)})
            rows_x3.append(x3)
        except Exception as exc:
            failed.append({"id_code": getattr(row, "id_code", ""), "error": str(exc)})
        if i % 250 == 0:
            print(f"processed {i}/{len(manifest)}", flush=True)

    x2 = pd.DataFrame(rows_x2)
    x3 = pd.DataFrame(rows_x3)
    x2_out = Path(args.x2_output)
    x3_out = Path(args.x3_output)
    x2_out.parent.mkdir(parents=True, exist_ok=True)
    x3_out.parent.mkdir(parents=True, exist_ok=True)
    x2.to_csv(x2_out, index=False)
    x3.to_csv(x3_out, index=False)

    report = {
        "manifest_rows": int(len(manifest)),
        "x2_rows": int(len(x2)),
        "x3_rows": int(len(x3)),
        "x2_feature_count": int(len([c for c in x2.columns if c.startswith("x2_")])),
        "x3_feature_count": int(len([c for c in x3.columns if c.startswith("x3_")])),
        "x2_includes_neovascularization": "x2_neovascularization_mean" in x2.columns,
        "x2_map_embedding_dim": int(len([c for c in x2.columns if c.startswith("x2_map_embed_")])),
        "x2_map_embedding_method": "balanced five-lesion-channel 145D raw vector projected deterministically to 128D",
        "x2_source": "DR-XAI-style fundus preprocessing plus weak lesion-evidence maps; use trained DR-XAI Grad-CAM++ checkpoint when available",
        "preprocessing": {
            "image_size": args.image_size,
            "retina_mask": not args.no_mask,
            "clahe": not args.no_clahe,
            "green_channel_emphasis": not args.no_green,
        },
        "neovascularization_policy": "conservative fine/disordered-vessel proxy; rare channel retained but sparsified to reduce over-representation bias",
        "neovascularization_area_035_mean": float(np.mean(nv_area_035)) if nv_area_035 else None,
        "other_lesion_area_035_mean": float(np.mean(other_area_035)) if other_area_035 else None,
        "failed_rows": int(len(failed)),
        "x2_output": str(x2_out),
        "x3_output": str(x3_out),
        "failed_examples": failed[:10],
    }
    x2_out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    x3_out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
