from __future__ import annotations

# Warning: This code is for research and educational purposes only. Any clinical deployment requires IRB approval and prospective field validation.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rxg.dr_xai_evidence import (
    DRXAIPreprocessor,
    LESION_CHANNELS,
    lesion_evidence_maps,
    lesion_map_embedding,
    map_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build X2 lesion evidence maps/statistics for the spatial X12 branch"
    )
    parser.add_argument("--manifest", required=True, help="CSV with id_code, diagnosis, source_id, stream, image_path")
    parser.add_argument("--output", required=True, help="Output CSV containing x2_ lesion statistics and x2_map_embed_ columns")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--no-green", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    required = {"id_code", "diagnosis", "source_id", "stream", "image_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    preprocess = DRXAIPreprocessor(
        image_size=args.image_size,
        use_mask=not args.no_mask,
        use_clahe=not args.no_clahe,
        use_green=not args.no_green,
    )

    rows = []
    failed = []
    nv_area_035: list[float] = []
    other_area_035: list[float] = []

    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        try:
            rgb = preprocess(row.image_path)
            maps = lesion_evidence_maps(rgb)

            # X2 is lesion evidence, not the contrastive/image embedding.
            # These columns are used by X12 to mix lesion evidence with X1 vessel evidence.
            out = {
                "id_code": row.id_code,
                "diagnosis": int(row.diagnosis),
                "source_id": row.source_id,
                "stream": row.stream,
            }
            for name in LESION_CHANNELS:
                out.update(map_stats(maps[name], f"x2_{name}"))
                area = out[f"x2_{name}_area_035"]
                if name == "neovascularization":
                    nv_area_035.append(area)
                else:
                    other_area_035.append(area)

            stacked = np.stack([maps[n] for n in LESION_CHANNELS], axis=0)
            out["x2_total_evidence_mean"] = float(stacked.mean())
            out["x2_total_evidence_max"] = float(stacked.max())

            # x2_map_embed_* is a compact map descriptor for X12 spatial fusion only.
            # It should not be described as X3. X3 is produced separately.
            x2_map = lesion_map_embedding(maps, dim=128)
            out.update({f"x2_map_embed_{j:03d}": float(v) for j, v in enumerate(x2_map)})
            rows.append(out)
        except Exception as exc:
            failed.append({"id_code": getattr(row, "id_code", ""), "error": str(exc)})
        if i % 250 == 0:
            print(f"processed {i}/{len(manifest)}", flush=True)

    x2 = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    x2.to_csv(output, index=False)

    report = {
        "manifest_rows": int(len(manifest)),
        "x2_rows": int(len(x2)),
        "x2_feature_count": int(len([c for c in x2.columns if c.startswith("x2_")])),
        "x2_definition": "lesion evidence maps/statistics plus x2_map_embed descriptors for X12",
        "x2_is_not_x3": True,
        "lesion_channels": LESION_CHANNELS,
        "x2_map_embedding_dim": int(len([c for c in x2.columns if c.startswith("x2_map_embed_")])),
        "neovascularization_policy": "conservative fine/disordered-vessel proxy to reduce over-representation bias",
        "neovascularization_area_035_mean": float(np.mean(nv_area_035)) if nv_area_035 else None,
        "other_lesion_area_035_mean": float(np.mean(other_area_035)) if other_area_035 else None,
        "failed_rows": int(len(failed)),
        "output_csv": str(output),
        "failed_examples": failed[:10],
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
