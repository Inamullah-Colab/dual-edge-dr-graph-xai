from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rxg.dr_xai_evidence import DRXAIPreprocessor, image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Build X3 128-D image/lesion embeddings from a matched fundus manifest")
    parser.add_argument("--manifest", required=True, help="CSV with id_code, diagnosis, source_id, stream, image_path")
    parser.add_argument("--output", required=True, help="Output CSV containing x3_image_embed_000..127")
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
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        try:
            rgb = preprocess(row.image_path)
            emb = image_embedding(rgb, dim=128)
            if emb.shape != (128,):
                raise ValueError(f"Expected 128-D X3 embedding, got shape {emb.shape}")
            out = {
                "id_code": row.id_code,
                "diagnosis": int(row.diagnosis),
                "source_id": row.source_id,
                "stream": row.stream,
            }
            out.update({f"x3_image_embed_{j:03d}": float(v) for j, v in enumerate(emb)})
            rows.append(out)
        except Exception as exc:
            failed.append({"id_code": getattr(row, "id_code", ""), "error": str(exc)})
        if i % 250 == 0:
            print(f"processed {i}/{len(manifest)}", flush=True)

    x3 = pd.DataFrame(rows)
    x3_cols = [c for c in x3.columns if c.startswith("x3_image_embed_")]
    if len(x3_cols) != 128 and not x3.empty:
        raise ValueError(f"X3 output must contain exactly 128 embedding columns; found {len(x3_cols)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    x3.to_csv(output, index=False)
    report = {
        "manifest_rows": int(len(manifest)),
        "x3_rows": int(len(x3)),
        "x3_embedding_dim": int(len(x3_cols)),
        "x3_column_prefix": "x3_image_embed_",
        "x3_definition": "128-D image/lesion embedding used by the X34 Jacobian branch",
        "failed_rows": int(len(failed)),
        "output_csv": str(output),
        "failed_examples": failed[:10],
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
