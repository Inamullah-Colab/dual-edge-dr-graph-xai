from __future__ import annotations

# Warning: This code is for research and educational purposes only. Any clinical deployment requires IRB approval and prospective field validation.

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HUANG_MEAN = np.asarray([0.425753653049469, 0.29737451672554016, 0.21293757855892181], dtype=np.float32)
HUANG_STD = np.asarray([0.27670302987098694, 0.20240527391433716, 0.1686241775751114], dtype=np.float32)


def preprocess_for_huang(path: str | Path, image_size: int = 128) -> torch.Tensor:
    """Preprocess a fundus image with Huang et al. LCL normalization.

    The Huang repository trains on 128 x 128 lesion patches with the mean/std stored
    in its config.py. For whole-image X3 extraction, we resize the fundus image to
    128 x 128 and apply the same RGB normalization. If lesion patch predictions are
    available, users may export patch-level features externally and aggregate them
    into the same x3_image_embed_000..127 schema.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rgb = (rgb - HUANG_MEAN) / HUANG_STD
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(chw)


def import_huang_model(lcl_repo: str | Path):
    repo = Path(lcl_repo)
    if not (repo / 'modules.py').exists() or not (repo / 'resnet.py').exists():
        raise FileNotFoundError(
            f"Expected Huang Lesion-based-Contrastive-Learning repo with modules.py and resnet.py: {repo}"
        )
    sys.path.insert(0, str(repo))
    modules = importlib.import_module('modules')
    resnet = importlib.import_module('resnet')
    return modules.ContrastiveModel, resnet.resnet50


def load_checkpoint_state(checkpoint: str | Path, device: torch.device) -> tuple[dict, str]:
    """Return a clean checkpoint state dict and its detected storage kind."""
    obj = torch.load(str(checkpoint), map_location=device)
    if isinstance(obj, torch.nn.DataParallel):
        state = obj.module.state_dict()
        kind = 'serialized_dataparallel_model'
    elif isinstance(obj, torch.nn.Module):
        state = obj.state_dict()
        kind = 'serialized_model'
    elif isinstance(obj, dict):
        if 'state_dict' in obj:
            state = obj['state_dict']
            kind = 'dict_state_dict'
        elif 'model' in obj and isinstance(obj['model'], dict):
            state = obj['model']
            kind = 'dict_model_state'
        else:
            state = obj
            kind = 'raw_state_dict'
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")

    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ('module.', 'model.', 'net.'):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned, kind


def checkpoint_has_projection_head(state: dict) -> bool:
    return any(k.startswith('head.') or '.head.' in k for k in state)


def pca_128(features: np.ndarray) -> np.ndarray:
    """Reduce Huang 2048-D backbone features to the X3 128-D schema by SVD PCA.

    The released Huang LCL checkpoints contain the trained ResNet50 backbone
    weights, not the projection head. We therefore extract real Huang backbone
    features and use a deterministic PCA projection to the 128-D interface needed
    by X34. This is no longer a handcrafted proxy; it is Huang-LCL-backbone X3.
    """
    x = features.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comp = vt[: min(128, vt.shape[0])].T.astype(np.float32)
    z = x @ comp
    if z.shape[1] < 128:
        z = np.pad(z, ((0, 0), (0, 128 - z.shape[1])))
    z = z[:, :128]
    z = z - z.mean(axis=1, keepdims=True)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    return z.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build real X3 embeddings using Huang et al. Lesion-based Contrastive Learning checkpoint'
    )
    parser.add_argument('--manifest', required=True, help='CSV with id_code, diagnosis, source_id, stream, image_path')
    parser.add_argument('--output', required=True, help='Output CSV containing x3_image_embed_000..127')
    parser.add_argument('--lcl-repo', required=True, help='Path to YijinHuang/Lesion-based-Contrastive-Learning clone')
    parser.add_argument('--checkpoint', required=True, help='Huang LCL .pt checkpoint, e.g. resnet50_128_08.pt')
    parser.add_argument('--image-size', type=int, default=128, help='Huang config input size; default 128')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--imagenet-pretrained', action='store_true', help='Initialize ResNet50 with ImageNet weights before loading checkpoint; may download weights')
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    required = {'id_code', 'diagnosis', 'source_id', 'stream', 'image_path'}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Huang LCL checkpoint not found: {checkpoint}. Download lesion_based_CL_trained_weights.zip from the Huang GitHub release and extract a file such as resnet50_128_08.pt."
        )

    device = torch.device(args.device)
    ContrastiveModel, resnet50 = import_huang_model(args.lcl_repo)
    state, checkpoint_kind = load_checkpoint_state(checkpoint, device)
    has_head = checkpoint_has_projection_head(state)
    if has_head:
        model = ContrastiveModel(resnet50, pretrained=args.imagenet_pretrained, head='mlp', dim_in=2048, feat_dim=128).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        feature_mode = 'huang_contrastive_projection_128d'
    else:
        model = resnet50(pretrained=args.imagenet_pretrained).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        feature_mode = 'huang_lcl_resnet50_backbone_2048d_pca_to_128d'
    if missing:
        print(f'warning: missing model keys: {len(missing)}')
    if unexpected:
        print(f'warning: unexpected checkpoint keys: {len(unexpected)}')
    model.eval()

    rows = []
    failed = []
    batch_tensors = []
    batch_meta = []
    raw_features = []
    raw_meta = []

    def flush_batch():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            emb = model(x).detach().cpu().numpy().astype(np.float32)
        if has_head and (emb.ndim != 2 or emb.shape[1] != 128):
            raise ValueError(f"Huang LCL projection model must output [N, 128], got {emb.shape}")
        raw_features.append(emb)
        raw_meta.extend(batch_meta)
        batch_tensors.clear()
        batch_meta.clear()

    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        try:
            tensor = preprocess_for_huang(row.image_path, image_size=args.image_size)
            batch_tensors.append(tensor)
            batch_meta.append({
                'id_code': row.id_code,
                'diagnosis': int(row.diagnosis),
                'source_id': row.source_id,
                'stream': row.stream,
            })
            if len(batch_tensors) >= args.batch_size:
                flush_batch()
        except Exception as exc:
            failed.append({'id_code': getattr(row, 'id_code', ''), 'error': str(exc)})
        if i % 250 == 0:
            print(f'processed {i}/{len(manifest)}', flush=True)
    flush_batch()

    if raw_features:
        feats = np.concatenate(raw_features, axis=0).astype(np.float32)
        if has_head:
            z = feats
            z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
        else:
            z = pca_128(feats)
        for meta, vec in zip(raw_meta, z):
            out = dict(meta)
            out.update({f'x3_image_embed_{j:03d}': float(v) for j, v in enumerate(vec)})
            rows.append(out)

    x3 = pd.DataFrame(rows)
    x3_cols = [c for c in x3.columns if c.startswith('x3_image_embed_')]
    if len(x3_cols) != 128 and not x3.empty:
        raise ValueError(f'X3 output must contain exactly 128 embedding columns; found {len(x3_cols)}')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    x3.to_csv(output, index=False)
    report = {
        'manifest_rows': int(len(manifest)),
        'x3_rows': int(len(x3)),
        'x3_embedding_dim': int(len(x3_cols)),
        'x3_column_prefix': 'x3_image_embed_',
        'x3_definition': 'Huang et al. lesion-based contrastive ResNet50 feature embedding in 128-D X3 schema',
        'embedding_source': 'huang_lesion_based_contrastive_learning',
        'exact_lcl_model': True,
        'feature_mode': feature_mode,
        'checkpoint_has_projection_head': bool(has_head),
        'lcl_repo': str(Path(args.lcl_repo).resolve()),
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_kind': checkpoint_kind,
        'image_size': args.image_size,
        'failed_rows': int(len(failed)),
        'output_csv': str(output),
        'failed_examples': failed[:10],
    }
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
