from __future__ import annotations

import cv2
import numpy as np

from rxg.dr_xai_evidence import (
    DRXAIPreprocessor,
    LESION_CHANNELS,
    lesion_evidence_maps,
    lesion_map_embedding,
)


def synthetic_fundus(size: int = 160) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 8, (96, 58, 36), -1)
    cv2.circle(img, (size // 2 + 35, size // 2 - 5), 10, (235, 210, 160), -1)
    for offset in [-36, -18, 0, 18, 36]:
        cv2.line(img, (28, size // 2 + offset), (size - 24, size // 2 - offset // 2), (55, 28, 24), 2)
    cv2.circle(img, (size // 2 - 22, size // 2 + 16), 4, (120, 18, 12), -1)
    cv2.circle(img, (size // 2 - 8, size // 2 + 22), 3, (130, 20, 16), -1)
    return img


def test_dr_xai_preprocessor_shape(tmp_path):
    rgb = synthetic_fundus()
    path = tmp_path / "fundus.png"
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    out = DRXAIPreprocessor(image_size=128)(path)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8


def test_lesion_maps_and_embedding_are_balanced_128d():
    maps = lesion_evidence_maps(synthetic_fundus())
    assert set(maps) == set(LESION_CHANNELS)
    emb = lesion_map_embedding(maps, dim=128)
    assert emb.shape == (128,)
    assert np.isfinite(emb).all()
    assert np.isclose(np.linalg.norm(emb), 1.0)


def test_neovascularization_channel_is_conservative():
    maps = lesion_evidence_maps(synthetic_fundus())
    nv_area = float((maps["neovascularization"] >= 0.35).mean())
    hemorrhage_area = float((maps["hemorrhage"] >= 0.35).mean())
    assert nv_area < 0.10
    assert nv_area < hemorrhage_area
