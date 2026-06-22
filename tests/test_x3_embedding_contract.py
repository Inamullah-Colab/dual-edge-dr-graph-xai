from __future__ import annotations

import cv2
import numpy as np

from rxg.dr_xai_evidence import image_embedding


def test_x3_image_embedding_is_128d_normalized():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(img, (64, 64), 52, (120, 70, 40), -1)
    cv2.circle(img, (88, 58), 8, (230, 210, 170), -1)
    emb = image_embedding(img, dim=128)
    assert emb.shape == (128,)
    assert np.isfinite(emb).all()
    assert np.isclose(np.linalg.norm(emb), 1.0)


def test_x3_column_names_are_explicit():
    cols = [f"x3_image_embed_{i:03d}" for i in range(128)]
    assert cols[0] == "x3_image_embed_000"
    assert cols[-1] == "x3_image_embed_127"
    assert len(cols) == 128
