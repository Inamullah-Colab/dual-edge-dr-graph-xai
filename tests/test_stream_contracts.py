from __future__ import annotations

from scripts.run_full_fusion_graph import select_x3_columns


def test_x3_schema_requires_exact_128_image_embedding_columns():
    cols = [f"x3_image_embed_{i:03d}" for i in range(128)]
    assert select_x3_columns(cols) == cols


def test_x3_schema_rejects_noncanonical_embedding_names():
    cols = [f"embed_{i:03d}" for i in range(128)]
    try:
        select_x3_columns(cols)
    except ValueError as exc:
        assert "x3_image_embed_000" in str(exc)
    else:
        raise AssertionError("noncanonical X3 columns should be rejected")
