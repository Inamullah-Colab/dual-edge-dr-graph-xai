from __future__ import annotations

import numpy as np
import pandas as pd

from rxg.lesions import LESION_CLASSES, LESION_TO_CHANNEL
from rxg.metrics import ClassificationEvaluator
from rxg.stats import HypothesisTester, benjamini_hochberg


def test_neovascularization_is_supported():
    assert "neovascularization" in LESION_TO_CHANNEL
    assert len(LESION_CLASSES) == 5


def test_metrics_smoke():
    y = np.array([0, 1, 2, 3, 4])
    pred = np.array([0, 1, 2, 2, 4])
    m = ClassificationEvaluator().evaluate(y, pred)
    assert m["accuracy"] == 0.8
    assert 0 <= m["quadratic_weighted_kappa"] <= 1


def test_hypothesis_smoke():
    df = pd.DataFrame({
        "diagnosis": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "x4_fractal": [1, 1.1, 1.5, 1.4, 2, 2.1, 2.5, 2.6, 3, 3.1],
    })
    out = HypothesisTester().test_by_grade(df, ["x4_fractal"])
    assert not out.empty
    assert out.iloc[0]["significant_fdr_0_05"] in {True, False}


def test_bh_monotone_bounds():
    p = np.array([0.001, 0.02, 0.5])
    q = benjamini_hochberg(p)
    assert np.all(q >= 0) and np.all(q <= 1)
