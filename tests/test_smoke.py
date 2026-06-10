"""Smoke tests for the vrtf package."""

import numpy as np
import pytest


def test_import_public_api():
    """All public symbols are importable."""
    from vrtf import (
        BlockScore,
        BookEvaluation,
        PageScore,
        QualityEvaluationConfig,
        QualityEvaluationService,
        TextLayout,
    )
    assert QualityEvaluationService is not None
    assert QualityEvaluationConfig is not None
    assert BlockScore is not None
    assert PageScore is not None
    assert BookEvaluation is not None
    assert TextLayout is not None


def test_backward_compat_alias():
    """_TextLayout alias exists for backward compatibility."""
    from vrtf import TextLayout, _TextLayout

    assert _TextLayout is TextLayout


def test_config_defaults():
    """Config has sensible defaults."""
    from vrtf import QualityEvaluationConfig

    cfg = QualityEvaluationConfig()
    assert cfg.dt_threshold == 20.0
    assert cfg.use_dt_overlap is True
    assert cfg.font_size_multiplier == 1.10


def test_metrics_binarize():
    """Metrics.binarize produces a binary image."""
    from vrtf.metric import Metrics
    from vrtf import QualityEvaluationConfig

    m = Metrics(QualityEvaluationConfig())
    img = np.full((50, 100), 200, dtype=np.uint8)
    img[10:20, 10:80] = 30  # dark band
    result = m.binarize(img)
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_metrics_f1_perfect():
    """Identical images produce F1 ~ 1.0."""
    from vrtf.metric import Metrics
    from vrtf import QualityEvaluationConfig

    m = Metrics(QualityEvaluationConfig())
    img = np.full((50, 100), 255, dtype=np.uint8)
    img[10:20, 10:80] = 0
    f1, recall, precision = m.compute_f1_overlap(img, img.copy())
    assert f1 > 0.99


def test_page_score_empty():
    """Empty page scores zero."""
    from vrtf import PageScore

    ps = PageScore(page_idx=0)
    assert ps.score_percent() == 0.0
    assert ps.text_quality == 0.0


def test_block_score_fields():
    """BlockScore has all expected fields."""
    from vrtf import BlockScore

    bs = BlockScore(
        ink_overlap=0.95,
        ssim=0.5,
        text_density_ratio=0.3,
        bbox_area_px=10000,
        evaluated=True,
        block_type="text",
        winning_renderer="text",
    )
    assert bs.ink_overlap == 0.95
    assert bs.evaluated is True


def test_xcorr_shift():
    """Cross-correlation shift finds zero shift for identical images."""
    from vrtf.metric import xcorr_shift

    img = np.zeros((50, 100), dtype=np.uint8)
    img[20:30, 40:60] = 255
    dy, dx = xcorr_shift(img, img.copy(), max_shift=10)
    assert abs(dy) <= 1
    assert abs(dx) <= 1
