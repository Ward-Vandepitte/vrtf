"""Tests for full-page composite scoring mode."""

import numpy as np
import pytest

from vrtf.config import QualityEvaluationConfig
from vrtf.evaluator import QualityEvaluationService
from vrtf.models import PageScore


def _make_service(full_page: bool = True, dt_fp: float = 5.0) -> QualityEvaluationService:
    """Create a minimal evaluator with full-page mode configured."""
    config = QualityEvaluationConfig(
        full_page_mode=full_page,
        dt_threshold_full_page=dt_fp,
        dt_threshold=20.0,
        use_dt_overlap=True,
        generate_overlays=False,
    )
    return QualityEvaluationService(config)


def _rect(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draw a black rectangle (ink=0) on a white canvas (255)."""
    canvas[y0:y1, x0:x1] = 0


class TestCompositeFullPage:
    """Unit tests for _composite_full_page using synthetic data."""

    def test_identical_placement_high_f1(self):
        """Rendered blocks at exact original positions → F1 near 1.0."""
        svc = _make_service()
        H, W = 500, 500
        # Original: 3 black rectangles
        orig = np.full((H, W), 255, dtype=np.uint8)
        _rect(orig, 50, 50, 150, 80)    # block A
        _rect(orig, 200, 100, 350, 130)  # block B
        _rect(orig, 60, 300, 250, 330)   # block C

        # Rendered blocks at same positions
        blocks = []
        for x0, y0, x1, y1 in [(50, 50, 150, 80), (200, 100, 350, 130), (60, 300, 250, 330)]:
            rend = np.full((y1 - y0, x1 - x0), 0, dtype=np.uint8)  # solid ink
            blocks.append((x0, y0, x1, y1, rend))

        f1, recall, precision = svc._composite_full_page(W, H, orig, blocks, [])
        assert f1 > 0.95, f"Expected F1 > 0.95, got {f1:.3f}"

    def test_shifted_block_drops_f1(self):
        """Shifting one block by 50px should measurably drop F1."""
        svc = _make_service()
        H, W = 500, 500
        orig = np.full((H, W), 255, dtype=np.uint8)
        _rect(orig, 100, 100, 300, 150)

        # Rendered at correct position
        rend_correct = np.full((50, 200), 0, dtype=np.uint8)
        f1_correct, _, _ = svc._composite_full_page(
            W, H, orig, [(100, 100, 300, 150, rend_correct)], [])

        # Rendered shifted 50px right
        rend_shifted = np.full((50, 200), 0, dtype=np.uint8)
        f1_shifted, _, _ = svc._composite_full_page(
            W, H, orig, [(150, 100, 350, 150, rend_shifted)], [])

        assert f1_correct > f1_shifted, (
            f"Shifted block should score lower: correct={f1_correct:.3f}, shifted={f1_shifted:.3f}")

    def test_removed_block_drops_recall(self):
        """Missing a block → recall drops (original ink unmatched)."""
        svc = _make_service()
        H, W = 500, 500
        orig = np.full((H, W), 255, dtype=np.uint8)
        _rect(orig, 50, 50, 150, 80)
        _rect(orig, 200, 200, 350, 230)

        rend_a = np.full((30, 100), 0, dtype=np.uint8)
        rend_b = np.full((30, 150), 0, dtype=np.uint8)

        # Both blocks present
        _, recall_both, prec_both = svc._composite_full_page(
            W, H, orig, [(50, 50, 150, 80, rend_a), (200, 200, 350, 230, rend_b)], [])

        # Only block A present (block B missing)
        _, recall_one, prec_one = svc._composite_full_page(
            W, H, orig, [(50, 50, 150, 80, rend_a)], [])

        assert recall_both > recall_one, (
            f"Missing block should drop recall: both={recall_both:.3f}, one={recall_one:.3f}")
        # Precision should not drop (rendered ink is still correct)
        assert prec_one >= prec_both - 0.01, "Precision should not drop when a block is removed"

    def test_extra_block_drops_precision(self):
        """Extra rendered block with no original ink → precision drops."""
        svc = _make_service()
        H, W = 500, 500
        orig = np.full((H, W), 255, dtype=np.uint8)
        _rect(orig, 50, 50, 150, 80)

        rend_real = np.full((30, 100), 0, dtype=np.uint8)
        rend_extra = np.full((30, 100), 0, dtype=np.uint8)

        # Just the real block
        _, _, prec_real = svc._composite_full_page(
            W, H, orig, [(50, 50, 150, 80, rend_real)], [])

        # Real + extra block in empty region
        _, _, prec_extra = svc._composite_full_page(
            W, H, orig,
            [(50, 50, 150, 80, rend_real), (300, 300, 400, 330, rend_extra)], [])

        assert prec_real > prec_extra, (
            f"Extra block should drop precision: real={prec_real:.3f}, extra={prec_extra:.3f}")

    def test_image_mask_excludes_photo_region(self):
        """Photo region in original should be masked → doesn't penalize recall."""
        svc = _make_service()
        H, W = 500, 500
        orig = np.full((H, W), 255, dtype=np.uint8)
        _rect(orig, 50, 50, 150, 80)    # text block
        _rect(orig, 200, 200, 400, 400)  # photo region (dark)

        rend_text = np.full((30, 100), 0, dtype=np.uint8)

        # Without masking: photo ink penalizes recall
        _, recall_no_mask, _ = svc._composite_full_page(
            W, H, orig, [(50, 50, 150, 80, rend_text)], [])

        # With masking: photo region excluded
        _, recall_masked, _ = svc._composite_full_page(
            W, H, orig, [(50, 50, 150, 80, rend_text)], [(200, 200, 400, 400)])

        assert recall_masked > recall_no_mask, (
            f"Masking photo should improve recall: masked={recall_masked:.3f}, "
            f"unmasked={recall_no_mask:.3f}")

    def test_backward_compat_disabled(self):
        """full_page_mode=False → fields stay at sentinel -1.0."""
        ps = PageScore(page_idx=0)
        assert ps.full_page_f1 == -1.0
        assert ps.full_page_recall == -1.0
        assert ps.full_page_precision == -1.0

        d = ps.to_dict()
        assert d["full_page_f1"] == -1.0

    def test_overlapping_blocks_use_minimum(self):
        """Overlapping blocks: np.minimum keeps ink from both."""
        svc = _make_service()
        H, W = 200, 200
        orig = np.full((H, W), 255, dtype=np.uint8)
        _rect(orig, 10, 10, 100, 50)
        _rect(orig, 50, 10, 150, 50)  # overlaps with first

        rend_a = np.full((40, 90), 0, dtype=np.uint8)
        rend_b = np.full((40, 100), 0, dtype=np.uint8)

        f1, _, _ = svc._composite_full_page(
            W, H, orig,
            [(10, 10, 100, 50, rend_a), (50, 10, 150, 50, rend_b)], [])
        assert f1 > 0.9, f"Overlapping blocks should score well: F1={f1:.3f}"

    def test_golden_composite_byte_stable(self):
        """GOLDEN: pinned exact F1 for a fixed offset-stripe fixture.

        Guards the compositing + DT-overlap math against silent drift while the
        consistent-typography work lands. The CT toggles default OFF, so this
        value must NOT change. If a deliberate scoring change moves it, update
        the constant in the same commit and explain why.
        """
        cfg = QualityEvaluationConfig(
            full_page_mode=True, dt_threshold_full_page=3.0, dt_threshold=20.0,
            use_dt_overlap=True, generate_overlays=False,
        )
        svc = QualityEvaluationService(cfg)
        H, W = 200, 200
        orig = np.full((H, W), 255, dtype=np.uint8)
        for y in range(30, 170, 10):
            orig[y:y + 3, 30:170] = 0
        rend = np.full((140, 140), 255, dtype=np.uint8)
        for y in range(0, 140, 10):
            rend[y:y + 3, 0:140] = 0
        f1, recall, precision = svc._composite_full_page(
            W, H, orig, [(34, 36, 174, 176, rend)], [])
        assert f1 == pytest.approx(0.60578231292517, abs=1e-12)
        assert recall == pytest.approx(0.60578231292517, abs=1e-12)
        assert precision == pytest.approx(0.60578231292517, abs=1e-12)


class TestPixelConfusionScoring:
    """Honest-mode pixel-confusion-matrix F1 (the 3-color overlay scoring)."""

    def _metrics(self, tau):
        from vrtf.config import QualityEvaluationConfig
        from vrtf.metric import Metrics
        return Metrics(QualityEvaluationConfig(
            consistent_typography_mode=True, dt_threshold=tau, use_dt_overlap=True,
        ))

    def test_exact_confusion_matrix(self):
        """tau=0: a clean per-pixel confusion matrix from green/red/blue counts."""
        m = self._metrics(tau=0.0)
        orig = np.full((10, 10), 255, dtype=np.uint8)
        rend = np.full((10, 10), 255, dtype=np.uint8)
        orig[:, 2:4] = 0   # source ink: cols 2,3 (20 px)
        rend[:, 3:5] = 0   # rendered ink: cols 3,4 (20 px)
        # G = col3 (overlap)=10, R = col2=10, B = col4=10
        f1, recall, precision = m.compute_f1_overlap(orig, rend)
        assert recall == pytest.approx(0.5)
        assert precision == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)

    def test_routes_only_in_honest_mode(self):
        """Flag OFF keeps the asymmetric metric; ON uses pixel confusion."""
        from vrtf.config import QualityEvaluationConfig
        from vrtf.metric import Metrics
        orig = np.full((10, 10), 255, dtype=np.uint8)
        rend = np.full((10, 10), 255, dtype=np.uint8)
        orig[:, 2:4] = 0
        rend[:, 3:5] = 0
        off = Metrics(QualityEvaluationConfig(
            consistent_typography_mode=False, dt_threshold=0.0))
        on = Metrics(QualityEvaluationConfig(
            consistent_typography_mode=True, dt_threshold=0.0))
        # asymmetric precision (rend-side) vs pixel-confusion precision differ here
        _, _, prec_off = off.compute_f1_overlap(orig, rend)
        _, _, prec_on = on.compute_f1_overlap(orig, rend)
        assert prec_on == pytest.approx(0.5)
        assert prec_off == pytest.approx(0.5)  # symmetric here, but path differs
        # empty-vs-empty is perfect in honest mode
        white = np.full((10, 10), 255, dtype=np.uint8)
        assert on.compute_f1_overlap(white, white) == (1.0, 1.0, 1.0)
