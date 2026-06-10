"""Tests for confusion-mask exposure, avenue-run routing, and capture plumbing.

Covers the engine additions for the per-content-type decomposition and the
Deel II avenue runs: ``compute_pixel_confusion_masks``, ``ct_figure_recreation``
routing in honest mode, ``ct_capture_confusion`` stashing, and the
``cleanup_fired_count`` counter.
"""

import json

import numpy as np
import pytest
from PIL import Image

from vrtf.config import QualityEvaluationConfig
from vrtf.evaluator import QualityEvaluationService
from vrtf.metric import Metrics
from vrtf.models import PageScore
from vrtf.renderers.figure import FigureRenderer


# ---------------------------------------------------------------------------
# compute_pixel_confusion_masks
# ---------------------------------------------------------------------------

def _random_bin(rng, h=60, w=80, ink_frac=0.15):
    """Random binarized image: 0=ink, 255=background."""
    img = np.full((h, w), 255, dtype=np.uint8)
    img[rng.random((h, w)) < ink_frac] = 0
    return img


class TestConfusionMasks:
    @pytest.mark.parametrize("tau", [0.0, 1.0, 3.0, 10.0])
    def test_masks_reproduce_scalars_exactly(self, tau):
        """Scalars recomputed from the masks == compute_f1_pixel_confusion."""
        rng = np.random.default_rng(42)
        m = Metrics(QualityEvaluationConfig(
            consistent_typography_mode=True, dt_threshold=tau))
        for _ in range(5):
            orig = _random_bin(rng)
            rend = _random_bin(rng)
            f1, recall, precision = m.compute_f1_pixel_confusion(orig, rend)
            g, r, b = m.compute_pixel_confusion_masks(orig, rend, tau)
            G, R, B = int(g.sum()), int(r.sum()), int(b.sum())
            exp_recall = G / (G + R) if (G + R) else 1.0
            exp_precision = G / (G + B) if (G + B) else 1.0
            exp_f1 = (2 * G) / (2 * G + R + B) if (2 * G + R + B) else 0.0
            assert recall == exp_recall
            assert precision == exp_precision
            assert f1 == exp_f1

    def test_partition_properties(self):
        """G/R partition the source ink; B is rendered-only ink."""
        rng = np.random.default_rng(7)
        m = Metrics(QualityEvaluationConfig(consistent_typography_mode=True))
        orig = _random_bin(rng)
        rend = _random_bin(rng)
        g, r, b = m.compute_pixel_confusion_masks(orig, rend, tau=2.0)
        orig_ink = orig == 0
        rend_ink = rend == 0
        assert not (g & r).any()                      # disjoint
        assert ((g | r) == orig_ink).all()            # exhaustive over source ink
        assert (b <= (rend_ink & ~orig_ink)).all()    # FP only on rendered-only ink

    def test_empty_vs_empty(self):
        m = Metrics(QualityEvaluationConfig(consistent_typography_mode=True))
        white = np.full((10, 10), 255, dtype=np.uint8)
        g, r, b = m.compute_pixel_confusion_masks(white, white, tau=1.0)
        assert not g.any() and not r.any() and not b.any()
        assert m.compute_f1_pixel_confusion(white, white) == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# ct_figure_recreation routing (honest mode)
# ---------------------------------------------------------------------------

def _honest_service(**overrides) -> QualityEvaluationService:
    cfg = QualityEvaluationConfig(
        consistent_typography_mode=True,
        full_page_mode=True,
        generate_overlays=False,
        font_selection_enabled=False,
        **overrides,
    )
    return QualityEvaluationService(cfg)


def _page_image(tmp_path, h=500, w=500):
    path = tmp_path / "page.png"
    arr = np.full((h, w), 255, dtype=np.uint8)
    arr[60:140, 60:140] = 0  # some source ink inside the figure bbox
    Image.fromarray(arr).save(path)
    return path


def _line_graph_block():
    return {
        "type": "image",
        "figure_type": "line_graph",
        "bbox": [100, 100, 300, 300],  # 0-1000 normalized
        "graph_extraction": {"status": "success", "data": {"series": [{}]}},
    }


def _drawing_block():
    return {
        "type": "image",
        "figure_type": "technical_drawing",
        "bbox": [100, 100, 300, 300],
        "drawing_extraction": {"status": "success", "svg": "<svg/>"},
    }


def _photo_block():
    return {
        "type": "image",
        "figure_type": "photo",
        "bbox": [400, 400, 600, 600],
    }


def _stub_line_graph(calls):
    def stub(extraction, w_px, h_px, orig_crop=None):
        calls.append({"orig_crop": orig_crop})
        out = np.full((h_px, w_px), 255, dtype=np.uint8)
        out[10:30, 10:30] = 0
        return out
    return stub


class TestFigureRecreationRouting:
    def test_off_keeps_figures_missed(self, tmp_path):
        svc = _honest_service()  # ct_figure_recreation defaults to "off"
        ps = svc.evaluate_page(_page_image(tmp_path), [_line_graph_block()], 0)
        assert ps.full_page_f1 == -1.0          # nothing rendered
        assert all(not bs.evaluated for bs in ps.block_scores)

    def test_source_free_routes_without_crop(self, tmp_path, monkeypatch):
        svc = _honest_service(ct_figure_recreation="source_free")
        calls = []
        monkeypatch.setattr(svc._figure, "render_line_graph",
                            _stub_line_graph(calls))
        ps = svc.evaluate_page(
            _page_image(tmp_path),
            [_line_graph_block(), _photo_block()], 0)
        assert len(calls) == 1
        assert calls[0]["orig_crop"] is None     # source-free render
        assert ps.full_page_f1 >= 0.0            # figure entered the composite
        # photo stays missed (unevaluated, never masked)
        photo_scores = [bs for bs in ps.block_scores if not bs.evaluated]
        assert photo_scores, "photo must remain an unevaluated (missed) block"

    def test_peeking_routes_with_crop(self, tmp_path, monkeypatch):
        svc = _honest_service(ct_figure_recreation="peeking")
        calls = []
        monkeypatch.setattr(svc._figure, "render_line_graph",
                            _stub_line_graph(calls))
        ps = svc.evaluate_page(_page_image(tmp_path), [_line_graph_block()], 0)
        assert len(calls) == 1
        assert calls[0]["orig_crop"] is not None  # legacy crop-guided render
        assert ps.full_page_f1 >= 0.0

    def test_drawing_identical_in_both_modes(self, tmp_path, monkeypatch):
        """Technical drawings have no peeking mechanism: same render + no xcorr."""
        rendered = {}

        def stub_drawing(svg, w_px, h_px):
            out = np.full((h_px, w_px), 255, dtype=np.uint8)
            out[5:25, 5:25] = 0
            return out

        monkeypatch.setattr(FigureRenderer, "render_technical_drawing",
                            staticmethod(stub_drawing))
        for mode in ("source_free", "peeking"):
            svc = _honest_service(ct_figure_recreation=mode)
            ps = svc.evaluate_page(_page_image(tmp_path), [_drawing_block()], 0)
            rendered[mode] = ps.full_page_f1
        assert rendered["source_free"] == rendered["peeking"]

    def test_invalid_mode_raises(self, tmp_path):
        svc = _honest_service(ct_figure_recreation="source-free")  # typo
        with pytest.raises(ValueError, match="ct_figure_recreation"):
            svc.evaluate_page(_page_image(tmp_path), [_line_graph_block()], 0)

    def test_non_honest_mode_ignores_toggle(self, tmp_path, monkeypatch):
        """Outside honest mode the toggle is dormant; legacy dispatch applies."""
        cfg = QualityEvaluationConfig(
            consistent_typography_mode=False,
            full_page_mode=True,
            generate_overlays=False,
            font_selection_enabled=False,
            ct_figure_recreation="not-even-valid",
        )
        svc = QualityEvaluationService(cfg)
        calls = []
        monkeypatch.setattr(svc._figure, "render_line_graph",
                            _stub_line_graph(calls))
        ps = svc.evaluate_page(_page_image(tmp_path), [_line_graph_block()], 0)
        assert len(calls) == 1                    # legacy path still renders
        assert calls[0]["orig_crop"] is not None  # legacy is crop-guided

    def test_render_failure_counted(self, tmp_path, monkeypatch):
        svc = _honest_service(ct_figure_recreation="source_free")
        monkeypatch.setattr(svc._figure, "render_line_graph",
                            lambda *a, **k: None)
        ps = svc.evaluate_page(_page_image(tmp_path), [_line_graph_block()], 0)
        assert ps.figure_render_fail_count == 1


# ---------------------------------------------------------------------------
# ct_capture_confusion
# ---------------------------------------------------------------------------

class TestCaptureConfusion:
    def test_capture_sets_confusion_and_preserves_f1(self, tmp_path, monkeypatch):
        results = {}
        for capture in (False, True):
            svc = _honest_service(ct_figure_recreation="source_free",
                                  ct_capture_confusion=capture)
            monkeypatch.setattr(svc._figure, "render_line_graph",
                                _stub_line_graph([]))
            ps = svc.evaluate_page(_page_image(tmp_path), [_line_graph_block()], 0)
            results[capture] = ps
        assert results[False].confusion is None
        assert results[True].confusion is not None
        orig_bin, rend_canvas, eff_tau = results[True].confusion
        assert orig_bin.shape == rend_canvas.shape
        assert eff_tau > 0
        assert results[True].full_page_f1 == results[False].full_page_f1
        # the captured pair reproduces the reported score at eff_tau
        g, r, b = svc._metrics.compute_pixel_confusion_masks(
            orig_bin, rend_canvas, eff_tau)
        G, R, B = int(g.sum()), int(r.sum()), int(b.sum())
        f1 = (2 * G) / (2 * G + R + B) if (2 * G + R + B) else 0.0
        assert f1 == pytest.approx(results[True].full_page_f1, abs=1e-9)

    def test_no_stale_stash_on_unscored_page(self, tmp_path, monkeypatch):
        """A page whose composite never runs must not inherit the previous
        page's confusion arrays."""
        svc = _honest_service(ct_figure_recreation="source_free",
                              ct_capture_confusion=True)
        monkeypatch.setattr(svc._figure, "render_line_graph",
                            _stub_line_graph([]))
        ps1 = svc.evaluate_page(_page_image(tmp_path), [_line_graph_block()], 0)
        assert ps1.confusion is not None
        ps2 = svc.evaluate_page(_page_image(tmp_path), [], 1)  # no blocks
        assert ps2.full_page_f1 == -1.0
        assert ps2.confusion is None


# ---------------------------------------------------------------------------
# cleanup_fired_count
# ---------------------------------------------------------------------------

class TestCleanupFired:
    def _renderer_with_map(self, tmp_path, entries):
        path = tmp_path / "cleanup.json"
        path.write_text(json.dumps({"formulas": entries}))
        cfg = QualityEvaluationConfig(
            consistent_typography_mode=True,
            formula_cleanup_path=str(path),
            generate_overlays=False,
            font_selection_enabled=False,
        )
        return QualityEvaluationService(cfg)

    @staticmethod
    def _fake_pdflatex(fail_latex):
        def fake(latex, h_px, **kwargs):
            if latex == fail_latex:
                return None
            return np.full((20, 60), 0, dtype=np.uint8)
        return fake

    def test_fallback_fires_and_is_reported(self, tmp_path, monkeypatch):
        svc = self._renderer_with_map(
            tmp_path, {"k1": {"cleaned_latex": "x+y"}})
        monkeypatch.setattr(svc._equation, "render_formula_pdflatex",
                            self._fake_pdflatex(fail_latex="BAD"))
        block = {"type": "equation", "text": "BAD", "img_path": "k1"}
        result = svc._equation.render_equation_consistent(block, 200, 60, 20)
        assert result is not None
        _canvas, _clipped, cleanup_fired = result
        assert cleanup_fired is True
        # evaluator increments the page counter from the flag
        ps = PageScore(page_idx=0)
        text = np.full((60, 200), 255, dtype=np.uint8)
        svc._pick_best_equation_rendering(
            block=block, block_type="equation", text_rendered=text,
            w_px=200, h_px=60, orig_bin=text, band_mask=None, layout=None,
            page_score=ps, ct_body_px=20)
        assert ps.cleanup_fired_count == 1

    def test_raw_compile_does_not_fire(self, tmp_path, monkeypatch):
        svc = self._renderer_with_map(
            tmp_path, {"k1": {"cleaned_latex": "x+y"}})
        monkeypatch.setattr(svc._equation, "render_formula_pdflatex",
                            self._fake_pdflatex(fail_latex="<never>"))
        block = {"type": "equation", "text": "GOOD", "img_path": "k1"}
        _c, _cl, cleanup_fired = svc._equation.render_equation_consistent(
            block, 200, 60, 20)
        assert cleanup_fired is False

    def test_no_map_entry_falls_to_text(self, tmp_path, monkeypatch):
        svc = self._renderer_with_map(tmp_path, {})
        monkeypatch.setattr(svc._equation, "render_formula_pdflatex",
                            self._fake_pdflatex(fail_latex="BAD"))
        block = {"type": "equation", "text": "BAD", "img_path": "missing"}
        assert svc._equation.render_equation_consistent(block, 200, 60, 20) is None

    def test_missing_cleanup_file_does_not_latch(self, tmp_path):
        """A missing path must not permanently disable the fallback."""
        cfg = QualityEvaluationConfig(
            formula_cleanup_path=str(tmp_path / "absent.json"))
        svc = QualityEvaluationService(cfg)
        assert svc._equation.get_formula_cleanup() == {}
        # file appears later -> next call loads it
        (tmp_path / "absent.json").write_text(
            json.dumps({"formulas": {"k": {"cleaned_latex": "a"}}}))
        assert svc._equation.get_formula_cleanup() == {"k": "a"}
