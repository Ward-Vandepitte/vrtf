"""Data models for VRTF quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BBOX_SCALE = 1000.0  # MinerU fixed normalization divisor


@dataclass(frozen=True)
class TextLayout:
    """Layout information extracted from an original crop.

    Previously named _TextLayout; renamed for public API.
    """

    line_height: int        # spacing between consecutive line centres
    band_height: int        # median height of individual text bands
    line_tops: list[int]
    x_left: int
    text_y0: int
    text_y1: int
    x_right: int = 0  # right extent of text; 0 = unknown
    bands: tuple[tuple[int, int], ...] = ()  # text band intervals (start, end)
    has_non_text_bands: bool = False  # True when oversized bands were removed


# Backward-compat alias
_TextLayout = TextLayout


@dataclass(frozen=True)
class TypographyProfile:
    """Per-book consistent-typography calibration (honest-mode rendering).

    Produced once per book by ``vrtf.calibrate`` from a global
    (not per-block) measurement of the corpus. Drives source-free rendering:
    body text and each heading level get a single fixed font size, so the
    renderer no longer resizes per block to absorb bbox-detection error.

    **Sizes are stored in normalized per-mille units** (0-1000, same space as
    MinerU bboxes), NOT pixels -- a corpus can mix scan resolutions (e.g. book
    scans at 960x1346 with ground-truth crops at 1810x2562),
    so a pixel size would be wrong at a different resolution. Convert to pixels
    per page with the evaluator's ``scale_x = img_w/BBOX_SCALE`` (widths) and
    ``scale_y = img_h/BBOX_SCALE`` (heights) via the helpers below.

    ``body_leading`` is a ratio (line-pitch / band-height) and is therefore
    already resolution-independent.

    ``heading_levels`` maps a normalized band-height threshold to a normalized
    font height, sorted by threshold **descending**: a heading block is assigned
    the first entry whose (normalized) bbox/band height meets the threshold.
    """

    body_h_norm: float          # body single-line height, per-mille of page height
    body_leading: float         # line-pitch / band-height (resolution-independent)
    pitch_ratio: float = 36.0 / 64.0  # monospace advance / font height (template ratio)
    heading_levels: tuple[tuple[float, float], ...] = ()
    degenerate: bool = False    # True -> calibration fell back to a default

    def body_px(self, scale_y: float) -> int:
        """Body font size in pixels at this page's vertical scale."""
        return max(1, round(self.body_h_norm * scale_y))

    def pitch_px(self, scale_y: float) -> float:
        """Monospace advance in pixels (derived from body height; glyphs are
        square-pixel, so the advance/height ratio holds in render pixels)."""
        return max(1.0, self.body_px(scale_y) * self.pitch_ratio)

    def heading_px(self, band_h_norm: float, scale_y: float) -> int:
        """Heading font size (px) for a block of given normalized band height.

        Falls back to ``body_px`` when no heading levels were calibrated. Level
        lookup uses the block's OCR bbox/band height (metadata), not a fresh
        per-block source-ink measurement.
        """
        for threshold, size in self.heading_levels:
            if band_h_norm >= threshold:
                return max(1, round(size * scale_y))
        if self.heading_levels:
            return max(1, round(self.heading_levels[-1][1] * scale_y))
        return self.body_px(scale_y)


@dataclass(frozen=True)
class BlockScore:
    """Quality score for a single OCR block."""

    ink_overlap: float
    ssim: float
    text_density_ratio: float
    bbox_area_px: int
    evaluated: bool
    block_type: str
    mixed_content_masked: bool = False
    is_formula_heavy: bool = False
    ink_recall: float = -1.0         # recall component of ink_overlap F1
    ink_precision: float = -1.0      # precision component of ink_overlap F1
    winning_renderer: str = ""       # "text", "pdflatex", "composite", ""
    score_text_f1: float = -1.0      # F1 from three-way comparison
    score_pdflatex_f1: float = -1.0  # F1 from three-way comparison
    score_composite_f1: float = -1.0  # F1 from three-way comparison


@dataclass
class PageScore:
    """Aggregated quality score for a single page."""

    page_idx: int
    block_scores: list[BlockScore] = field(default_factory=list)
    empty_block_count: int = 0
    skipped_block_count: int = 0
    note: str = ""
    pdflatex_win_count: int = 0    # equation blocks where pdflatex scored higher
    pdflatex_lose_count: int = 0   # equation blocks where text scored higher
    pdflatex_fail_count: int = 0   # equation blocks where pdflatex failed to render
    full_page_f1: float = -1.0         # full-page composite F1 (layout fidelity)
    full_page_recall: float = -1.0     # full-page recall
    full_page_precision: float = -1.0  # full-page precision
    # --- Consistent-typography (honest-mode) diagnostics ---
    clipped_block_count: int = 0   # blocks whose rendered ink overflowed the bbox
    mixed_block_count: int = 0     # text blocks containing non-text (e.g. equation) bands
    masked_block_count: int = 0    # blocks where the source band-mask was applied
    cleanup_fired_count: int = 0   # equations rescued by the formula-cleanup fallback
    figure_render_fail_count: int = 0  # figure blocks w/ successful extraction but failed render
    # Transient analysis payload (ct_capture_confusion): (orig_bin, rend_canvas,
    # eff_tau) of the full-page composite. Never serialized; consumers must clear
    # it after use (full-page arrays — retaining 100s of pages is multi-GB).
    confusion: tuple | None = None

    @property
    def evaluated_scores(self) -> list[BlockScore]:
        return [bs for bs in self.block_scores if bs.evaluated]

    @property
    def text_quality(self) -> float:
        """Bbox-area-weighted average ink_overlap across ALL blocks.

        Skipped blocks (images, headings, discarded) contribute 0 score
        weighted by their bbox area -- they represent unreproduced content.
        """
        with_area = [bs for bs in self.block_scores if bs.bbox_area_px > 0]
        if not with_area:
            return 0.0
        total_area = sum(bs.bbox_area_px for bs in with_area)
        if total_area == 0:
            return 0.0
        return sum(bs.ink_overlap * bs.bbox_area_px for bs in with_area) / total_area

    @property
    def ssim_average(self) -> float:
        """Mean SSIM across evaluated blocks."""
        scores = self.evaluated_scores
        if not scores:
            return 0.0
        return sum(bs.ssim for bs in scores) / len(scores)

    def score_percent(self, w_overlap: float = 1.0, w_ssim: float = 0.0) -> float:
        """Combined score as percentage.

        Default weights use overlap only -- SSIM on binary text images
        is anti-correlated with overlap (r=-0.42) and does not add signal.
        """
        return (w_overlap * self.text_quality + w_ssim * self.ssim_average) * 100

    def to_dict(self, w_overlap: float = 1.0, w_ssim: float = 0.0) -> dict[str, Any]:
        d: dict[str, Any] = {
            "page_idx": self.page_idx,
            "score_percent": round(self.score_percent(w_overlap, w_ssim), 2),
            "text_quality": round(self.text_quality, 4),
            "ssim_average": round(self.ssim_average, 4),
            "evaluated_blocks": len(self.evaluated_scores),
            "empty_blocks": self.empty_block_count,
            "skipped_blocks": self.skipped_block_count,
            "total_blocks": len(self.block_scores),
            "note": self.note,
            "full_page_f1": round(self.full_page_f1, 4),
            "full_page_recall": round(self.full_page_recall, 4),
            "full_page_precision": round(self.full_page_precision, 4),
        }
        if self.pdflatex_win_count or self.pdflatex_lose_count or self.pdflatex_fail_count:
            d["pdflatex_win"] = self.pdflatex_win_count
            d["pdflatex_lose"] = self.pdflatex_lose_count
            d["pdflatex_fail"] = self.pdflatex_fail_count
        # CT diagnostics emitted only when nonzero -> old JSON stays byte-stable
        if self.clipped_block_count or self.mixed_block_count or self.masked_block_count:
            d["clipped_blocks"] = self.clipped_block_count
            d["mixed_blocks"] = self.mixed_block_count
            d["masked_blocks"] = self.masked_block_count
        if self.cleanup_fired_count or self.figure_render_fail_count:
            d["cleanup_fired"] = self.cleanup_fired_count
            d["figure_render_fails"] = self.figure_render_fail_count
        if any(bs.winning_renderer for bs in self.block_scores):
            d["detail_blocks"] = [
                {
                    "type": bs.block_type,
                    "ink_overlap": round(bs.ink_overlap, 4),
                    "ink_recall": round(bs.ink_recall, 4),
                    "ink_precision": round(bs.ink_precision, 4),
                    "density_ratio": round(bs.text_density_ratio, 2),
                    "masked": bs.mixed_content_masked,
                    "renderer": bs.winning_renderer,
                    "f1_text": round(bs.score_text_f1, 4),
                    "f1_pdf": round(bs.score_pdflatex_f1, 4),
                    "f1_comp": round(bs.score_composite_f1, 4),
                }
                for bs in self.block_scores
                if bs.evaluated
            ]
        return d


@dataclass
class BookEvaluation:
    """Evaluation results for an entire book/run."""

    page_scores: list[PageScore] = field(default_factory=list)

    @property
    def average_score(self) -> float:
        scored = [ps for ps in self.page_scores if ps.evaluated_scores]
        if not scored:
            return 0.0
        return sum(ps.score_percent() for ps in scored) / len(scored)

    @property
    def average_full_page_f1(self) -> float:
        scored = [ps for ps in self.page_scores if ps.full_page_f1 >= 0]
        if not scored:
            return -1.0
        return sum(ps.full_page_f1 for ps in scored) / len(scored)

    def worst_pages(self, n: int = 10) -> list[PageScore]:
        scored = [ps for ps in self.page_scores if ps.evaluated_scores]
        return sorted(scored, key=lambda ps: ps.score_percent())[:n]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "average_score": round(self.average_score, 2),
            "total_pages": len(self.page_scores),
            "pages_with_scores": len(
                [ps for ps in self.page_scores if ps.evaluated_scores]
            ),
            "worst_pages": [
                {"page_idx": ps.page_idx, "score": round(ps.score_percent(), 2)}
                for ps in self.worst_pages()
            ],
            "pages": [ps.to_dict() for ps in self.page_scores],
        }
        avg_fp = self.average_full_page_f1
        if avg_fp >= 0:
            d["average_full_page_f1"] = round(avg_fp, 4)
        return d
