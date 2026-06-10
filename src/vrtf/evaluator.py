"""Orchestration hub for Visual Roundtrip Fidelity evaluation.

Composes all renderer, metric, font, and overlay modules and exposes the
public API: ``evaluate_book``, ``evaluate_run``, ``evaluate_page``.

This module was extracted from ``services/quality_evaluation_service.py``
to decouple evaluation orchestration from the individual rendering and
scoring pipelines.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from vrtf.config import QualityEvaluationConfig
from vrtf.metric import Metrics, shift_image, xcorr_shift
from vrtf.models import BBOX_SCALE, BlockScore, BookEvaluation, PageScore, TextLayout
from vrtf.overlay import OverlayGenerator
from vrtf.renderers.equation import EquationRenderer
from vrtf.renderers.figure import FigureRenderer
from vrtf.renderers.table import TableRenderer
from vrtf.renderers.text import TextRenderer
from vrtf.utils.font import FontCache
from vrtf.utils.latex import simplify_latex, strip_latex

logger = logging.getLogger(__name__)

# Scoring constants extracted from evaluate_page; values tuned on the
# development corpus (sensitivity study).

# Text-block skip gate: reject blocks whose text length exceeds the pixel
# area divided by this factor. At 300 DPI a glyph is ~10x30 px = ~300 px^2,
# so 100 allows ~3x packing density before flagging the block as merged.
TEXT_AREA_PER_CHAR_PX2 = 100

# Floor on the chars-per-area gate so tiny blocks still accept short text.
TEXT_MIN_REASONABLE_CHARS = 500

# Diagram detector: confidence above this skips the block from OCR scoring.
DIAGRAM_SKIP_CONFIDENCE = 0.6

# Photo-block scoring weights (detection + placement + VLM description).
PHOTO_SCORE_DETECTION = 0.2
PHOTO_SCORE_PLACEMENT = 0.2
PHOTO_SCORE_DESCRIPTION = 0.6


class QualityEvaluationService:
    """Evaluates OCR quality by comparing rendered text to original scans.

    Composes ``FontCache``, ``Metrics``, four block-type renderers
    (text / equation / table / figure), and ``OverlayGenerator``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: QualityEvaluationConfig | None = None) -> None:
        self.config = config or QualityEvaluationConfig()
        self._fonts = FontCache(self.config)
        self._metrics = Metrics(self.config)
        self._text = TextRenderer(self.config, self._fonts, self._metrics)
        self._equation = EquationRenderer(self.config, self._fonts, self._metrics)
        self._table = TableRenderer(self.config, self._fonts)
        self._figure = FigureRenderer(self.config, self._metrics)
        self._overlay = OverlayGenerator(self.config, self._fonts)
        # Transient per-page stash for ct_capture_confusion (set by
        # _composite_full_page, moved onto PageScore by evaluate_page).
        self._last_confusion: tuple | None = None

    # ------------------------------------------------------------------
    # Static / class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unevaluated_score(
        block_type: str,
        bbox_area_px: int = 0,
        text_density_ratio: float = 0.0,
        is_formula_heavy: bool = False,
    ) -> BlockScore:
        """Create a BlockScore for a block that was not evaluated."""
        return BlockScore(
            ink_overlap=0.0, ssim=0.0,
            text_density_ratio=text_density_ratio,
            bbox_area_px=bbox_area_px, evaluated=False,
            block_type=block_type,
            is_formula_heavy=is_formula_heavy,
        )

    @staticmethod
    def _parse_batch_num(path: Path) -> int | None:
        """Extract batch number from a content_list filename."""
        match = re.search(r'batch_(\d+)', path.name)
        if not match:
            return None
        return int(match.group(1))

    @property
    def missing_templates(self) -> set[str]:
        """Characters encountered during evaluation that had no bitmap template."""
        return self._fonts._missing_templates.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_book(
        self,
        run_dirs: list[Path],
        source_images: list[Path],
        batch_size: int | None = None,
        book_output_dir: Path | None = None,
    ) -> BookEvaluation:
        """Evaluate a book by aggregating content_lists from one or more run dirs.

        Args:
            run_dirs: Run directories to walk. May be a single dir (legacy) or
                many (dots.ocr per-batch layout). Caller should pre-filter to a
                single engine via ImageProcessorService.discover_runs.
            source_images: Natsorted list of ALL source images for the book.
            batch_size: Pages per batch. If None, auto-detected from
                max(page_idx)+1 across loaded content_lists. Auto-detection
                only works when loaded batches are anchored at batch 1; raises
                ValueError otherwise. If passed and disagrees with observation,
                raises ValueError.
            book_output_dir: If provided, atomic-write the BookEvaluation JSON
                to book_output_dir/"quality"/"book_evaluation.json".

        Returns:
            BookEvaluation with per-page scores. Logs WARNING on:
                - unparseable batch numbers
                - batch numbers < 1
                - gaps in the batch sequence
                - pages_evaluated mismatched against source_images count

        Raises:
            FileNotFoundError: empty run_dirs or no content_lists found
            ValueError: batch_size cannot be inferred or contradicts observation
        """
        if not run_dirs:
            raise FileNotFoundError("evaluate_book called with no run_dirs")

        # Reset per-eval instance state (B2/B8 fix)
        self._fonts._missing_templates = set()

        content_lists = self._load_content_lists_multi(run_dirs)
        if not content_lists:
            raise FileNotFoundError(
                f"No content_lists found across run dirs: "
                f"{[d.name for d in run_dirs]}"
            )

        # Compute batch numbers (already deduped, sorted)
        batch_nums_with_none = {self._parse_batch_num(p) for p, _ in content_lists}
        batch_nums = sorted(n for n in batch_nums_with_none if n is not None)
        if not batch_nums:
            raise ValueError("No parseable batch numbers in loaded content_lists")

        # Auto-detect batch_size -- only safe when anchored at batch 1 (D10)
        if batch_size is None:
            if batch_nums[0] != 1:
                raise ValueError(
                    f"Cannot auto-detect batch_size: loaded batches start at "
                    f"{batch_nums[0]}, not 1. Pass batch_size explicitly."
                )
            detected = max(
                max((b.get("page_idx", 0) for b in blocks), default=-1)
                for _, blocks in content_lists
            ) + 1
            batch_size = max(detected, 1)
        elif batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        else:
            # Sanity-check explicit batch_size against observation
            observed_max = max(
                max((b.get("page_idx", 0) for b in blocks), default=-1)
                for _, blocks in content_lists
            ) + 1
            if observed_max > batch_size:
                raise ValueError(
                    f"batch_size={batch_size} < observed max page_idx+1={observed_max}. "
                    f"Pages would be silently misaligned."
                )

        # Validate batch sequence (B6)
        expected = list(range(batch_nums[0], batch_nums[-1] + 1))
        if batch_nums != expected:
            missing = set(expected) - set(batch_nums)
            logger.warning("Batch sequence has gaps: missing %s", sorted(missing))

        evaluation = BookEvaluation()
        # defensive dedup; _load_content_lists_multi already dedups by batch_num
        seen_pages: set[int] = set()
        skipped_count = 0

        for cl_path, blocks in content_lists:
            bn = self._parse_batch_num(cl_path)
            if bn is None or bn < 1:
                logger.warning(
                    "Skipping content_list with invalid batch num: %s", cl_path
                )
                skipped_count += 1
                continue

            start = (bn - 1) * batch_size
            batch_images = source_images[start: start + batch_size]

            pages: dict[int, list[dict]] = {}
            for block in blocks:
                pid = block.get("page_idx", 0)
                pages.setdefault(pid, []).append(block)

            for page_idx in sorted(pages.keys()):
                global_idx = page_idx + start
                if global_idx in seen_pages:
                    continue
                if page_idx >= len(batch_images):
                    logger.warning(
                        "page_idx %d exceeds batch images (%d) in %s, skipping",
                        page_idx, len(batch_images), cl_path.name,
                    )
                    skipped_count += 1
                    continue
                source_image = batch_images[page_idx]
                page_score = self.evaluate_page(
                    source_image, pages[page_idx], global_idx,
                )
                # Never retain confusion arrays across a whole book (multi-GB);
                # per-page consumers use evaluate_page directly.
                page_score.confusion = None
                evaluation.page_scores.append(page_score)
                seen_pages.add(global_idx)
                if len(seen_pages) % 10 == 0:
                    gc.collect()

        evaluation.page_scores.sort(key=lambda ps: ps.page_idx)

        # Coverage summary (B5)
        logger.info(
            "evaluate_book: pages_evaluated=%d skipped=%d source_images_total=%d",
            len(evaluation.page_scores), skipped_count, len(source_images),
        )

        # Missing templates summary
        if self._fonts._missing_templates:
            sorted_chars = sorted(self._fonts._missing_templates)
            logger.info(
                "Missing bitmap templates for %d unique characters: %s",
                len(sorted_chars),
                ", ".join(f"{ch!r} (U+{ord(ch):04X})" for ch in sorted_chars),
            )

        # Atomic write
        if book_output_dir is not None:
            out_path = book_output_dir / "quality" / "book_evaluation.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(out_path, evaluation.to_dict())

        return evaluation

    def evaluate_run(
        self,
        run_dir: Path,
        source_images: list[Path],
        batch_size: int = 50,
    ) -> BookEvaluation:
        """Evaluate all content_list.json files in a single run directory.

        Thin compatibility wrapper around evaluate_book for legacy callers
        (process_pdf.py, evaluate_quality.py run). Output goes to
        <run_dir>/quality/evaluation.json (preserved per-run convention).
        """
        # Note: evaluate_book resets _missing_templates internally
        evaluation = self.evaluate_book(
            run_dirs=[run_dir],
            source_images=source_images,
            batch_size=batch_size,
            book_output_dir=None,  # different output convention than book mode
        )
        out_path = run_dir / "quality" / "evaluation.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(out_path, evaluation.to_dict())
        return evaluation

    def evaluate_page(
        self,
        source_image_path: Path,
        blocks: list[dict],
        page_idx: int,
    ) -> PageScore:
        """Evaluate a single page against its OCR blocks.

        Args:
            source_image_path: Path to the source image (original or upscaled).
            blocks: List of block dicts from content_list.json for this page.
            page_idx: Global page index for reporting.

        Returns:
            PageScore with per-block results.
        """
        page_score = PageScore(page_idx=page_idx)
        # Reset the capture stash here, not only in _composite_full_page: on a
        # page with no renderable blocks the composite never runs, and a stale
        # stash from the previous page must not be attached to this one.
        self._last_confusion = None

        with Image.open(source_image_path) as img:
            img_w, img_h = img.size
            img_array = np.array(img)

        scale_x = img_w / BBOX_SCALE
        scale_y = img_h / BBOX_SCALE

        # Full-page composite: collect rendered blocks and image mask regions
        fp_rendered: list[tuple[int, int, int, int, np.ndarray]] = []
        fp_image_masks: list[tuple[int, int, int, int]] = []

        for block in blocks:
            block_type = block.get("type", "unknown")

            if block_type not in ("text", "equation", "table", "heading"):
                self._score_non_text_block(
                    block, img_array, scale_x, scale_y, img_w, img_h,
                    page_score, fp_rendered, fp_image_masks,
                )
                continue

            if block_type == "table":
                self._score_table_block(
                    block, img_array, scale_x, scale_y, img_w, img_h,
                    page_score, fp_rendered, page_idx,
                )
                continue

            # text / equation / heading blocks
            self._score_text_block(
                block, img_array, scale_x, scale_y, img_w, img_h,
                page_score, fp_rendered, page_idx,
            )

        if not page_score.evaluated_scores:
            page_score.note = "No evaluatable text blocks on this page"

        # Full-page composite scoring
        if self.config.full_page_mode and fp_rendered:
            try:
                fp_f1, fp_recall, fp_precision = self._composite_full_page(
                    img_w, img_h, img_array, fp_rendered, fp_image_masks)
                page_score.full_page_f1 = fp_f1
                page_score.full_page_recall = fp_recall
                page_score.full_page_precision = fp_precision
                page_score.confusion = self._last_confusion
            except Exception:
                logger.warning("Full-page composite failed (page %d)", page_idx)

        return page_score

    def _score_non_text_block(
        self,
        block: dict,
        img_array: np.ndarray,
        scale_x: float,
        scale_y: float,
        img_w: int,
        img_h: int,
        page_score: PageScore,
        fp_rendered: list[tuple[int, int, int, int, np.ndarray]],
        fp_image_masks: list[tuple[int, int, int, int]],
    ) -> None:
        """Handle blocks that are not text/equation/table/heading.

        Image blocks are scored by figure_type (photo / line_graph /
        technical_drawing); other types (discarded, unknown) get an
        unevaluated score with zero area.
        """
        page_score.skipped_block_count += 1
        block_type = block.get("type", "unknown")
        bbox = block.get("bbox")
        if not (block_type == "image" and bbox and len(bbox) == 4):
            page_score.block_scores.append(
                self._unevaluated_score(block_type, bbox_area_px=0))
            return

        sx0 = max(0, min(int(bbox[0] * scale_x), img_w - 1))
        sy0 = max(0, min(int(bbox[1] * scale_y), img_h - 1))
        sx1 = max(sx0 + 1, min(int(bbox[2] * scale_x), img_w))
        sy1 = max(sy0 + 1, min(int(bbox[3] * scale_y), img_h))
        skip_area = (sx1 - sx0) * (sy1 - sy0)

        # Honest mode: never recreate or mask figures/photos. Whether a book's
        # figures were classified (figure_type) and recreated is a pipeline
        # accident that would make cross-book scores incomparable, so figure
        # ink simply counts as missed (red), uniformly across the corpus.
        # Exception: ct_figure_recreation routes line_graph/technical_drawing
        # through recreation for single-book avenue runs ("source_free" renders
        # without the source crop; "peeking" is the legacy crop-guided render,
        # measured only as an evaluation-inflation contrast). Photos and
        # unclassified images stay missed in every mode and are never masked.
        if self.config.consistent_typography_mode:
            mode = self.config.ct_figure_recreation
            if mode not in ("off", "source_free", "peeking"):
                raise ValueError(
                    f"ct_figure_recreation must be 'off', 'source_free' or "
                    f"'peeking', got {mode!r}")
            figure_type = block.get("figure_type")
            if mode != "off" and figure_type == "line_graph":
                self._score_line_graph_block(
                    block, img_array, sx0, sy0, sx1, sy1,
                    skip_area, page_score, fp_rendered,
                    source_free=(mode == "source_free"),
                )
                return
            if mode != "off" and figure_type == "technical_drawing":
                self._score_technical_drawing_block(
                    block, img_array, sx0, sy0, sx1, sy1,
                    skip_area, page_score, fp_rendered,
                )
                return
            page_score.block_scores.append(
                self._unevaluated_score("image", bbox_area_px=skip_area))
            return

        figure_type = block.get("figure_type")
        if figure_type == "photo":
            self._score_photo_block(
                block, img_array, sx0, sy0, sx1, sy1,
                skip_area, page_score, fp_image_masks,
            )
        elif figure_type == "line_graph":
            self._score_line_graph_block(
                block, img_array, sx0, sy0, sx1, sy1,
                skip_area, page_score, fp_rendered,
            )
        elif figure_type == "technical_drawing":
            self._score_technical_drawing_block(
                block, img_array, sx0, sy0, sx1, sy1,
                skip_area, page_score, fp_rendered,
            )
        else:
            # flowchart, table_image, unknown, None
            page_score.block_scores.append(
                self._unevaluated_score("image", bbox_area_px=skip_area))

    def _score_table_block(
        self,
        block: dict,
        img_array: np.ndarray,
        scale_x: float,
        scale_y: float,
        img_w: int,
        img_h: int,
        page_score: PageScore,
        fp_rendered: list[tuple[int, int, int, int, np.ndarray]],
        page_idx: int,
    ) -> None:
        """Score a table block with dedicated grid-aware renderer."""
        text = block.get("text", "")
        bbox = block.get("bbox")
        block_type = "table"

        if not self.config.evaluate_tables:
            page_score.skipped_block_count += 1
            page_score.block_scores.append(self._unevaluated_score(block_type))
            return

        table_html = block.get("table_body", "")
        has_pipe_text = text and '|' in text
        if bbox is None or (not table_html and not has_pipe_text):
            page_score.skipped_block_count += 1
            page_score.block_scores.append(self._unevaluated_score(block_type))
            return

        px_x0 = max(0, min(int(bbox[0] * scale_x), img_w - 1))
        px_y0 = max(0, min(int(bbox[1] * scale_y), img_h - 1))
        px_x1 = max(px_x0 + 1, min(int(bbox[2] * scale_x), img_w))
        px_y1 = max(px_y0 + 1, min(int(bbox[3] * scale_y), img_h))
        w_px = px_x1 - px_x0
        h_px = px_y1 - px_y0
        area = w_px * h_px
        if area < self.config.min_block_area_px:
            page_score.skipped_block_count += 1
            page_score.block_scores.append(
                self._unevaluated_score(block_type, bbox_area_px=area))
            return

        try:
            grid = None
            if table_html:
                grid = self._table.parse_html_table(table_html)
            if grid is None and has_pipe_text:
                grid = self._table.parse_markdown_table(text)
            if not grid:
                raise ValueError("no parseable grid")

            orig_crop = img_array[px_y0:px_y1, px_x0:px_x1]
            orig_bin = self._metrics.binarize(orig_crop, normalize_bg=True)

            profile = self.config.typography
            if self.config.ct_skip_table_grid and profile is not None:
                # Honest: consistent-typography grid, no source grid detection,
                # no stretch-to-bbox.
                font_px = profile.body_px(scale_y)
                rendered = self._table.render_grid_consistent(
                    grid, w_px, h_px,
                    font_px=font_px, leading=profile.body_leading)
            else:
                # Image-guided layout detection (source grid -> leaky)
                det_col_xs = det_row_ys = cell_types = None
                if self.config.use_image_guided_tables:
                    n_rows = len(grid)
                    n_cols = max(len(r) for r in grid)
                    det_row_ys, det_col_xs = self._table.detect_table_grid_lines(
                        orig_bin, n_rows, n_cols)
                    if det_row_ys is not None and det_col_xs is not None:
                        cell_types = self._table.classify_table_cells(
                            orig_bin, det_row_ys, det_col_xs)

                rendered = self._table.render_grid_in_bbox(
                    grid, w_px, h_px,
                    col_xs=det_col_xs, row_ys=det_row_ys,
                    cell_types=cell_types)
        except Exception:
            logger.warning("Table render failed (page %d, %dx%d px)",
                           page_idx, w_px, h_px)
            rendered = None

        if rendered is None:
            page_score.skipped_block_count += 1
            page_score.block_scores.append(
                self._unevaluated_score(block_type, bbox_area_px=area))
            return

        rend_bin = self._metrics.binarize(rendered, normalize_bg=False)

        # No xcorr -- grid borders provide spatial anchoring
        ink_overlap, ink_recall, ink_precision = self._metrics.compute_f1_overlap(
            orig_bin, rend_bin)
        ssim_val = Metrics.compute_ssim(orig_bin, rend_bin)
        orig_ink = np.count_nonzero(orig_bin == 0)
        rend_ink = np.count_nonzero(rend_bin == 0)
        density_ratio = (rend_ink / max(orig_ink, 1)) if orig_ink > 0 else 0.0

        page_score.block_scores.append(BlockScore(
            ink_overlap=ink_overlap,
            ssim=ssim_val,
            text_density_ratio=density_ratio,
            bbox_area_px=area,
            evaluated=True,
            block_type="table",
            ink_recall=ink_recall,
            ink_precision=ink_precision,
            winning_renderer="table",
        ))
        if self.config.full_page_mode:
            fp_rendered.append((px_x0, px_y0, px_x1, px_y1, rend_bin))

    def _score_text_block(
        self,
        block: dict,
        img_array: np.ndarray,
        scale_x: float,
        scale_y: float,
        img_w: int,
        img_h: int,
        page_score: PageScore,
        fp_rendered: list[tuple[int, int, int, int, np.ndarray]],
        page_idx: int,
    ) -> None:
        """Score a text / equation / heading block.

        Runs gates (empty, oversized, diagram), text render, optional
        three-way pdflatex contest for equations, final F1 / SSIM / density
        against the (optionally band-masked) original binarization.
        """
        block_type = block.get("type", "unknown")
        text = block.get("text", "")
        bbox = block.get("bbox")

        # Empty-text and missing-bbox gates (heading blocks with no bbox fall here)
        simplified_text = simplify_latex(text) if text else ""
        if not simplified_text or not simplified_text.strip():
            page_score.empty_block_count += 1
            page_score.block_scores.append(self._unevaluated_score(block_type))
            return
        if bbox is None:
            page_score.skipped_block_count += 1
            page_score.block_scores.append(self._unevaluated_score(block_type))
            return

        # Normalized bbox -> clamped pixel rect
        px_x0 = int(bbox[0] * scale_x)
        px_y0 = int(bbox[1] * scale_y)
        px_x1 = int(bbox[2] * scale_x)
        px_y1 = int(bbox[3] * scale_y)
        px_x0 = max(0, min(px_x0, img_w - 1))
        px_y0 = max(0, min(px_y0, img_h - 1))
        px_x1 = max(px_x0 + 1, min(px_x1, img_w))
        px_y1 = max(px_y0 + 1, min(px_y1, img_h))

        w_px = px_x1 - px_x0
        h_px = px_y1 - px_y0
        area = w_px * h_px
        if area < self.config.min_block_area_px:
            page_score.skipped_block_count += 1
            page_score.block_scores.append(
                self._unevaluated_score(block_type, bbox_area_px=area))
            return

        # Density gate: reject blocks whose text is absurdly long for the bbox
        # (e.g. Docling merging multiple paragraphs).  At 300 DPI a glyph is
        # ~10x30 px = ~300 px^2; the TEXT_AREA_PER_CHAR_PX2 divisor allows
        # about 3x realistic packing before flagging the block as merged.
        max_reasonable_chars = max(
            TEXT_MIN_REASONABLE_CHARS, area // TEXT_AREA_PER_CHAR_PX2,
        )
        if len(simplified_text) > max_reasonable_chars:
            logger.debug(
                "Skipping oversized text block: %d chars for %d px area",
                len(simplified_text), area,
            )
            page_score.skipped_block_count += 1
            page_score.block_scores.append(
                self._unevaluated_score(block_type, bbox_area_px=area))
            return

        # Formula-heavy flag (stat only, skip gate is legacy)
        stripped = strip_latex(text)
        is_formula_heavy = (
            len(stripped.strip()) < len(text.strip()) * self.config.formula_min_text_ratio
        )
        if is_formula_heavy and self.config.skip_formula_heavy:
            page_score.skipped_block_count += 1
            page_score.block_scores.append(
                self._unevaluated_score(
                    block_type, bbox_area_px=area, is_formula_heavy=True))
            return

        orig_crop = img_array[px_y0:px_y1, px_x0:px_x1]

        # Diagram detection -- applies to text and equation blocks
        if self.config.detect_diagrams and block_type in ("text", "equation"):
            diagram_conf = self._text.is_diagram_block(orig_crop, text)
            if diagram_conf > DIAGRAM_SKIP_CONFIDENCE:
                page_score.skipped_block_count += 1
                page_score.block_scores.append(
                    self._unevaluated_score(
                        "diagram_detected", bbox_area_px=area,
                        text_density_ratio=diagram_conf))
                return

        layout = self._text.analyse_text_layout(orig_crop)
        orig_bin = self._metrics.binarize(orig_crop, normalize_bg=True)

        # Honest mode: source-free flow at the calibrated per-book typography.
        ct_flow = self.config.ct_flow_source_free and self.config.typography is not None
        profile = self.config.typography
        try:
            if ct_flow:
                if block_type == "heading":
                    band_h_norm = float(bbox[3] - bbox[1])  # bbox is 0-1000 -> per-mille
                    font_px = profile.heading_px(band_h_norm, scale_y)
                else:
                    font_px = profile.body_px(scale_y)
                pitch_px = profile.pitch_px(scale_y)
                line_h = max(font_px + 1, int(round(font_px * profile.body_leading)))
                text_rendered, n_drawn, n_total = self._text.render_text_consistent(
                    simplified_text, w_px, h_px,
                    font_px=font_px, pitch_px=pitch_px, line_h=line_h,
                )
                if n_total > n_drawn:
                    page_score.clipped_block_count += 1
            else:
                text_rendered = self._text.render_text_in_bbox(
                    simplified_text, w_px, h_px, orig_crop, layout=layout,
                    orig_bin=orig_bin,
                )
        except Exception:
            logger.warning(
                "Render failed (page %d, %d chars, %dx%d px), skipping",
                page_idx, len(simplified_text), w_px, h_px,
            )
            page_score.skipped_block_count += 1
            page_score.block_scores.append(
                self._unevaluated_score(block_type, bbox_area_px=area))
            return

        # Band mask (reused by three-way scoring + final masking).
        # Honest mode (ct_skip_band_mask) does NOT mask the source with
        # source-detected bands -- scores against the unmasked original.
        if layout.has_non_text_bands:
            page_score.mixed_block_count += 1
        band_mask = None
        if (self.config.mask_mixed_content and not self.config.ct_skip_band_mask
                and layout.has_non_text_bands and layout.bands):
            band_mask = Metrics.build_band_mask(
                layout.bands, orig_bin.shape[0], orig_bin.shape[1],
            )

        ct_body_px = (
            profile.body_px(scale_y)
            if (self.config.ct_eq_pdflatex_only and profile is not None)
            else None
        )
        rendered, winning_renderer, score_text_raw, score_pdf_raw, score_comp_raw = (
            self._pick_best_equation_rendering(
                block=block,
                block_type=block_type,
                text_rendered=text_rendered,
                w_px=w_px,
                h_px=h_px,
                orig_bin=orig_bin,
                band_mask=band_mask,
                layout=layout,
                page_score=page_score,
                ct_body_px=ct_body_px,
            )
        )

        rend_bin = self._metrics.binarize(rendered, normalize_bg=False)

        masked_final = False
        if band_mask is not None:
            orig_bin = np.where(band_mask, orig_bin, 255).astype(np.uint8)
            masked_final = True
            page_score.masked_block_count += 1

        ink_overlap, ink_recall, ink_precision = self._metrics.compute_f1_overlap(orig_bin, rend_bin)
        ssim_val = Metrics.compute_ssim(orig_bin, rend_bin)
        orig_ink = np.count_nonzero(orig_bin == 0)
        rend_ink = np.count_nonzero(rend_bin == 0)
        density_ratio = (rend_ink / max(orig_ink, 1)) if orig_ink > 0 else 0.0

        page_score.block_scores.append(BlockScore(
            ink_overlap=ink_overlap,
            ssim=ssim_val,
            text_density_ratio=density_ratio,
            bbox_area_px=area,
            evaluated=True,
            block_type=block_type,
            mixed_content_masked=masked_final,
            is_formula_heavy=is_formula_heavy,
            ink_recall=ink_recall,
            ink_precision=ink_precision,
            winning_renderer=winning_renderer,
            score_text_f1=score_text_raw,
            score_pdflatex_f1=score_pdf_raw,
            score_composite_f1=score_comp_raw,
        ))
        if self.config.full_page_mode:
            fp_rendered.append((px_x0, px_y0, px_x1, px_y1, rend_bin))

    def _score_photo_block(
        self,
        block: dict,
        img_array: np.ndarray,
        sx0: int, sy0: int, sx1: int, sy1: int,
        skip_area: int,
        page_score: PageScore,
        fp_image_masks: list[tuple[int, int, int, int]],
    ) -> None:
        """Score a photo block: 20% detection + 20% placement + 60% VLM description."""
        description = block.get("_photo_description", "")
        desc_score = FigureRenderer.score_photo_description(
            img_array[sy0:sy1, sx0:sx1], description,
        ) if description else 0.0
        photo_score = (
            PHOTO_SCORE_DETECTION
            + PHOTO_SCORE_PLACEMENT
            + PHOTO_SCORE_DESCRIPTION * desc_score
        )
        page_score.block_scores.append(BlockScore(
            ink_overlap=photo_score,
            ssim=0.0, text_density_ratio=0.0,
            bbox_area_px=skip_area, evaluated=True,
            block_type="photo",
        ))
        if self.config.full_page_mode:
            fp_image_masks.append((sx0, sy0, sx1, sy1))

    def _score_rendered_image_crop(
        self,
        orig_crop: np.ndarray,
        rendered: np.ndarray,
        skip_area: int,
        block_type: str,
        page_score: PageScore,
    ) -> np.ndarray:
        """Binarize + align + score (F1 / SSIM / density) a rendered image block.

        Appends a BlockScore to ``page_score`` and returns the aligned binarized
        rendering so the caller can feed it into full-page composite mode.
        """
        orig_bin = self._metrics.binarize(orig_crop, normalize_bg=True)
        rend_bin = self._metrics.binarize(rendered, normalize_bg=False)
        # Honest mode strips the figure xcorr (it shifts the rendered figure onto
        # the source ink -- the same compensation family as the text/equation
        # xcorr). Figures then sit at their detected bbox like every other block.
        # Exception: the "peeking" avenue-run contrast deliberately restores the
        # full legacy peeking family for line graphs (crop-guided render + xcorr);
        # technical drawings have no peeking mechanism and never xcorr here.
        if not self.config.consistent_typography_mode or (
                self.config.ct_figure_recreation == "peeking"
                and block_type == "line_graph"):
            rend_bin = self._figure.align_graph_xcorr(orig_bin, rend_bin)

        ink_overlap, ink_recall, ink_precision = (
            self._metrics.compute_f1_overlap(orig_bin, rend_bin))
        ssim_val = Metrics.compute_ssim(orig_bin, rend_bin)
        orig_ink = np.count_nonzero(orig_bin == 0)
        rend_ink = np.count_nonzero(rend_bin == 0)
        density_ratio = rend_ink / max(orig_ink, 1)

        page_score.block_scores.append(BlockScore(
            ink_overlap=ink_overlap, ssim=ssim_val,
            text_density_ratio=density_ratio,
            bbox_area_px=skip_area, evaluated=True,
            block_type=block_type,
            ink_recall=ink_recall,
            ink_precision=ink_precision,
        ))
        return rend_bin

    def _score_line_graph_block(
        self,
        block: dict,
        img_array: np.ndarray,
        sx0: int, sy0: int, sx1: int, sy1: int,
        skip_area: int,
        page_score: PageScore,
        fp_rendered: list[tuple[int, int, int, int, np.ndarray]],
        source_free: bool = False,
    ) -> None:
        """Render and score a line_graph figure block.

        ``source_free=True`` renders purely from the extracted data (no
        plot-area/grid detection on the source crop) — the honest avenue-run
        variant; the default crop-guided render is the legacy/peeking path.
        """
        graph_ext = block.get("graph_extraction", {})
        if graph_ext.get("status") != "success":
            # No floor: graph rendering is either accurate or wrong,
            # unlike photo detection which only confirms presence.
            page_score.block_scores.append(
                self._unevaluated_score("image", bbox_area_px=skip_area))
            return

        orig_crop = img_array[sy0:sy1, sx0:sx1]
        rendered = self._figure.render_line_graph(
            graph_ext["data"], sx1 - sx0, sy1 - sy0,
            orig_crop=None if source_free else orig_crop,
        )
        if rendered is None:
            page_score.figure_render_fail_count += 1
            page_score.block_scores.append(
                self._unevaluated_score("image", bbox_area_px=skip_area))
            return

        rend_bin = self._score_rendered_image_crop(
            orig_crop, rendered, skip_area, "line_graph", page_score,
        )
        if self.config.full_page_mode:
            fp_rendered.append((sx0, sy0, sx1, sy1, rend_bin))

    def _score_technical_drawing_block(
        self,
        block: dict,
        img_array: np.ndarray,
        sx0: int, sy0: int, sx1: int, sy1: int,
        skip_area: int,
        page_score: PageScore,
        fp_rendered: list[tuple[int, int, int, int, np.ndarray]],
    ) -> None:
        """Render and score a technical_drawing figure block (SVG input)."""
        drawing_ext = block.get("drawing_extraction", {})
        svg_str = drawing_ext.get("svg", "") if drawing_ext.get("status") == "success" else ""
        if not svg_str:
            page_score.block_scores.append(
                self._unevaluated_score("image", bbox_area_px=skip_area))
            return

        orig_crop = img_array[sy0:sy1, sx0:sx1]
        rendered = FigureRenderer.render_technical_drawing(
            svg_str, sx1 - sx0, sy1 - sy0,
        )
        if rendered is None:
            page_score.figure_render_fail_count += 1
            page_score.block_scores.append(
                self._unevaluated_score("image", bbox_area_px=skip_area))
            return

        rend_bin = self._score_rendered_image_crop(
            orig_crop, rendered, skip_area, "technical_drawing", page_score,
        )
        if self.config.full_page_mode:
            fp_rendered.append((sx0, sy0, sx1, sy1, rend_bin))

    def _pick_best_equation_rendering(
        self,
        *,
        block: dict,
        block_type: str,
        text_rendered: np.ndarray,
        w_px: int,
        h_px: int,
        orig_bin: np.ndarray,
        band_mask: np.ndarray | None,
        layout: "TextLayout",
        page_score: PageScore,
        ct_body_px: int | None = None,
    ) -> tuple[np.ndarray, str, float, float, float]:
        """Try pdflatex + composite for equations; pick the best F1 vs orig_bin.

        For non-equation blocks (or when pdflatex is disabled / block is too
        tall), returns the text rendering unchanged with raw scores of -1.0.

        Mutates ``page_score`` counters (``pdflatex_win_count`` /
        ``pdflatex_lose_count`` / ``pdflatex_fail_count``) as a side effect.

        Returns
        -------
        (rendered, winning_renderer, score_text_f1, score_pdf_f1, score_comp_f1)
            ``winning_renderer`` is one of ``"text"``, ``"pdflatex"``,
            ``"composite"``.
        """
        pdflatex_applies = (
            block_type == "equation"
            and self.config.use_pdflatex_for_equations
            and h_px <= self.config.pdflatex_max_block_height
        )
        if not pdflatex_applies:
            return text_rendered, "text", -1.0, -1.0, -1.0

        # Honest mode: pdflatex-only at body scale, NO F1 best-of-three pick and
        # NO bbox rescale / xcorr. Keep the compile fallback chain (not a leak):
        # pdflatex(raw) -> pdflatex(UniMERNet-cleaned) -> consistent-typography text.
        if self.config.ct_eq_pdflatex_only and ct_body_px:
            result = self._equation.render_equation_consistent(
                block, w_px, h_px, ct_body_px,
            )
            if result is None:
                page_score.pdflatex_fail_count += 1
                return text_rendered, "text_fallback_ct", -1.0, -1.0, -1.0
            canvas, clipped, cleanup_fired = result
            if clipped:
                page_score.clipped_block_count += 1
            if cleanup_fired:
                page_score.cleanup_fired_count += 1
            page_score.pdflatex_win_count += 1
            return canvas, "pdflatex_ct", -1.0, -1.0, -1.0

        pdflatex_rendered = self._equation.render_equation_block(
            block, w_px, h_px, orig_bin, layout=layout,
        )
        if pdflatex_rendered is None:
            page_score.pdflatex_fail_count += 1
            return text_rendered, "text", -1.0, -1.0, -1.0

        # Three-way F1 contest: binarize all renderings, compare with band-masked orig
        rend_bin_pdf = self._metrics.binarize(pdflatex_rendered, normalize_bg=False)
        rend_bin_text = self._metrics.binarize(text_rendered, normalize_bg=False)
        cmp_orig = orig_bin.copy()
        if band_mask is not None:
            cmp_orig = np.where(band_mask, cmp_orig, 255).astype(np.uint8)

        f1_pdf, _, _ = self._metrics.compute_f1_overlap(cmp_orig, rend_bin_pdf)
        f1_text, _, _ = self._metrics.compute_f1_overlap(cmp_orig, rend_bin_text)

        if self.config.use_pdflatex_composite and layout.bands:
            composite = self._equation.composite_equation_text(
                text_rendered, pdflatex_rendered, layout,
            )
            comp_bin = self._metrics.binarize(composite, normalize_bg=False)
            f1_comp, _, _ = self._metrics.compute_f1_overlap(cmp_orig, comp_bin)
        else:
            composite = None
            f1_comp = -1.0

        best_f1 = max(f1_text, f1_pdf, f1_comp)
        if best_f1 == f1_comp and composite is not None:
            page_score.pdflatex_win_count += 1
            return composite, "composite", f1_text, f1_pdf, f1_comp
        if best_f1 == f1_pdf:
            page_score.pdflatex_win_count += 1
            return pdflatex_rendered, "pdflatex", f1_text, f1_pdf, f1_comp
        page_score.pdflatex_lose_count += 1
        return text_rendered, "text", f1_text, f1_pdf, f1_comp

    # ------------------------------------------------------------------
    # Full-page composite scoring
    # ------------------------------------------------------------------

    def _composite_full_page(
        self,
        img_w: int,
        img_h: int,
        img_array: np.ndarray,
        rendered_blocks: list[tuple[int, int, int, int, np.ndarray]],
        image_mask_bboxes: list[tuple[int, int, int, int]],
    ) -> tuple[float, float, float]:
        """Compute full-page F1 by compositing rendered blocks vs. raw scan.

        Renders all successfully-evaluated blocks onto a white canvas at their
        detected positions, binarizes the full original page, masks out image
        block regions, and computes F1 with a tighter DT threshold.

        Returns (f1, recall, precision).
        """
        import cv2  # local import — cv2 not needed elsewhere in evaluator

        # Binarize full original page with capped kernel (max 51px)
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()

        bg_median = np.median(gray)
        if bg_median < 240:
            h, w = gray.shape
            k = min(51, max(15, min(h, w) // 20)) | 1  # cap at 51
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            background = np.maximum(background, 1)
            gray = np.clip(
                gray.astype(float) / background.astype(float) * 255,
                0, 255,
            ).astype(np.uint8)

        _, orig_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink_ratio = np.count_nonzero(orig_bin == 0) / max(orig_bin.size, 1)
        if ink_ratio < 0.02 or ink_ratio > 0.98:
            _, orig_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Mask image block regions (photos, failed figures) in original
        for x0, y0, x1, y1 in image_mask_bboxes:
            orig_bin[y0:y1, x0:x1] = 255

        # Composite rendered blocks onto white canvas
        rend_canvas = np.full((img_h, img_w), 255, dtype=np.uint8)
        for x0, y0, x1, y1, rend_bin in rendered_blocks:
            bh, bw = rend_bin.shape[:2]
            # Clamp to canvas bounds and block dimensions
            paste_h = min(bh, img_h - y0, y1 - y0)
            paste_w = min(bw, img_w - x0, x1 - x0)
            if paste_h > 0 and paste_w > 0:
                rend_canvas[y0:y0 + paste_h, x0:x0 + paste_w] = np.minimum(
                    rend_canvas[y0:y0 + paste_h, x0:x0 + paste_w],
                    rend_bin[:paste_h, :paste_w],
                )

        # Compute F1 with the full-page DT threshold. In honest mode with
        # ct_tau_body_frac>0, tau is tied to the calibrated body size
        # (tau = frac * body_px at this page's scale) so the tolerance is the
        # same fraction of the text height regardless of scan resolution --
        # otherwise a fixed-pixel tau scores high-DPI books far more strictly.
        eff_tau = self.config.dt_threshold_full_page
        if (self.config.consistent_typography_mode
                and self.config.ct_tau_body_frac > 0
                and self.config.typography is not None):
            scale_y = img_h / BBOX_SCALE
            eff_tau = self.config.ct_tau_body_frac * self.config.typography.body_px(scale_y)
        saved_dt = self.config.dt_threshold
        # QualityEvaluationConfig is frozen, so use object.__setattr__
        object.__setattr__(self.config, "dt_threshold", eff_tau)
        try:
            f1, recall, precision = self._metrics.compute_f1_overlap(orig_bin, rend_canvas)
        finally:
            object.__setattr__(self.config, "dt_threshold", saved_dt)

        if self.config.ct_capture_confusion:
            self._last_confusion = (orig_bin, rend_canvas, eff_tau)

        return f1, recall, precision

    # ------------------------------------------------------------------
    # Overlay / report delegation
    # ------------------------------------------------------------------

    def generate_overlay(
        self,
        source_image_path: Path,
        page_score: PageScore,
        blocks: list[dict],
        output_path: Path,
    ) -> Path:
        """Delegate overlay generation to OverlayGenerator."""
        return self._overlay.generate_overlay(
            source_image_path, page_score, blocks, output_path,
        )

    def generate_report(
        self,
        evaluation: BookEvaluation,
        output_path: Path,
    ) -> Path:
        """Delegate report generation to OverlayGenerator."""
        return self._overlay.generate_report(evaluation, output_path)

    # ------------------------------------------------------------------
    # Facade methods for backward compatibility
    # ------------------------------------------------------------------

    def _binarize(self, *args, **kwargs):
        return self._metrics.binarize(*args, **kwargs)

    def _render_line_graph(self, *args, **kwargs):
        return self._figure.render_line_graph(*args, **kwargs)

    def _align_graph_xcorr(self, *args, **kwargs):
        return self._figure.align_graph_xcorr(*args, **kwargs)

    def _compute_f1_overlap(self, *args, **kwargs):
        return self._metrics.compute_f1_overlap(*args, **kwargs)

    def _compute_ssim(self, *args, **kwargs):
        return Metrics.compute_ssim(*args, **kwargs)

    def _build_band_mask(self, *args, **kwargs):
        return Metrics.build_band_mask(*args, **kwargs)

    def _xcorr_shift(self, *args, **kwargs):
        return xcorr_shift(*args, **kwargs)

    def _shift_image(self, *args, **kwargs):
        return shift_image(*args, **kwargs)

    def _render_text_in_bbox(self, *args, **kwargs):
        return self._text.render_text_in_bbox(*args, **kwargs)

    def _analyse_text_layout(self, *args, **kwargs):
        return self._text.analyse_text_layout(*args, **kwargs)

    def _is_diagram_block(self, *args, **kwargs):
        return self._text.is_diagram_block(*args, **kwargs)

    def _render_equation_block(self, *args, **kwargs):
        return self._equation.render_equation_block(*args, **kwargs)

    def _render_formula_pdflatex(self, *args, **kwargs):
        return self._equation.render_formula_pdflatex(*args, **kwargs)

    def _composite_equation_text(self, *args, **kwargs):
        return self._equation.composite_equation_text(*args, **kwargs)

    def _render_table_in_bbox(self, *args, **kwargs):
        return self._table.render_table_in_bbox(*args, **kwargs)

    def _render_grid_in_bbox(self, *args, **kwargs):
        return self._table.render_grid_in_bbox(*args, **kwargs)

    def _detect_table_grid_lines(self, *args, **kwargs):
        return self._table.detect_table_grid_lines(*args, **kwargs)

    def _classify_table_cells(self, *args, **kwargs):
        return self._table.classify_table_cells(*args, **kwargs)

    def _parse_html_table(self, *args, **kwargs):
        return self._table.parse_html_table(*args, **kwargs)

    def _parse_markdown_table(self, *args, **kwargs):
        return self._table.parse_markdown_table(*args, **kwargs)

    def _get_font(self, *args, **kwargs):
        return self._fonts.get_font(*args, **kwargs)

    def _get_bitmap_renderer(self):
        return self._fonts.get_bitmap_renderer()

    def _compute_ink_overlap(self, *args, **kwargs):
        return self._metrics.compute_ink_overlap(*args, **kwargs)

    def _get_font_advance(self, *args, **kwargs):
        return FontCache.get_font_advance(*args, **kwargs)

    def _find_font_size_for_pitch(self, *args, **kwargs):
        return self._fonts.find_font_size_for_pitch(*args, **kwargs)

    def _get_hybrid_renderer(self):
        return self._fonts.get_hybrid_renderer()

    def _get_formula_cleanup(self):
        return self._equation.get_formula_cleanup()

    @property
    def _missing_templates(self) -> set[str]:
        return self._fonts._missing_templates

    @_missing_templates.setter
    def _missing_templates(self, value: set[str]) -> None:
        self._fonts._missing_templates = value

    @property
    def _bitmap_renderer(self):
        return self._fonts._bitmap_renderer

    @_bitmap_renderer.setter
    def _bitmap_renderer(self, value):
        self._fonts._bitmap_renderer = value

    @property
    def _formula_cleanup_loaded(self) -> bool:
        return self._equation._formula_cleanup_loaded

    @_formula_cleanup_loaded.setter
    def _formula_cleanup_loaded(self, value: bool) -> None:
        self._equation._formula_cleanup_loaded = value

    _render_technical_drawing = staticmethod(FigureRenderer.render_technical_drawing)
    _score_photo_description = staticmethod(FigureRenderer.score_photo_description)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_content_lists(
        self, run_dir: Path,
    ) -> list[tuple[Path, list[dict]]]:
        """Glob for content_list.json files in a run directory.

        Prefers merged content_list (post-merge text with Docling corrections)
        over MinerU raw output. Falls back to MinerU if no merged files exist.

        Raises FileNotFoundError if none found.
        Skips corrupt JSON with a warning.
        """
        # Prefer merged content_list (reflects final pipeline output)
        merged_pattern = "merged/*_content_list.json"
        paths = sorted(run_dir.glob(merged_pattern))
        if paths:
            logger.info("Using merged content_list (%d files)", len(paths))
        else:
            # Fall back to MinerU raw output
            pattern = "mineru_output/*/ocr/*content_list.json"
            paths = sorted(run_dir.glob(pattern))
        if not paths:
            # Fallback to recursive glob for non-standard layouts
            paths = sorted(run_dir.glob("**/*content_list.json"))
            # Deduplicate: keep only the shallowest path per batch name
            seen_batches: dict[str, Path] = {}
            for p in paths:
                batch_match = re.search(r'batch_\d+', p.name)
                key = batch_match.group(0) if batch_match else p.name
                if key not in seen_batches or len(p.parts) < len(seen_batches[key].parts):
                    seen_batches[key] = p
            paths = sorted(seen_batches.values())

        if not paths:
            raise FileNotFoundError(
                f"No *content_list.json files found in {run_dir}"
            )

        results = []
        for p in paths:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                results.append((p, data))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("Skipping corrupt content_list %s: %s", p, exc)

        return results

    def _load_content_lists_multi(
        self,
        run_dirs: list[Path],
    ) -> list[tuple[Path, list[dict]]]:
        """Load content_lists from multiple run dirs, dedup by batch number.

        Caller should pass run_dirs from a single engine (use
        ImageProcessorService.discover_runs to pre-filter). Within the input
        set, dedup keeps the entry from the chronologically latest run dir
        (parsed timestamp from run name, not lexicographic compare).

        Returns list of (path, blocks) tuples sorted by batch number.
        """
        run_re = re.compile(r"^run_(\d{8}_\d{6})_")
        by_batch: dict[int, tuple[datetime, Path, list[dict]]] = {}
        for run_dir in run_dirs:
            m = run_re.match(run_dir.name)
            if not m:
                logger.warning("Cannot parse timestamp from %s", run_dir.name)
                continue
            ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            try:
                cls = self._load_content_lists(run_dir)
            except FileNotFoundError:
                logger.warning("No content_list in %s", run_dir.name)
                continue
            for cl_path, blocks in cls:
                bn = self._parse_batch_num(cl_path)
                if bn is None:
                    logger.warning(
                        "Cannot parse batch number from %s", cl_path.name
                    )
                    continue
                source = "merged" if "merged" in str(cl_path) else "raw"
                logger.info(
                    "Loaded batch %d from %s (%s)", bn, run_dir.name, source
                )
                if bn not in by_batch or ts > by_batch[bn][0]:
                    by_batch[bn] = (ts, cl_path, blocks)
        return [(path, blocks) for _, (_, path, blocks) in sorted(by_batch.items())]

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        """Write JSON atomically via tmp file + os.replace."""
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        os.replace(tmp_path, path)
