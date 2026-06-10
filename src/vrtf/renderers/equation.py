"""Equation rendering via pdflatex with hybrid bitmap glyph stamping.

Extracted from QualityEvaluationService to isolate the LaTeX compilation
and glyph-stamping pipeline into a dedicated renderer module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from vrtf.config import QualityEvaluationConfig
from vrtf.metric import Metrics, xcorr_shift
from vrtf.models import TextLayout
from vrtf.utils.font import FontCache
from vrtf.utils.latex import simplify_latex
from vrtf.utils.unicode_aliases import UNICODE_ALIASES as _UNICODE_ALIASES

logger = logging.getLogger(__name__)

_HYBRID_MIN_GLYPH_PX = 8  # skip glyphs smaller than this (subscripts too small to stamp)


class EquationRenderer:
    """Render equation blocks via pdflatex with optional hybrid bitmap stamping.

    Parameters
    ----------
    config : QualityEvaluationConfig
        Evaluation configuration (pdflatex flags, hybrid stamping, etc.).
    fonts : FontCache
        Shared font/renderer cache (provides hybrid renderer access).
    metrics : Metrics
        Image-level metric helpers (band masking, overlap computation).
    """

    def __init__(
        self,
        config: QualityEvaluationConfig,
        fonts: FontCache,
        metrics: Metrics,
    ) -> None:
        self.config = config
        self.fonts = fonts
        self.metrics = metrics
        self._formula_cleanup: dict[str, str] | None = None
        self._formula_cleanup_loaded: bool = False
        self._pdflatex_available: bool | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_equation_block(
        self, block: dict, w_px: int, h_px: int, orig_bin: np.ndarray,
        layout: TextLayout | None = None,
    ) -> np.ndarray | None:
        """Render equation block via pdflatex with UniMERNet fallback.

        Args:
            block: OCR block dict with "text" (MinerU LaTeX) and "img_path".
            w_px, h_px: bbox pixel dimensions.
            orig_bin: pre-binarized original crop for xcorr alignment.

        Returns:
            Binary canvas (0=ink, 255=bg) sized (h_px, w_px), or None on failure.
        """
        raw_latex = block.get("text", "")
        img_path = block.get("img_path", "")

        # Try MinerU original first (92% success), then UniMERNet cleaned (95% combined)
        rendered = self.render_formula_pdflatex(
            raw_latex, h_px, display_mode=True, hybrid_stamp=True,
        )

        if rendered is None and img_path:
            fc = self.get_formula_cleanup()
            cleaned = fc.get(img_path, "")
            if cleaned:
                rendered = self.render_formula_pdflatex(
                    cleaned, h_px, display_mode=True, hybrid_stamp=True,
                )

        if rendered is None:
            return None  # caller falls back to text pipeline

        # Scale to fit BOTH bbox dimensions (preserve aspect ratio)
        rh, rw = rendered.shape
        scale_h = h_px / rh if rh > 0 else 1.0
        scale_w = w_px / rw if rw > 0 else 1.0
        scale = min(scale_h, scale_w)
        if scale != 1.0 and scale > 0:
            new_h = max(1, int(rh * scale))
            new_w = max(1, int(rw * scale))
            rendered = cv2.resize(rendered, (new_w, new_h), interpolation=cv2.INTER_AREA)
            _, rendered = cv2.threshold(rendered, 128, 255, cv2.THRESH_BINARY)
            rh, rw = rendered.shape

        # Place on bbox-sized canvas (centered)
        canvas = np.full((h_px, w_px), 255, dtype=np.uint8)
        y_off = max(0, (h_px - rh) // 2)
        x_off = max(0, layout.x_left if layout is not None else (w_px - rw) // 2)
        paste_h = min(rh, h_px - y_off)
        paste_w = min(rw, w_px - x_off)
        canvas[y_off:y_off + paste_h, x_off:x_off + paste_w] = rendered[:paste_h, :paste_w]

        # Align via xcorr (scale max_shift to block size)
        max_shift = min(50, h_px // 4, w_px // 4)
        dy, dx = xcorr_shift(orig_bin, canvas, max_shift=max_shift)
        if dy != 0 or dx != 0:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            canvas = cv2.warpAffine(canvas, M, (w_px, h_px), borderValue=255)
            # Clean border artifacts from interpolation
            _, canvas = cv2.threshold(canvas, 128, 255, cv2.THRESH_BINARY)

        return canvas

    def render_equation_consistent(
        self, block: dict, w_px: int, h_px: int, body_px: int, x_left: int = 0,
    ) -> tuple[np.ndarray, bool, bool] | None:
        """Honest equation render: pdflatex at body scale, placed top-left.

        Renders via pdflatex at a DPI tied to the per-book body font size (so the
        equation's font matches body text), does NOT rescale to the bbox and does
        NOT xcorr-align to the source. Placed at the bbox top-left; ink that
        overflows the bbox is clipped (precision loss), under-fill stays blank
        (recall loss). Keeps the pdflatex -> UniMERNet compile fallback (a compile
        fallback is not source leakage; only the F1 best-of-three pick and the
        bbox rescale/xcorr are removed).

        Returns ``(canvas, clipped, cleanup_fired)`` or ``None`` if pdflatex (and
        the cleanup fallback) failed to compile -- the caller then falls back to
        rendering the LaTeX as consistent-typography text. ``cleanup_fired`` is
        True iff the raw LaTeX failed and the UniMERNet-cleaned variant compiled.
        """
        raw_latex = block.get("text", "")
        img_path = block.get("img_path", "")
        # 10pt standalone math: font_px ~= 10 * dpi / 72  ->  dpi = 7.2 * body_px
        dpi = max(150, min(600, int(round(body_px * 7.2))))

        cleanup_fired = False
        rend = self.render_formula_pdflatex(
            raw_latex, h_px, display_mode=True, hybrid_stamp=True,
            dpi_override=dpi, rescale=False,
        )
        if rend is None and img_path:
            cleaned = self.get_formula_cleanup().get(img_path, "")
            if cleaned:
                rend = self.render_formula_pdflatex(
                    cleaned, h_px, display_mode=True, hybrid_stamp=True,
                    dpi_override=dpi, rescale=False,
                )
                cleanup_fired = rend is not None
        if rend is None:
            return None

        canvas = np.full((h_px, w_px), 255, dtype=np.uint8)
        rh, rw = rend.shape
        x0 = max(0, x_left)
        ph = min(rh, h_px)
        pw = min(rw, w_px - x0)
        clipped = rh > h_px or rw > (w_px - x0)
        if ph > 0 and pw > 0:
            canvas[0:ph, x0:x0 + pw] = np.minimum(
                canvas[0:ph, x0:x0 + pw], rend[:ph, :pw],
            )
        return canvas, clipped, cleanup_fired

    def render_formula_pdflatex(
        self, latex: str, target_h: int, *,
        display_mode: bool = False, hybrid_stamp: bool = False,
        dpi_override: int | None = None, rescale: bool = True,
    ) -> np.ndarray | None:
        """Render formula via pdflatex + pdftoppm.

        Args:
            hybrid_stamp: If True AND use_hybrid_glyph_stamping config is enabled,
                stamp bitmap templates over CM glyphs. Only True from equation block path.
            dpi_override: render at this DPI instead of deriving it from target_h
                (honest mode ties DPI to the body font size).
            rescale: if False, return the natural-size render at the given DPI
                (no scale-to-target_h) -- used by the honest equation path so the
                equation font matches body and the height is its intrinsic height.
        """
        import subprocess
        import tempfile

        if self._pdflatex_available is None:
            import shutil
            self._pdflatex_available = bool(
                shutil.which("pdflatex") and shutil.which("pdftoppm")
            )
        if not self._pdflatex_available:
            if not getattr(self, "_warned_no_pdflatex", False):
                logger.warning(
                    "pdflatex/pdftoppm not found on PATH — equation blocks "
                    "fall back to plain-text rendering, which lowers scores. "
                    "Install TeX (e.g. apt install texlive-latex-base "
                    "texlive-latex-extra poppler-utils) for the full metric.")
                self._warned_no_pdflatex = True
            return None

        # Strip delimiters
        clean = latex.strip()
        for delim in ('$$', '$'):
            if clean.startswith(delim) and clean.endswith(delim):
                clean = clean[len(delim):-len(delim)].strip()
                break

        if not clean:
            return None

        if display_mode:
            content = f"$\\displaystyle{{{clean}}}$"
        else:
            content = f"${clean}$"

        dpi = dpi_override if dpi_override else max(150, min(600, 72 * target_h // 20))
        tex_source = (
            r"\documentclass[border=2pt]{standalone}" "\n"
            r"\usepackage{amsmath,amssymb,bm}" "\n"
            r"\begin{document}" "\n"
            f"{content}\n"
            r"\end{document}" "\n"
        )

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tex_path = Path(tmpdir) / "formula.tex"
                tex_path.write_text(tex_source, encoding="utf-8")

                result = subprocess.run(
                    ["pdflatex", "--no-shell-escape", "-interaction=nonstopmode",
                     "-halt-on-error", "-output-directory", tmpdir,
                     str(tex_path)],
                    capture_output=True, timeout=5,
                )
                if result.returncode != 0:
                    return None

                pdf_path = Path(tmpdir) / "formula.pdf"
                if not pdf_path.exists():
                    return None

                png_prefix = Path(tmpdir) / "formula_out"
                result = subprocess.run(
                    ["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                     str(pdf_path), str(png_prefix)],
                    capture_output=True, timeout=5,
                )
                if result.returncode != 0:
                    return None

                png_path = Path(tmpdir) / "formula_out.png"
                if not png_path.exists():
                    return None

                img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return None

                h, w = img.shape
                if h <= 0:
                    return None
                if rescale:
                    # Scale to target height
                    scale = target_h / h
                    new_w = max(1, int(w * scale))
                    img = cv2.resize(img, (new_w, target_h),
                                     interpolation=cv2.INTER_AREA)
                    stamp_h = target_h
                else:
                    stamp_h = h  # honest mode: keep natural size at the chosen DPI
                _, binary = cv2.threshold(
                    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                # Hybrid glyph stamping: replace CM glyphs with bitmap templates
                if hybrid_stamp and self.config.use_hybrid_glyph_stamping:
                    binary = self._stamp_bitmap_glyphs(binary, pdf_path, stamp_h)
                return binary

        except (subprocess.TimeoutExpired, OSError):
            return None

    def get_formula_cleanup(self) -> dict[str, str]:
        """Lazy-load formula_cleanup.json (img_path -> cleaned LaTeX)."""
        if not self._formula_cleanup_loaded:
            path = self.config.formula_cleanup_path
            if path and not Path(path).exists():
                # Do NOT latch the loaded flag on a missing file: a typo'd path
                # would otherwise silently disable the fallback forever.
                logger.warning("formula_cleanup_path does not exist: %s", path)
                return {}
            self._formula_cleanup_loaded = True
            if path:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                self._formula_cleanup = {
                    img_path: entry["cleaned_latex"]
                    for img_path, entry in data.get("formulas", {}).items()
                    if entry.get("cleaned_latex")
                }
        return self._formula_cleanup or {}

    def composite_equation_text(
        self,
        text_canvas: np.ndarray,
        pdf_canvas: np.ndarray,
        layout: TextLayout,
    ) -> np.ndarray:
        """Composite: text pipeline for text-band rows, pdflatex for non-text rows."""
        h, w = text_canvas.shape[:2]
        if not layout.bands:
            return pdf_canvas.copy()
        band_mask = self.metrics.build_band_mask(layout.bands, h, w)
        # band_mask is 2D bool; expand to match canvas dimensions
        if text_canvas.ndim == 3:
            band_mask = band_mask[:, :, np.newaxis]
        return np.where(band_mask, text_canvas, pdf_canvas).astype(np.uint8)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stamp_bitmap_glyphs(
        self,
        binary: np.ndarray,
        pdf_path: Path,
        target_h: int,
    ) -> np.ndarray:
        """Stamp bitmap font templates over Computer Modern glyphs in pdflatex output.

        Uses pdfminer.six to extract character positions from the PDF, then
        composites the corresponding bitmap template at each position using
        np.minimum (additive ink only -- no erase step).

        Args:
            binary: (target_h, new_w) array, 0=ink, 255=bg.
            pdf_path: Path to the pdflatex-generated PDF.
            target_h: Target height in pixels.

        Returns:
            Modified binary array with bitmap templates stamped over CM glyphs.
        """
        renderer = self.fonts.get_hybrid_renderer()
        if renderer is None:
            return binary

        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTChar, LTTextBox, LTTextLine
        except ImportError:
            return binary

        try:
            canvas = binary.copy()
            stamped = 0
            unmapped = set()

            for page_layout in extract_pages(str(pdf_path)):
                page_h_pts = page_layout.bbox[3] - page_layout.bbox[1]
                if page_h_pts <= 0:
                    return binary
                pts_to_px = target_h / page_h_pts

                for element in page_layout:
                    if not isinstance(element, LTTextBox):
                        continue
                    for line in element:
                        if not isinstance(line, LTTextLine):
                            continue
                        for char in line:
                            if not isinstance(char, LTChar):
                                continue
                            text = char.get_text()
                            if len(text) != 1:
                                continue

                            # Apply Unicode aliases
                            ch = _UNICODE_ALIASES.get(text, text)

                            # Look up template
                            if ch not in renderer.templates:
                                if ch.strip():
                                    unmapped.add(ch)
                                continue

                            # Compute glyph dimensions in pixel space
                            x0, y0, x1, y1 = char.bbox
                            glyph_h = round((y1 - y0) * pts_to_px)
                            if glyph_h < _HYBRID_MIN_GLYPH_PX:
                                continue

                            nat_w = renderer.char_width_at_height(glyph_h)
                            if nat_w < 2:
                                continue

                            # Center position (PDF y-axis is bottom-up)
                            cx = round(((x0 + x1) / 2) * pts_to_px)
                            cy = round((page_h_pts - (y0 + y1) / 2) * pts_to_px)

                            # Stamp region bounds
                            sx0 = cx - nat_w // 2
                            sy0 = cy - glyph_h // 2
                            sx1 = sx0 + nat_w
                            sy1 = sy0 + glyph_h

                            # Clip to canvas bounds
                            img_h, img_w = canvas.shape
                            cx0 = max(0, sx0)
                            cy0 = max(0, sy0)
                            cx1 = min(img_w, sx1)
                            cy1 = min(img_h, sy1)
                            if cx1 <= cx0 or cy1 <= cy0:
                                continue

                            # Template region offset (for clipping)
                            tx0 = cx0 - sx0
                            ty0 = cy0 - sy0
                            tw = cx1 - cx0
                            th = cy1 - cy0

                            # Scale template to natural size and binarize
                            tmpl = renderer.templates[ch]
                            scaled = cv2.resize(tmpl, (nat_w, glyph_h),
                                                interpolation=cv2.INTER_AREA)
                            _, scaled_bin = cv2.threshold(
                                scaled, 128, 255, cv2.THRESH_BINARY,
                            )

                            # Stamp with np.minimum (additive ink, no erase)
                            region = scaled_bin[ty0:ty0 + th, tx0:tx0 + tw]
                            canvas[cy0:cy1, cx0:cx1] = np.minimum(
                                canvas[cy0:cy1, cx0:cx1], region,
                            )
                            stamped += 1

            if unmapped:
                logger.debug(
                    "Hybrid stamp: %d unmapped chars: %s",
                    len(unmapped),
                    ", ".join(f"U+{ord(c):04X}" for c in sorted(unmapped)),
                )
            logger.debug("Hybrid stamp: stamped %d glyphs", stamped)
            return canvas

        except Exception:
            logger.debug("Hybrid stamp failed, returning original", exc_info=True)
            return binary
