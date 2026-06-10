"""Text rendering for VRTF quality evaluation.

Renders OCR text back into bounding box regions using layout analysis,
pitch-aware font sizing, and per-line cross-correlation alignment against
original scan crops.
"""

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw

from vrtf.metric import shift_image, xcorr_shift
from vrtf.models import TextLayout
from vrtf.utils.latex import simplify_latex
from vrtf.utils.unicode_aliases import UNICODE_ALIASES as _UNICODE_ALIASES

if TYPE_CHECKING:
    from PIL import ImageFont

    from vrtf.config import QualityEvaluationConfig

logger = logging.getLogger(__name__)

# Bands below this height are fragments (dots/accents), not real text lines
_MIN_BAND_FOR_GUARD = 8

from vrtf.utils.protocols import FontCacheProtocol as FontCache  # noqa: E402
from vrtf.utils.protocols import MetricsProtocol as Metrics  # noqa: E402


class TextRenderer:
    """Render OCR text into binarized images for VRTF scoring."""

    def __init__(
        self,
        config: QualityEvaluationConfig,
        fonts: FontCache,
        metrics: Metrics,
    ) -> None:
        self.config = config
        self.fonts = fonts
        self.metrics = metrics

    # -- Consistent-typography (honest-mode) rendering -----------------------

    def render_text_consistent(
        self,
        text: str,
        w_px: int,
        h_px: int,
        *,
        font_px: int,
        pitch_px: float,
        line_h: int,
        x_left: int = 0,
    ) -> tuple[np.ndarray, int, int]:
        """Render text at fixed typography from the bbox top-left (source-free).

        The honest reconstruction: a single corpus-consistent ``font_px`` and
        monospace ``pitch_px`` (both from the per-book TypographyProfile), lines
        flowed top-down at ``line_h`` from the bbox top-left. **No source ink is
        consulted** -- no layout analysis, no per-line pitch, no xcorr. Ink that
        overflows ``h_px`` is clipped (precision loss); under-fill stays blank
        (recall loss). ``\\n`` is treated as a paragraph break, then each
        paragraph is word-wrapped (NOT visual-line wrapped -- an explicit
        ``\\\\``-line dead end).

        Returns ``(image, n_lines_drawn, n_lines_total)`` -- white bg, black ink.
        The caller uses ``n_lines_total > n_lines_drawn`` to flag clipped blocks.
        """
        text = simplify_latex(text)
        output = np.full((h_px, w_px), 255, dtype=np.uint8)
        if not text.strip():
            return output, 0, 0

        chars_per_line = max(1, int(w_px / max(1.0, pitch_px)))
        # paragraph breaks first, then word-wrap each paragraph
        lines: list[str] = []
        for para in text.replace("\r", "").split("\n"):
            para = para.strip()
            if not para:
                continue
            lines.extend(textwrap.fill(para, width=chars_per_line).split("\n"))
        if not lines:
            return output, 0, 0

        n_total = len(lines)
        bitmap_renderer = self.fonts.get_bitmap_renderer()
        font = self.fonts.get_font(font_px)
        char_w = max(1, int(round(pitch_px)))

        n_drawn = 0
        for i, line_text in enumerate(lines):
            y_top = i * line_h
            if y_top >= h_px:
                break  # remaining lines overflow -> clipped
            stripped = line_text.strip()
            if not stripped:
                n_drawn += 1
                continue
            glyph_h = min(font_px, h_px - y_top)  # clip last partial line
            if glyph_h < 2:
                break
            pil_line = Image.new("L", (w_px, glyph_h), 255)
            pil_draw = ImageDraw.Draw(pil_line)
            line_arr = np.full((glyph_h, w_px), 255, dtype=np.uint8)
            has_pil = False
            for ci, ch in enumerate(stripped):
                cx = int(x_left + ci * pitch_px)
                if cx >= w_px:
                    break
                mapped = _UNICODE_ALIASES.get(ch, ch)
                if bitmap_renderer and mapped in bitmap_renderer.templates:
                    bm = bitmap_renderer.render_line(mapped, font_px, char_w=char_w)
                    bh, bw = bm.shape
                    y_end = min(bh, glyph_h)
                    x_end = min(cx + bw, w_px)
                    if x_end > cx and y_end > 0:
                        line_arr[0:y_end, cx:x_end] = np.minimum(
                            line_arr[0:y_end, cx:x_end], bm[0:y_end, 0:x_end - cx],
                        )
                else:
                    if bitmap_renderer and mapped not in bitmap_renderer.templates:
                        if mapped not in self.fonts._missing_templates:
                            self.fonts._missing_templates.add(mapped)
                            logger.info(
                                "Missing bitmap template for %r (U+%04X)", ch, ord(ch),
                            )
                    has_pil = True
                    pil_draw.text((cx, 0), ch, fill=0, font=font)
            if has_pil:
                line_arr = np.minimum(line_arr, np.array(pil_line))
            output[y_top:y_top + glyph_h, 0:w_px] = np.minimum(
                output[y_top:y_top + glyph_h, 0:w_px], line_arr,
            )
            n_drawn += 1

        return output, n_drawn, n_total

    # -- Main text rendering --------------------------------------------------

    def render_text_in_bbox(
        self,
        text: str,
        w_px: int,
        h_px: int,
        orig_crop: np.ndarray | None = None,
        layout: TextLayout | None = None,
        orig_bin: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render text into an image of size (h_px, w_px).

        Per-line rendering pipeline:
        1. Detect text-line regions in original via horizontal projection.
        2. Choose font size from median band height.
        3. Wrap text to match detected line count.
        4. Render each line (bitmap templates if available, else PIL font),
           apply ink-spread dilation, then cross-correlate against the
           corresponding original line region to find the best (dx, dy)
           alignment within +-30 px.
        5. Composite aligned lines into the output image.

        Returns grayscale numpy array (white bg, black text).
        """
        text = simplify_latex(text)
        if not text.strip():
            return np.full((h_px, w_px), 255, dtype=np.uint8)

        # Analyse original crop for per-line layout info (reuse if passed)
        if layout is None:
            if orig_crop is not None:
                layout = self.analyse_text_layout(orig_crop)
            else:
                layout = TextLayout(
                    line_height=28, band_height=28, line_tops=[0],
                    x_left=4, text_y0=0, text_y1=h_px, x_right=w_px,
                )

        # Character width for text wrapping
        n_lines = len(layout.line_tops)
        text_width = max(1, layout.x_right - layout.x_left)

        # Font size from band height (calibrated for NimbusMonoPS:
        # typewriter ink spread ~1.30x, font ratio ~1.22x -> 0.94 combined)
        font_size = max(6, int(layout.band_height * self.config.font_size_multiplier))
        font = self.fonts.get_font(font_size)
        bb = font.getbbox("e")
        char_w = max(1, bb[2] - bb[0])

        chars_per_line = max(1, int(text_width / char_w))

        # Flatten newlines from \\ for rendering -- word-wrap distributes text
        # to match detected image bands better than explicit line breaks
        # (which rarely match band count; see analysis: only 7/106 exact match).
        render_text = text.replace('\n', ' ') if '\n' in text else text

        # Wrap text to match detected line count
        if n_lines > 1:
            wrapped_lines = self._wrap_to_n_lines(render_text, n_lines, chars_per_line)
        else:
            wrapped_lines = [render_text]

        # F1 guard: if per-char pitch would be sub-pixel (<2px), layout
        # under-detected lines (merged bands from inline math).  Only
        # re-wrap when the bbox is tall enough for the new lines (>=15px each).
        max_line_len = max((len(l) for l in wrapped_lines), default=0)
        estimated_pitch = text_width / max_line_len if max_line_len > 0 else 999
        rewrap_n = -(-len(render_text) // max(1, chars_per_line))  # ceil div
        if (estimated_pitch < 2.0
                and len(render_text) > chars_per_line
                and rewrap_n * 15 <= h_px):
            wrapped_lines = textwrap.fill(
                render_text, width=chars_per_line,
            ).split("\n")

        # Binarize original for per-line alignment (prefer pre-binarized)
        if orig_bin is None and orig_crop is not None:
            orig_gray = (
                cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY)
                if orig_crop.ndim == 3 else orig_crop
            )
            _, orig_bin = cv2.threshold(
                orig_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

        wrapped_lines, line_regions, n_wrapped, n_lines = self._build_line_regions(
            wrapped_lines, n_lines, layout, h_px,
        )

        # Render each line, align to original, composite
        output = np.full((h_px, w_px), 255, dtype=np.uint8)

        for i, line_text in enumerate(wrapped_lines):
            if not line_text.strip():
                continue
            if i >= len(line_regions):
                break

            ltop, lbot = line_regions[i]
            lh = lbot - ltop
            if lh < 4:
                continue

            # Reset font to block-level size for each line (pitch-aware
            # sizing or fallback clamping may shrink it per-line)
            line_font_size = font_size
            font = self.fonts.get_font(line_font_size)

            # Measure typewriter pitch from original ink pattern
            orig_pitch = None
            orig_x_start = layout.x_left
            if orig_bin is not None:
                orig_line_region = orig_bin[ltop:lbot, :]
                ink_cols = np.where(
                    np.any(orig_line_region == 0, axis=0)
                )[0]
                if len(ink_cols) > 5 and len(line_text.strip()) > 1:
                    ink_start = int(ink_cols[0])
                    ink_end = int(ink_cols[-1])
                    ink_width = ink_end - ink_start + 1
                    orig_pitch = ink_width / len(line_text.strip())
                    orig_x_start = ink_start

            # Pitch-aware font sizing: shrink font if advance >> detected pitch
            # Gate 1: only headings (few lines); Gate 2: only high ratio
            if (self.config.pitch_aware_font
                    and orig_pitch and orig_pitch > 4
                    and n_lines <= self.config.pitch_max_lines):
                font_advance = self.fonts.get_font_advance(font)
                ratio = font_advance / orig_pitch
                if ratio > self.config.pitch_min_ratio:
                    min_sz = max(6, int(layout.band_height * 0.5))
                    new_size = self.fonts.find_font_size_for_pitch(
                        orig_pitch, font_size, min_sz,
                    )
                    if new_size != font_size:
                        font = self.fonts.get_font(new_size)

            # Render each character at calculated positions
            bitmap_renderer = self.fonts.get_bitmap_renderer()
            line_img = Image.new("L", (w_px, lh), 255)
            line_draw = ImageDraw.Draw(line_img)

            # Vertical centering for single-line blocks only --
            # multi-line centering shifts text from band tops, breaking xcorr
            y_offset = 0
            if n_wrapped <= 1 and lh > font_size * 1.2:
                ascent, descent = font.getmetrics()
                text_height = ascent + descent
                if lh > text_height + 2:
                    y_offset = max(0, (lh - text_height) // 2)

            if orig_pitch and orig_pitch > 4:
                line_img = self._render_line_at_pitch(
                    line_img, line_draw, line_text,
                    w_px=w_px, lh=lh,
                    orig_pitch=orig_pitch, orig_x_start=orig_x_start,
                    y_offset=y_offset, font=font,
                    bitmap_renderer=bitmap_renderer,
                )
            else:
                self._render_line_full_width(
                    line_draw, line_text,
                    layout=layout, y_offset=y_offset, font=font,
                    font_size=font_size, chars_per_line=chars_per_line,
                    text_width=text_width,
                )

            line_arr = np.array(line_img)

            # Binarize rendered line
            _, line_bin = cv2.threshold(
                line_arr, 128, 255, cv2.THRESH_BINARY,
            )

            # Optional rendering dilation (old behavior: kernel=5)
            if self.config.render_dilation_kernel > 0:
                dil_k = np.ones(
                    (self.config.render_dilation_kernel,) * 2, np.uint8,
                )
                ink = (line_bin == 0).astype(np.uint8)
                dilated = cv2.dilate(ink, dil_k, iterations=1)
                line_bin = np.where(dilated > 0, 0, 255).astype(np.uint8)

            # Align to original via cross-correlation (if original available)
            if orig_bin is not None and lh >= 8:
                orig_line = orig_bin[ltop:lbot, :]
                dy, dx = xcorr_shift(orig_line, line_bin, max_shift=min(50, lh // 2))
                line_bin = shift_image(line_bin, dy, dx)

            # Composite into output
            output[ltop:lbot, :] = line_bin

        return output

    @staticmethod
    def _build_line_regions(
        wrapped_lines: list[str],
        n_lines: int,
        layout: TextLayout,
        h_px: int,
    ) -> tuple[list[str], list[tuple[int, int]], int, int]:
        """Compute (top, bot) y-regions for each wrapped line.

        Uses detected ``layout.line_tops`` when the number of wrapped lines
        fits, otherwise synthesizes evenly spaced regions. If any region is
        too thin to render legibly (<8 px), truncates the wrapped-line list
        and re-spaces.

        Returns (wrapped_lines, line_regions, n_wrapped, n_lines) -- the
        first two may be truncated from the input.
        """
        n_wrapped = len(wrapped_lines)
        if n_wrapped > n_lines:
            y0 = layout.text_y0
            y1 = layout.text_y1
            spacing = (y1 - y0) / n_wrapped
            line_regions = [
                (max(0, int(y0 + i * spacing)),
                 min(h_px, int(y0 + (i + 1) * spacing)))
                for i in range(n_wrapped)
            ]
        else:
            line_regions = []
            for i in range(n_lines):
                top = layout.line_tops[i]
                bot = layout.line_tops[i + 1] if i + 1 < n_lines else layout.text_y1
                line_regions.append((max(0, top), min(h_px, bot)))

        MIN_RENDER_LH = 8
        if any((lbot - ltop) < MIN_RENDER_LH for ltop, lbot in line_regions):
            y0, y1 = layout.text_y0, layout.text_y1
            max_lines = max(1, (y1 - y0) // MIN_RENDER_LH)
            n_wrapped = min(len(wrapped_lines), max_lines)
            wrapped_lines = wrapped_lines[:n_wrapped]
            n_lines = n_wrapped
            spacing = (y1 - y0) / max(1, n_wrapped)
            line_regions = [
                (max(0, int(y0 + i * spacing)),
                 min(h_px, int(y0 + (i + 1) * spacing)))
                for i in range(n_wrapped)
            ]

        return wrapped_lines, line_regions, n_wrapped, n_lines

    def _render_line_at_pitch(
        self,
        line_img: Image.Image,
        line_draw: ImageDraw.ImageDraw,
        line_text: str,
        *,
        w_px: int,
        lh: int,
        orig_pitch: float,
        orig_x_start: int,
        y_offset: int,
        font: "ImageFont.FreeTypeFont",
        bitmap_renderer,
    ) -> Image.Image:
        """Per-character placement at detected typewriter pitch.

        Uses the bitmap-template renderer when a template exists for the
        character, otherwise falls back to PIL font rendering. Missing
        templates are logged once per character. Returns the updated
        PIL image (a fresh Image built from the composited numpy array).
        """
        stripped = line_text.strip()
        line_arr = np.array(line_img)
        has_pil_chars = False
        for ci, ch in enumerate(stripped):
            cx = int(orig_x_start + ci * orig_pitch)
            mapped_ch = _UNICODE_ALIASES.get(ch, ch)
            if bitmap_renderer and mapped_ch in bitmap_renderer.templates:
                char_bm = bitmap_renderer.render_line(
                    mapped_ch, lh, char_w=max(1, int(orig_pitch)),
                )
                ch_h, ch_w = char_bm.shape
                x_end = min(cx + ch_w, w_px)
                y_end = min(ch_h, lh)
                if cx < w_px and x_end > cx:
                    bm_crop = char_bm[0:y_end, 0:x_end - cx]
                    line_arr[0:y_end, cx:x_end] = np.minimum(
                        line_arr[0:y_end, cx:x_end], bm_crop,
                    )
            else:
                if bitmap_renderer and mapped_ch not in bitmap_renderer.templates:
                    if mapped_ch not in self.fonts._missing_templates:
                        self.fonts._missing_templates.add(mapped_ch)
                        logger.info(
                            "Missing bitmap template for %r (U+%04X)",
                            ch, ord(ch),
                        )
                has_pil_chars = True
                line_draw.text((cx, y_offset), ch, fill=0, font=font)

        if has_pil_chars:
            pil_arr = np.array(line_img)
            line_arr = np.minimum(line_arr, pil_arr)
        return Image.fromarray(line_arr)

    def _render_line_full_width(
        self,
        line_draw: ImageDraw.ImageDraw,
        line_text: str,
        *,
        layout: TextLayout,
        y_offset: int,
        font: "ImageFont.FreeTypeFont",
        font_size: int,
        chars_per_line: int,
        text_width: int,
    ) -> None:
        """Render the whole line in one PIL ``text()`` call (no-pitch path).

        Only clamp font size for short text (headings); long text is wrapped
        and should overflow rather than crush the font to its minimum.
        Truncates excessively long lines to prevent a PIL image-mask bomb
        (1208 chars at 14 px → 256M px mask).
        """
        display_text = line_text.strip()
        if len(display_text) > chars_per_line * 2:
            display_text = display_text[:chars_per_line * 2]
        rendered_width = font.getlength(display_text)
        if (rendered_width > text_width * 1.1
                and len(display_text) <= chars_per_line * 1.5):
            scale = text_width / rendered_width
            adjusted = max(6, int(font_size * scale * 0.95))
            font = self.fonts.get_font(adjusted)
        line_draw.text((layout.x_left, y_offset), display_text, fill=0, font=font)

    # -- Layout analysis ------------------------------------------------------

    def analyse_text_layout(
        self, crop: np.ndarray,
    ) -> TextLayout:
        """Analyse original crop to extract per-line positions.

        Returns a TextLayout with:
        - line_height: median spacing between consecutive line centres
        - line_tops: y-coordinate of the top of each text line
        - x_left: left margin (first ink column in text region)
        - text_y0, text_y1: vertical extent of main text cluster
        """
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop

        h, w = gray.shape
        default = TextLayout(
            line_height=max(h, 8), band_height=max(h, 8),
            line_tops=[0], x_left=4,
            text_y0=0, text_y1=h, x_right=w,
        )
        if h < 4:
            return default

        # Binarize (with background normalization for yellowed pages)
        binary = self.metrics.binarize(gray, normalize_bg=True)

        # Horizontal projection
        ink_per_row = np.count_nonzero(binary == 0, axis=1).astype(float)
        if ink_per_row.max() == 0:
            return default

        # Smooth
        k = max(3, h // 20) | 1
        proj = cv2.GaussianBlur(
            ink_per_row.reshape(-1, 1), (1, k), 0,
        ).flatten()
        proj /= proj.max()

        # Detect text bands (contiguous rows with ink above threshold)
        is_text = proj > 0.10
        transitions = np.diff(is_text.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0] + 1

        if is_text[0]:
            starts = np.concatenate(([0], starts))
        if is_text[-1]:
            ends = np.concatenate((ends, [h]))

        n_bands = min(len(starts), len(ends))
        if n_bands == 0:
            return default

        bands = [(int(starts[i]), int(ends[i])) for i in range(n_bands)]

        # --- Merge close bands: small gaps are intra-character splits ---
        # Typewriter ascenders/descenders and dots can cause the horizontal
        # projection to dip below threshold, splitting one text line into
        # multiple thin bands.  Merge any pair of bands separated by <=5 px.
        # Three-way branching with height-aware guard:
        merge_gap = 5
        merged: list[tuple[int, int]] = [bands[0]]
        for s, e in bands[1:]:
            prev_s, prev_e = merged[-1]
            gap = s - prev_e
            cur_h = e - s
            prev_h = prev_e - prev_s
            smaller = min(cur_h, prev_h)
            larger = max(cur_h, prev_h)

            if gap > merge_gap:
                # (a) Far apart -- always keep separate
                merged.append((s, e))
            elif smaller >= _MIN_BAND_FOR_GUARD and larger > smaller * 2:
                # (b) Close but very different heights -- likely text line
                #     adjacent to diagram/equation, not intra-character split
                merged.append((s, e))
            else:
                # (c) Close and similar height, or one is a tiny fragment -- merge
                merged[-1] = (prev_s, e)
        bands = merged

        # --- Filter out oversized bands (figures / diagrams / tables) ---
        # A text line band is typically 15-60px at 2x upscale.
        # Bands >2.5x the median height of smaller bands are likely non-text.
        has_non_text_bands = False
        band_heights = [e - s for s, e in bands]
        if len(band_heights) >= 3:
            sorted_h = sorted(band_heights)
            # Median of smaller half (excludes outliers)
            ref_h = sorted_h[len(sorted_h) // 2]
            max_band_h = max(60, ref_h * 2.5)
            pre_filter_bands = bands[:]
            bands = [(s, e) for s, e in bands if (e - s) <= max_band_h]
            if not bands:
                bands = pre_filter_bands
            else:
                has_non_text_bands = len(pre_filter_bands) > len(bands)

        band_heights = [e - s for s, e in bands]
        median_band_h = max(8, int(np.median(band_heights)))
        centres = [(s + e) / 2 for s, e in bands]

        # --- Line height: median spacing between consecutive centres ---
        if len(centres) >= 2:
            spacings = [centres[i + 1] - centres[i]
                        for i in range(len(centres) - 1)]
            line_height = max(8, int(np.median(spacings)))
        else:
            line_height = max(8, median_band_h)

        # --- Robust band_h fallback ---
        # The horizontal projection can fragment text lines into thin bands
        # (dots, accents, horizontal strokes).  At 2x upscale, body text
        # characters are ~25-35 px tall and headings ~40-55 px.  If the
        # detected median band_h is too small relative to line spacing
        # or crop height, use a heuristic fallback.
        n_bands_detected = len(bands)
        if n_bands_detected >= 2 and line_height > median_band_h * 2:
            # Multi-line: character height ~ 60-70% of line spacing
            median_band_h = max(median_band_h, int(line_height * 0.65))
        elif n_bands_detected == 1 and median_band_h < 15:
            # Single band too thin -- estimate from ink row extent
            ink_rows = np.where(ink_per_row > 0)[0]
            if len(ink_rows) > 5:
                ink_extent = int(ink_rows[-1]) - int(ink_rows[0])
                median_band_h = max(median_band_h, int(ink_extent * 0.8))

        # --- Use all text-sized bands (after oversized filter) ---
        # Previous code picked only the largest cluster of closely-spaced
        # bands, but this discards text lines interspersed with equations
        # in mixed text+formula blocks.  The oversized filter already
        # removed equation/figure bands, so what remains is all text.
        text_y0 = max(0, bands[0][0])
        text_y1 = min(h, bands[-1][1])

        # Per-line top positions
        line_tops = [b[0] for b in bands]

        # --- Left/right margin: ink column extent within text region ---
        text_region = binary[text_y0:text_y1, :]
        ink_col = np.count_nonzero(text_region == 0, axis=0)
        col_threshold = max(1, (text_y1 - text_y0) * 0.02)
        text_cols = np.where(ink_col > col_threshold)[0]
        x_left = int(text_cols[0]) if len(text_cols) > 0 else 4
        x_right = int(text_cols[-1]) if len(text_cols) > 0 else w

        return TextLayout(
            line_height=line_height,
            band_height=median_band_h,
            line_tops=line_tops,
            x_left=x_left,
            text_y0=text_y0,
            text_y1=text_y1,
            x_right=x_right,
            bands=tuple(bands),
            has_non_text_bands=has_non_text_bands,
        )

    # -- Text wrapping --------------------------------------------------------

    @staticmethod
    def _wrap_to_n_lines(text: str, n_lines: int, chars_per_line: int) -> list[str]:
        """Wrap text aiming for n_lines output lines."""
        wrapped = textwrap.fill(text, width=chars_per_line)
        wrapped_lines = wrapped.split("\n")

        if len(wrapped_lines) == n_lines:
            return wrapped_lines

        # Binary search for wrap width that gives the right line count
        lo = max(1, len(text) // (n_lines + 2))
        hi = len(text)
        for _ in range(20):
            mid = (lo + hi) // 2
            trial = textwrap.fill(text, width=mid)
            trial_n = len(trial.split("\n"))
            if trial_n > n_lines:
                lo = mid + 1
            elif trial_n < n_lines:
                hi = mid - 1
            else:
                return trial.split("\n")

        return wrapped_lines

    # -- Diagram detection ----------------------------------------------------

    def is_diagram_block(
        self, orig_crop: np.ndarray, text: str,
    ) -> float:
        """Detect whether a block contains a technical diagram.

        Returns confidence score 0-1. Score > 0.6 suggests diagram content.

        Four heuristics combined with weights:
        1. Horizontal projection irregularity (0.3)
        2. Large connected components (0.3)
        3. Low text-per-ink ratio (0.2)
        4. Long line detection via HoughLinesP (0.2)
        """
        if orig_crop.ndim == 3:
            gray = cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = orig_crop.copy()

        h, w = gray.shape
        if h < 20 or w < 20:
            return 0.0

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        ink_count = np.count_nonzero(binary == 0)
        if ink_count == 0:
            return 0.0

        # 1. Horizontal projection irregularity
        ink_per_row = np.count_nonzero(binary == 0, axis=1).astype(float)
        if ink_per_row.max() > 0:
            ink_per_row_norm = ink_per_row / ink_per_row.max()
            # FFT of projection -- text has strong periodicity
            fft_vals = np.abs(np.fft.rfft(ink_per_row_norm))
            if len(fft_vals) > 2:
                peak_power = fft_vals[1:].max()
                mean_power = fft_vals[1:].mean()
                irregularity = 1.0 - (peak_power / max(mean_power * 3, 0.001))
                irregularity = max(0.0, min(1.0, irregularity))
            else:
                irregularity = 0.5
        else:
            irregularity = 0.5

        # 2. Large connected components
        inv = cv2.bitwise_not(binary)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv)
        if n_labels > 1:
            # Ignore background (label 0)
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = areas.max()
            total_area = h * w
            large_cc_score = min(1.0, max_area / (total_area * 0.1))
        else:
            large_cc_score = 0.0

        # 3. Text coverage -- diagrams have low text per ink pixel
        text_len = len(text.strip())
        if ink_count > 0 and text_len > 0:
            # Expected: ~1 char per ~100 ink pixels for text
            chars_per_ink = text_len / ink_count
            # Low ratio -> diagram
            text_coverage = min(1.0, max(0.0, 1.0 - chars_per_ink * 200))
        else:
            text_coverage = 0.5

        # 4. Line detection via HoughLinesP (on downsampled image)
        max_dim = 512
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            small = cv2.resize(binary, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small = binary

        edges = cv2.Canny(small, 50, 150)
        min_line_len = int(min(small.shape) * 0.2)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30,
            minLineLength=min_line_len, maxLineGap=10,
        )
        if lines is not None:
            n_long = len([l for l in lines if np.sqrt(
                (l[0][2] - l[0][0])**2 + (l[0][3] - l[0][1])**2
            ) > min_line_len])
            line_score = min(1.0, n_long / 5.0)
        else:
            line_score = 0.0

        # Weighted combination
        confidence = (
            0.3 * irregularity
            + 0.3 * large_cc_score
            + 0.2 * text_coverage
            + 0.2 * line_score
        )
        return confidence
