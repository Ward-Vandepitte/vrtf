"""Figure rendering: line graphs, technical drawings, and photo scoring.

Extracted from QualityEvaluationService to isolate figure-type block
rendering into a dedicated renderer module.
"""

from __future__ import annotations

import logging
import os
import re

import cv2
import numpy as np
from scipy.signal import fftconvolve

from vrtf.config import QualityEvaluationConfig
from vrtf.metric import Metrics, xcorr_shift, shift_image

logger = logging.getLogger(__name__)

# LLM judge model for score_photo_description. Named so the model used is a single
# recorded value; pinned to Sonnet 4.6 (a cheap, temperature-accepting model for this
# 0-10 rating call). Bump deliberately if the rating quality/behaviour needs to change.
_PHOTO_JUDGE_MODEL = "claude-sonnet-4-6"


class FigureRenderer:
    """Render figure blocks (line graphs, SVG drawings) and score photo descriptions.

    Parameters
    ----------
    config : QualityEvaluationConfig
        Evaluation configuration.
    metrics : Metrics
        Image-level metric helpers.
    """

    def __init__(
        self,
        config: QualityEvaluationConfig,
        metrics: Metrics,
    ) -> None:
        self.config = config
        self.metrics = metrics

    # ------------------------------------------------------------------
    # Line graph rendering
    # ------------------------------------------------------------------

    def render_line_graph(
        self,
        extraction: dict,
        w_px: int,
        h_px: int,
        orig_crop: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Render a line graph from structured extraction data.

        Args:
            extraction: Dict with x_axis, y_axis, grid, series keys.
            w_px: Target width in pixels.
            h_px: Target height in pixels.
            orig_crop: Original image crop (RGB) for plot area detection.

        Returns:
            Grayscale uint8 array (h_px, w_px) or None on failure.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import AutoLocator

            x_axis = extraction.get("x_axis", {})
            y_axis = extraction.get("y_axis", {})
            series_list = extraction.get("series", [])
            if not series_list:
                return None

            dpi = 150
            fig = plt.figure(
                figsize=(w_px / dpi, h_px / dpi), dpi=dpi,
            )

            # Detect plot area and grid positions from original
            plot_rect = None  # [left, bottom, width, height] in figure coords
            orig_grid_pos = None  # (h_positions, v_positions)
            if orig_crop is not None:
                plot_rect = self._detect_plot_area(
                    orig_crop, w_px, h_px)
                orig_grid_pos = self._detect_grid_positions(orig_crop)
            if plot_rect is not None:
                ax = fig.add_axes(plot_rect)
            else:
                ax = fig.add_subplot(111)
                fig.tight_layout(pad=0.3)

            # Axis ranges
            x_min, x_max = x_axis.get("min"), x_axis.get("max")
            y_min, y_max = y_axis.get("min"), y_axis.get("max")

            # Scale type
            x_scale = x_axis.get("scale", "linear")
            y_scale = y_axis.get("scale", "linear")
            if x_scale == "log":
                ax.set_xscale("log")
            if y_scale == "log":
                ax.set_yscale("log")

            # Scale line width with image size
            base_lw = max(2.5, min(5.0, w_px / 300))

            # Plot each series
            for series in series_list:
                self._plot_series(ax, series, base_lw)

            # Axis ranges (set after plotting so autoscale doesn't override)
            if x_min is not None and x_max is not None:
                ax.set_xlim(x_min, x_max)
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)

            # Explicit tick positions if provided
            x_ticks = extraction.get("x_ticks")
            if x_ticks:
                ax.set_xticks(x_ticks)
            y_ticks = extraction.get("y_ticks")
            if y_ticks:
                ax.set_yticks(y_ticks)

            # Grid: use detected original grid positions if available,
            # otherwise fall back to matplotlib's grid from extracted ticks
            use_mpl_grid = extraction.get("grid", False) and orig_grid_pos is None
            if use_mpl_grid:
                ax.grid(True, which="major", color="black",
                        linewidth=max(0.8, base_lw * 0.4), alpha=0.8)
                ax.minorticks_on()
                ax.grid(True, which="minor", color="gray",
                        linewidth=max(0.5, base_lw * 0.25), alpha=0.6)

            # Thicker axis spines
            for spine in ax.spines.values():
                spine.set_linewidth(max(1.0, base_lw * 0.6))

            # Labels
            x_label = x_axis.get("label", "")
            x_unit = x_axis.get("unit", "")
            y_label = y_axis.get("label", "")
            y_unit = y_axis.get("unit", "")
            font_size = max(6, min(12, h_px / 40))
            if x_label:
                lbl = f"{x_label} [{x_unit}]" if x_unit else x_label
                ax.set_xlabel(lbl, fontsize=font_size)
            if y_label:
                lbl = f"{y_label} [{y_unit}]" if y_unit else y_label
                ax.set_ylabel(lbl, fontsize=font_size)
            ax.tick_params(labelsize=max(6, font_size - 1))

            # Curve annotations (parameter labels next to curves)
            for ann in extraction.get("annotations", []):
                try:
                    ann_color = ann.get("color", "black")
                    try:
                        import matplotlib.colors as mcolors
                        mcolors.to_rgba(ann_color)
                    except (ValueError, KeyError):
                        ann_color = "black"
                    ax.annotate(
                        str(ann.get("text", "")),
                        (ann["x"], ann["y"]),
                        fontsize=max(5, font_size - 2),
                        color=ann_color,
                    )
                except (KeyError, TypeError):
                    pass

            title = extraction.get("title")
            if title:
                ax.set_title(title, fontsize=font_size)

            if plot_rect is None:
                fig.tight_layout(pad=0.3)

            gray = self._rasterize_figure_to_gray(fig, w_px, h_px)

            # Overlay grid lines from original image at exact pixel positions
            if orig_grid_pos is not None:
                h_pos, v_pos = orig_grid_pos
                grid_lw = max(2, int(base_lw * 0.5))
                gray = self._overlay_grid(
                    gray, h_pos, v_pos,
                    line_val=80, thickness=grid_lw)

            return gray

        except Exception:
            logger.warning(
                "line_graph_render_failed", exc_info=True,
            )
            # _rasterize_figure_to_gray closes the figure only on the success
            # path; close here too or failed renders leak via pyplot's registry.
            try:
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass
            return None

    @staticmethod
    def _plot_series(ax, series: dict, base_lw: float) -> None:
        """Plot one data series on `ax` with color validation and PCHIP smoothing."""
        import matplotlib.colors as mcolors

        points = series.get("points", [])
        if len(points) < 2:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        style_map = {"dashed": "--", "dotted": ":", "solid": "-"}
        linestyle = style_map.get(series.get("style", "solid"), "-")

        # Validate color -- fall back to black on bad values
        color = series.get("color", "black")
        try:
            mcolors.to_rgba(color)
        except (ValueError, KeyError):
            color = "black"

        # Interpolate for smooth curves if enough points
        if len(xs) >= 4:
            try:
                from scipy.interpolate import PchipInterpolator
                sorted_pairs = sorted(zip(xs, ys))
                xs_s = [p[0] for p in sorted_pairs]
                ys_s = [p[1] for p in sorted_pairs]
                xs_u, ys_u = [xs_s[0]], [ys_s[0]]
                for i in range(1, len(xs_s)):
                    if xs_s[i] != xs_u[-1]:
                        xs_u.append(xs_s[i])
                        ys_u.append(ys_s[i])
                if len(xs_u) >= 4:
                    interp = PchipInterpolator(xs_u, ys_u)
                    xs_fine = np.linspace(xs_u[0], xs_u[-1], 200)
                    ys_fine = interp(xs_fine)
                    ax.plot(
                        xs_fine, ys_fine,
                        linestyle=linestyle, color=color,
                        linewidth=base_lw,
                    )
                    return
            except Exception:
                pass

        # Fallback: plot raw points
        ax.plot(xs, ys, linestyle=linestyle, color=color, linewidth=base_lw)

    @staticmethod
    def _rasterize_figure_to_gray(fig, w_px: int, h_px: int) -> np.ndarray:
        """Render matplotlib figure to a grayscale uint8 array at (h_px, w_px).

        Uses min-channel RGB → gray so any colored ink on white background
        produces a low value (preserves curve colors like cyan/orange for
        downstream Otsu binarization).
        """
        import matplotlib.pyplot as plt

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        gray = np.min(rgba[..., :3], axis=2).astype(np.uint8)
        plt.close(fig)

        if gray.shape[0] != h_px or gray.shape[1] != w_px:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(gray).resize(
                (w_px, h_px), PILImage.LANCZOS,
            )
            gray = np.array(pil_img)
        return gray

    # ------------------------------------------------------------------
    # Photo description scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score_photo_description(
        orig_crop: np.ndarray,
        description: str,
    ) -> float:
        """Score a photo description using an LLM judge.

        Sends the original image crop + the extracted description to Claude
        and asks for a 0-10 quality rating. Returns score as 0.0-1.0.

        Falls back to a heuristic score if the API is unavailable.
        """
        import base64
        import io

        if not description or len(description.strip()) < 5:
            return 0.0

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # Heuristic fallback: length-based
            return min(1.0, len(description.strip()) / 100)

        try:
            import anthropic
            from PIL import Image as PILImage

            # Encode crop as base64 PNG
            if len(orig_crop.shape) == 2:
                pil_img = PILImage.fromarray(orig_crop)
            else:
                pil_img = PILImage.fromarray(orig_crop)
            # Downscale for cost efficiency
            max_dim = 512
            w, h = pil_img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                pil_img = pil_img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    PILImage.LANCZOS)

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")

            client = anthropic.Anthropic(api_key=api_key, max_retries=0)
            # #1508: pin temperature=0 and record the model so this LLM-judged score is
            # reproducible (default temperature gave run-to-run variance feeding ink_overlap).
            # Sonnet 4.6 still accepts temperature (Sonnet 5 / Opus 4.7+ would 400 on it).
            logger.debug("photo_description_judge model=%s temperature=0", _PHOTO_JUDGE_MODEL)
            message = client.messages.create(
                model=_PHOTO_JUDGE_MODEL,
                max_tokens=50,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f'Rate this description of the image on a '
                                f'scale of 0-10. Consider accuracy, '
                                f'completeness, and specificity.\n\n'
                                f'Description: "{description}"\n\n'
                                f'Respond with ONLY a single number 0-10.'
                            ),
                        },
                    ],
                }],
            )

            raw = message.content[0].text.strip()
            # Extract number from response
            match = re.search(r'\b(\d+(?:\.\d+)?)\b', raw)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score / 10.0))
            return 0.5  # unparseable -> middle score

        except Exception as e:
            logger.warning(
                "photo_description_scoring_failed: %s", e)
            # Heuristic fallback
            return min(1.0, len(description.strip()) / 100)

    # ------------------------------------------------------------------
    # Technical drawing (SVG) rendering
    # ------------------------------------------------------------------

    @staticmethod
    def render_technical_drawing(
        svg_str: str,
        w_px: int,
        h_px: int,
    ) -> np.ndarray | None:
        """Render an SVG technical drawing to a grayscale image.

        Args:
            svg_str: SVG markup string.
            w_px: Target width in pixels.
            h_px: Target height in pixels.

        Returns:
            Grayscale uint8 array (h_px, w_px) or None on failure.
        """
        try:
            import cairosvg
            import re
            from PIL import Image as PILImage
            import io

            svg_bytes = svg_str.encode("utf-8")

            # Ensure viewBox is present for correct scaling
            if b"viewBox" not in svg_bytes:
                svg_bytes = re.sub(
                    rb"(<svg\b)",
                    rf'\1 viewBox="0 0 {w_px} {h_px}"'.encode(),
                    svg_bytes,
                    count=1,
                )

            png_data = cairosvg.svg2png(
                bytestring=svg_bytes,
                output_width=w_px,
                output_height=h_px,
                background_color="white",
            )

            pil_img = PILImage.open(io.BytesIO(png_data)).convert("L")
            gray = np.array(pil_img)

            # Resize if dimensions don't match exactly
            if gray.shape[0] != h_px or gray.shape[1] != w_px:
                pil_img = PILImage.fromarray(gray)
                pil_img = pil_img.resize((w_px, h_px), PILImage.LANCZOS)
                gray = np.array(pil_img)

            return gray

        except Exception:
            logger.warning(
                "technical_drawing_render_failed", exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Plot area and grid detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_plot_area(
        orig_crop: np.ndarray,
        w_px: int,
        h_px: int,
    ) -> list[float] | None:
        """Detect the plot area frame in the original image.

        Returns [left, bottom, width, height] in matplotlib figure
        coordinates (0-1 range), or None if detection fails.
        """
        import cv2

        if len(orig_crop.shape) == 3:
            gray = cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = orig_crop
        h, w = gray.shape

        _, bw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect long horizontal lines (axis frame top/bottom)
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (w // 3, 1))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
        h_proj = np.sum(h_lines, axis=1) / 255
        h_rows = np.where(h_proj > w * 0.3)[0]

        # Detect long vertical lines (axis frame left/right)
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, h // 3))
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
        v_proj = np.sum(v_lines, axis=0) / 255
        v_cols = np.where(v_proj > h * 0.3)[0]

        if len(h_rows) < 2 or len(v_cols) < 2:
            return None

        plot_top = h_rows[0]
        plot_bottom = h_rows[-1]
        plot_left = v_cols[0]
        plot_right = v_cols[-1]

        # Sanity: plot area should be >30% of image in both dimensions
        if (plot_right - plot_left) < w * 0.3:
            return None
        if (plot_bottom - plot_top) < h * 0.3:
            return None

        # Convert to matplotlib figure coordinates (origin bottom-left)
        fig_left = plot_left / w
        fig_bottom = 1.0 - plot_bottom / h
        fig_width = (plot_right - plot_left) / w
        fig_height = (plot_bottom - plot_top) / h

        return [fig_left, fig_bottom, fig_width, fig_height]

    @staticmethod
    def _detect_grid_positions(
        orig_crop: np.ndarray,
    ) -> tuple[list[int], list[int]] | None:
        """Detect interior grid line positions from the original image.

        Returns (h_positions, v_positions) -- lists of y-pixel and x-pixel
        coordinates for horizontal and vertical grid lines inside the plot
        area, or None if detection fails.
        """
        import cv2

        if len(orig_crop.shape) == 3:
            gray = cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = orig_crop
        h, w = gray.shape

        _, bw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect horizontal lines (shorter kernel than frame detection)
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (w // 5, 1))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
        h_proj = np.sum(h_lines, axis=1) / 255
        h_thresh = w * 0.2
        h_candidates = np.where(h_proj > h_thresh)[0]

        # Detect vertical lines
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, h // 5))
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
        v_proj = np.sum(v_lines, axis=0) / 255
        v_thresh = h * 0.2
        v_candidates = np.where(v_proj > v_thresh)[0]

        if len(h_candidates) == 0 and len(v_candidates) == 0:
            return None

        # Cluster nearby pixels into single line positions
        def _cluster(positions: np.ndarray, min_gap: int = 5) -> list[int]:
            if len(positions) == 0:
                return []
            clusters: list[list[int]] = [[int(positions[0])]]
            for p in positions[1:]:
                if p - clusters[-1][-1] <= min_gap:
                    clusters[-1].append(int(p))
                else:
                    clusters.append([int(p)])
            return [int(np.mean(c)) for c in clusters]

        h_pos = _cluster(h_candidates)
        v_pos = _cluster(v_candidates)

        return h_pos, v_pos

    @staticmethod
    def _overlay_grid(
        rendered: np.ndarray,
        h_positions: list[int],
        v_positions: list[int],
        line_val: int = 80,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw grid lines on rendered grayscale image at specified positions.

        Args:
            rendered: Grayscale uint8 array.
            h_positions: Y-pixel positions for horizontal lines.
            v_positions: X-pixel positions for vertical lines.
            line_val: Grayscale value for grid lines (0=black, 255=white).
            thickness: Line thickness in pixels.
        """
        import cv2

        out = rendered.copy()
        h, w = out.shape
        half = thickness // 2
        for y in h_positions:
            y0 = max(0, y - half)
            y1 = min(h, y + half + 1)
            # Only darken -- don't lighten existing curve ink
            out[y0:y1, :] = np.minimum(out[y0:y1, :], line_val)
        for x in v_positions:
            x0 = max(0, x - half)
            x1 = min(w, x + half + 1)
            out[:, x0:x1] = np.minimum(out[:, x0:x1], line_val)
        return out

    # ------------------------------------------------------------------
    # Graph alignment
    # ------------------------------------------------------------------

    @staticmethod
    def align_graph_xcorr(
        orig_bin: np.ndarray,
        rend_bin: np.ndarray,
    ) -> np.ndarray:
        """Align rendered graph to original using cross-correlation.

        Finds the global (dx, dy) shift that maximizes ink overlap,
        then returns the shifted render. This compensates for matplotlib
        layout positioning vs the original chart's axis placement.
        """
        h, w = orig_bin.shape
        orig_f = (orig_bin == 0).astype(np.float32)
        rend_f = (rend_bin == 0).astype(np.float32)

        if orig_f.sum() == 0 or rend_f.sum() == 0:
            return rend_bin

        xcorr = fftconvolve(orig_f, rend_f[::-1, ::-1], mode="same")
        cy, cx = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        dy = cy - h // 2
        dx = cx - w // 2

        # Cap shift to 15% of image dimensions
        max_dx = int(w * 0.15)
        max_dy = int(h * 0.15)
        dx = max(-max_dx, min(max_dx, dx))
        dy = max(-max_dy, min(max_dy, dy))

        if dx == 0 and dy == 0:
            return rend_bin

        shifted = np.full_like(rend_bin, 255)
        sy_s = max(0, dy)
        sy_e = min(h, h + dy)
        sx_s = max(0, dx)
        sx_e = min(w, w + dx)
        ry_s = max(0, -dy)
        ry_e = min(h, h - dy)
        rx_s = max(0, -dx)
        rx_e = min(w, w - dx)
        shifted[sy_s:sy_e, sx_s:sx_e] = rend_bin[ry_s:ry_e, rx_s:rx_e]
        return shifted
