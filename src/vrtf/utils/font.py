"""Font loading, caching, and bitmap/hybrid renderer management.

Extracted from QualityEvaluationService to decouple font concerns from
the evaluation orchestration layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import ImageFont

from vrtf.config import QualityEvaluationConfig

logger = logging.getLogger(__name__)


class FontCache:
    """Manages TrueType font loading, bitmap renderer, and hybrid renderer.

    Parameters
    ----------
    config : QualityEvaluationConfig
        Supplies ``font_path``, ``use_bitmap_renderer``, and
        ``bitmap_templates_path``.
    """

    def __init__(self, config: QualityEvaluationConfig) -> None:
        self.config = config
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}
        self._bitmap_renderer = None if config.use_bitmap_renderer else False
        self._hybrid_renderer = None
        self._missing_templates: set[str] = set()
        self._warned_fallback = False

    # -- Font access --------------------------------------------------------

    def get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a cached font at the given size."""
        if size not in self._font_cache:
            try:
                self._font_cache[size] = ImageFont.truetype(
                    self.config.font_path, size,
                )
            except (OSError, IOError):
                if not self._warned_fallback:
                    logger.warning(
                        "Font not found at %s — falling back to PIL's default "
                        "font. Rendered text will not match the calibrated "
                        "size and scores will be unreliable; set "
                        "QualityEvaluationConfig.font_path to a monospace "
                        "font on this system.", self.config.font_path)
                    self._warned_fallback = True
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    @staticmethod
    def get_font_advance(font: ImageFont.FreeTypeFont) -> float:
        """Return the advance width of 'M' for the given font."""
        return font.getlength("M")

    def find_font_size_for_pitch(
        self, target_pitch: float, initial_size: int, min_size: int,
    ) -> int:
        """Binary-search for the largest font size whose advance <= *target_pitch*."""
        lo, hi = min_size, initial_size
        best = min_size
        for _ in range(10):
            if lo > hi:
                break
            mid = (lo + hi) // 2
            f = self.get_font(mid)
            adv = self.get_font_advance(f)
            if adv <= target_pitch:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    # -- Bitmap / hybrid renderers ------------------------------------------

    def get_bitmap_renderer(self):
        """Lazily load BitmapFontRenderer from templates.npz."""
        if self._bitmap_renderer is None:
            if self.config.bitmap_templates_path:
                templates_path = Path(self.config.bitmap_templates_path)
            else:
                templates_path = Path(__file__).parent.parent.parent / "font_data" / "templates.npz"
            if templates_path.exists():
                from build_typewriter_font import BitmapFontRenderer
                self._bitmap_renderer = BitmapFontRenderer(templates_path)
                logger.info(
                    "Loaded bitmap font renderer: %d templates",
                    len(self._bitmap_renderer.templates),
                )
            else:
                self._bitmap_renderer = False  # sentinel: file not found
        return self._bitmap_renderer if self._bitmap_renderer is not False else None

    def get_hybrid_renderer(self):
        """Lazy-load BitmapFontRenderer for hybrid stamping.

        Shares the instance with get_bitmap_renderer() when both are enabled.
        """
        if self._hybrid_renderer is None:
            # Reuse bitmap renderer instance if already loaded
            existing = self.get_bitmap_renderer()
            if existing is not None:
                self._hybrid_renderer = existing
            else:
                templates_path = (
                    Path(self.config.bitmap_templates_path)
                    if self.config.bitmap_templates_path
                    else Path(__file__).parent.parent.parent / "font_data" / "templates.npz"
                )
                if templates_path.exists():
                    from build_typewriter_font import BitmapFontRenderer
                    self._hybrid_renderer = BitmapFontRenderer(templates_path)
                else:
                    self._hybrid_renderer = False  # sentinel: not found
        return self._hybrid_renderer if self._hybrid_renderer is not False else None

    # -- Per-book state management ------------------------------------------

    @property
    def missing_templates(self) -> set[str]:
        """Characters encountered during rendering that lack bitmap templates."""
        return self._missing_templates.copy()

    def reset(self) -> None:
        """Reset per-book state (called at start of evaluate_book)."""
        self._missing_templates = set()
