"""Shared Protocol stubs for VRTF renderer dependencies.

The real implementations live in ``vrtf.utils.font`` (FontCache) and
``vrtf.metric`` (Metrics). Renderers use these Protocols under
``TYPE_CHECKING`` so they can be type-checked without pulling the heavy
implementations into import time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from PIL import ImageFont


class FontCacheProtocol(Protocol):
    """Font loading / sizing interface consumed by the VRTF renderers."""

    _missing_templates: set[str]

    def get_font(self, size: int) -> "ImageFont.FreeTypeFont": ...

    def get_font_advance(self, font: "ImageFont.FreeTypeFont") -> float: ...

    def find_font_size_for_pitch(
        self, target_pitch: float, max_size: int, min_size: int,
    ) -> int: ...

    def get_bitmap_renderer(self): ...


class MetricsProtocol(Protocol):
    """Image metric interface consumed by the VRTF renderers."""

    def binarize(
        self, image: np.ndarray, normalize_bg: bool = True,
    ) -> np.ndarray: ...
