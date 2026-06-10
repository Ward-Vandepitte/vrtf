"""Visual Roundtrip Fidelity (VRTF) -- reference-free OCR quality metric."""

from vrtf.config import (
    FeatureFlags,
    FontSelectionConfig,
    OverlayConfig,
    QualityEvaluationConfig,
    RendererConfig,
    ScoringConfig,
)
from vrtf.models import (
    BBOX_SCALE,
    BlockScore,
    BookEvaluation,
    PageScore,
    TextLayout,
    TypographyProfile,
)

# Backward-compat alias (renamed from _TextLayout to TextLayout)
_TextLayout = TextLayout

__all__ = [
    "QualityEvaluationConfig",
    "QualityEvaluationService",
    "calibrate_from_pages",
    "RendererConfig",
    "ScoringConfig",
    "FeatureFlags",
    "FontSelectionConfig",
    "OverlayConfig",
    "BlockScore",
    "BookEvaluation",
    "PageScore",
    "TextLayout",
    "TypographyProfile",
    "_TextLayout",
    "BBOX_SCALE",
]


def __getattr__(name: str):
    """Lazy import for QualityEvaluationService to avoid heavy deps at package level."""
    if name == "QualityEvaluationService":
        from vrtf.evaluator import QualityEvaluationService

        globals()["QualityEvaluationService"] = QualityEvaluationService
        return QualityEvaluationService
    if name == "calibrate_from_pages":
        from vrtf.calibrate import calibrate_from_pages

        globals()["calibrate_from_pages"] = calibrate_from_pages
        return calibrate_from_pages
    raise AttributeError(f"module 'vrtf' has no attribute {name!r}")
