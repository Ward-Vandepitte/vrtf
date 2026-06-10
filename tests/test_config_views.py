"""Tests for QualityEvaluationConfig typed views."""

from __future__ import annotations

import dataclasses

import pytest

from vrtf.config import (
    FeatureFlags,
    FontSelectionConfig,
    OverlayConfig,
    QualityEvaluationConfig,
    RendererConfig,
    ScoringConfig,
)
from vrtf.models import TypographyProfile

_VIEWS: list[tuple[str, type]] = [
    ("renderer", RendererConfig),
    ("scoring", ScoringConfig),
    ("features", FeatureFlags),
    ("font_selection", FontSelectionConfig),
    ("overlay", OverlayConfig),
]


def test_every_parent_field_in_exactly_one_view():
    """F3 forward coverage: no parent field is dropped or duplicated across views."""
    parent_fields = {f.name for f in dataclasses.fields(QualityEvaluationConfig)}
    view_fields: dict[str, list[str]] = {}
    for _, cls in _VIEWS:
        for f in dataclasses.fields(cls):
            view_fields.setdefault(f.name, []).append(cls.__name__)
    missing = parent_fields - view_fields.keys()
    assert not missing, f"Parent fields absent from all views: {sorted(missing)}"
    duplicates = {name: groups for name, groups in view_fields.items() if len(groups) > 1}
    assert not duplicates, f"Fields appearing in >1 view: {duplicates}"


def test_every_view_field_exists_on_parent():
    """F4 reverse coverage: no view has a field the parent doesn't have."""
    parent_fields = {f.name for f in dataclasses.fields(QualityEvaluationConfig)}
    for _, cls in _VIEWS:
        for f in dataclasses.fields(cls):
            assert f.name in parent_fields, (
                f"{cls.__name__}.{f.name} not present on QualityEvaluationConfig"
            )


@pytest.mark.parametrize(
    "field",
    [f.name for f in dataclasses.fields(QualityEvaluationConfig)],
)
def test_field_routes_to_correct_view(field: str):
    """F3 sentinel sweep: each field set to a distinct value propagates to its owning view."""
    parent_field = next(
        f for f in dataclasses.fields(QualityEvaluationConfig) if f.name == field
    )
    default = parent_field.default
    if isinstance(default, bool):
        sentinel: object = not default
    elif isinstance(default, int):
        sentinel = 424242
    elif isinstance(default, float):
        sentinel = 313.137
    elif isinstance(default, str):
        sentinel = f"__SENTINEL_{field}__"
    elif default is None:
        # e.g. typography: TypographyProfile | None = None
        sentinel = TypographyProfile(body_h_norm=12.0, body_leading=1.13)
    else:
        pytest.fail(f"Unhandled default type for {field}: {type(default).__name__}")

    cfg = QualityEvaluationConfig(**{field: sentinel})

    owning = [
        (attr, cls)
        for attr, cls in _VIEWS
        if field in {f.name for f in dataclasses.fields(cls)}
    ]
    assert len(owning) == 1, f"{field} should own exactly one view, got {owning}"
    view_attr, view_cls = owning[0]
    view_instance = getattr(cfg, view_attr)
    assert isinstance(view_instance, view_cls)
    assert getattr(view_instance, field) == sentinel, (
        f"{field} = {sentinel!r} on parent but "
        f"cfg.{view_attr}.{field} = {getattr(view_instance, field)!r}"
    )


def test_backward_compat_guarantees():
    """Load-bearing external APIs: ctor, attr, replace, fields() length, fresh-snapshot views."""
    # (a) flat-kwarg constructor still accepts every field
    cfg = QualityEvaluationConfig(font_path="x.otf", dt_threshold=5.0)
    # (b) flat attribute read
    assert cfg.font_path == "x.otf"
    assert cfg.dt_threshold == 5.0
    # (c) dataclasses.replace on flat kwargs still works
    cfg2 = dataclasses.replace(cfg, font_path="y.otf")
    assert cfg2.font_path == "y.otf"
    assert cfg2.renderer.font_path == "y.otf"
    # (d) __dataclass_fields__ still lists all flat fields
    #     (Openboeks config._filter_known depends on this)
    #     34 original + 11 consistent-typography fields + 1 ct_tau_body_frac
    #     + 2 avenue-run fields (ct_figure_recreation, ct_capture_confusion) = 48
    assert len(dataclasses.fields(QualityEvaluationConfig)) == 48
    # (e) views are fresh snapshots, not cached (prevents a future "helpful" cached_property)
    assert cfg.renderer is not cfg.renderer
    assert cfg.renderer == cfg.renderer
