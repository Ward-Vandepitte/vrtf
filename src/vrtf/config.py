"""Configuration for VRTF quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from vrtf.models import TypographyProfile


@dataclass(frozen=True, slots=True)
class RendererConfig:
    """Fields read by vrtf renderers (text, equation, table, figure)."""

    font_path: str = "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf"
    use_bitmap_renderer: bool = False
    bitmap_templates_path: str = ""
    render_dilation_kernel: int = 0
    font_size_multiplier: float = 1.10
    pitch_aware_font: bool = False
    pitch_max_lines: int = 3
    pitch_min_ratio: float = 1.5
    use_pdflatex_for_equations: bool = True
    pdflatex_max_block_height: int = 500
    use_pdflatex_composite: bool = True
    use_hybrid_glyph_stamping: bool = False
    formula_cleanup_path: str = ""
    # --- Consistent-typography (honest-mode) rendering toggles ---
    ct_use_profile_font: bool = False
    ct_disable_line_xcorr: bool = False
    ct_source_free_flow: bool = False
    ct_equation_pdflatex_only: bool = False
    ct_disable_table_grid_detect: bool = False
    ct_heading_levels: int = 3
    ct_leading: float = 0.0
    ct_kmeans_seed: int = 0
    typography: TypographyProfile | None = None


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Fields read by vrtf scoring (metric, overlay, compute_f1_overlap).

    ``dt_threshold`` is temporarily mutated by
    ``evaluator._composite_full_page`` via ``object.__setattr__``; do not
    cache ``cfg.scoring`` across that call — views are fresh snapshots.
    """

    binarize_method: str = "otsu"
    min_block_area_px: int = 100
    weight_ink_overlap: float = 1.0
    weight_ssim: float = 0.0
    dt_threshold: float = 20.0
    overlap_dilation_kernel: int = 5
    photo_credit_score: float = 0.5
    dt_threshold_full_page: float = 20.0
    ct_tau_body_frac: float = 0.0  # honest mode: full-page tau = frac*body_px (0=use fixed)


@dataclass(frozen=True, slots=True)
class FeatureFlags:
    """Boolean toggles and tuning scalars gating evaluator behavior."""

    use_dt_overlap: bool = True
    skip_formula_heavy: bool = False
    formula_min_text_ratio: float = 0.4
    detect_diagrams: bool = False
    mask_mixed_content: bool = True
    evaluate_tables: bool = True
    use_image_guided_tables: bool = True
    full_page_mode: bool = False
    consistent_typography_mode: bool = False
    ct_disable_band_mask: bool = False
    ct_figure_recreation: str = "off"
    ct_capture_confusion: bool = False


@dataclass(frozen=True, slots=True)
class FontSelectionConfig:
    """Exported for external font-selection services.

    No vrtf-internal consumers; provided as a typed lens so external code
    can narrow a ``QualityEvaluationConfig`` parameter to just the
    font-selection subset.
    """

    font_selection_enabled: bool = True
    font_selection_threshold: float = 0.5
    font_selection_sample_pages: int = 5


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """Fields read by vrtf overlay rendering."""

    generate_overlays: bool = True
    overlay_max_pages: int = 20


@dataclass(frozen=True)
class QualityEvaluationConfig:
    """Configuration for image-based OCR quality evaluation.

    Views (``cfg.renderer``, ``cfg.scoring``, ``cfg.features``,
    ``cfg.font_selection``, ``cfg.overlay``) are fresh snapshots built on
    each access — use ``==``, not ``is``, for comparison. Safe for
    page-level reads; avoid in pixel-level hot loops because each access
    allocates a new frozen dataclass. Views cannot be cached via
    ``functools.cached_property`` because it mutates ``__dict__``, which is
    incompatible with ``frozen=True``.
    """

    font_path: str = "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf"
    binarize_method: str = "otsu"
    min_block_area_px: int = 100
    generate_overlays: bool = True
    overlay_max_pages: int = 20
    weight_ink_overlap: float = 1.0
    weight_ssim: float = 0.0

    # --- Feature flags for changelog measurement ---
    use_dt_overlap: bool = True           # False -> old dilation-based overlap
    dt_threshold: float = 20.0            # DT tolerance in pixels
    overlap_dilation_kernel: int = 5      # kernel size when use_dt_overlap=False
    render_dilation_kernel: int = 0       # 0=disabled, 5=old behavior (dilate rendered text)
    skip_formula_heavy: bool = False      # True -> skip formula-heavy text blocks
    formula_min_text_ratio: float = 0.4   # skip if stripped/original < this
    font_size_multiplier: float = 1.10    # multiplier for band_height -> font_size
    use_bitmap_renderer: bool = False     # True -> use BitmapFontRenderer
    bitmap_templates_path: str = ""       # empty -> default font_data/templates.npz
    detect_diagrams: bool = False         # True -> detect diagram blocks
    mask_mixed_content: bool = True       # True -> mask non-text bands in orig before comparison
    pitch_aware_font: bool = False        # True -> shrink font when advance > detected pitch
    pitch_max_lines: int = 3              # Gate 1: only fire on blocks with <= this many lines
    pitch_min_ratio: float = 1.5          # Gate 2: only fire when advance/pitch exceeds this
    use_pdflatex_for_equations: bool = True   # True -> render equation blocks via pdflatex
    pdflatex_max_block_height: int = 500      # skip pdflatex for blocks taller than this (px)
    use_pdflatex_composite: bool = True       # composite pdflatex equation bands + text pipeline text bands
    formula_cleanup_path: str = ""            # path to formula_cleanup.json (empty -> skip UniMERNet fallback)
    use_hybrid_glyph_stamping: bool = False   # True -> stamp bitmap templates on pdflatex equation renders
    evaluate_tables: bool = True              # True -> render+score table blocks (pipe-delimited markdown)
    use_image_guided_tables: bool = True      # detect grid lines from original image for table layout

    # --- Font selection ---
    font_selection_enabled: bool = True       # auto-select best font before evaluation
    font_selection_threshold: float = 0.5     # minimum % improvement to switch font
    font_selection_sample_pages: int = 5      # pages to sample for font comparison

    # --- Figure scoring ---
    photo_credit_score: float = 0.5           # partial credit for detected photo blocks (0-1)

    # --- Full-page composite scoring ---
    full_page_mode: bool = False              # enable full-page composite F1 scoring
    dt_threshold_full_page: float = 20.0      # DT tolerance for full-page mode (px)
    ct_tau_body_frac: float = 0.0             # honest: full-page tau = frac*body_px (0=fixed, resolution-invariant)

    # --- Consistent-typography (honest-mode) — granular ablation toggles ---
    # Each removes one source-leakage compensator; the preset enables all.
    # Read the EFFECTIVE values via the ct_* helper properties below, which
    # OR-in the preset and the source_free_flow -> use_profile_font implication.
    consistent_typography_mode: bool = False  # preset: enable all ct_* below
    ct_use_profile_font: bool = False         # C1: size from profile, not band_height
    ct_disable_line_xcorr: bool = False       # C2: skip per-line xcorr shift
    ct_source_free_flow: bool = False         # C3: flow from bbox top-left (implies C1)
    ct_disable_band_mask: bool = False        # don't mask source with source bands
    ct_equation_pdflatex_only: bool = False   # drop equation F1 best-of-three pick
    ct_disable_table_grid_detect: bool = False  # drop source image-guided table grid
    ct_heading_levels: int = 3                # k for heading-size clustering
    ct_leading: float = 0.0                   # 0 -> use calibrated body_leading
    ct_kmeans_seed: int = 0                   # seed for heading clustering reproducibility
    # Honest-mode avenue-run toggles (analysis only; both default off so all
    # corpus numbers are unchanged). Read from the full config by the evaluator;
    # deliberately NOT plumbed into the renderer/scoring view snapshots.
    ct_figure_recreation: str = "off"         # "off"|"source_free"|"peeking": route
                                              # line_graph/technical_drawing through
                                              # recreation in honest mode
    ct_capture_confusion: bool = False        # stash (orig_bin, rend_canvas, eff_tau)
                                              # on PageScore.confusion for decomposition
    typography: TypographyProfile | None = None  # set by calibrate_typography via replace

    # --- Effective CT toggles (preset + implications expanded) ---
    @property
    def ct_flow_source_free(self) -> bool:
        return self.ct_source_free_flow or self.consistent_typography_mode

    @property
    def ct_font_from_profile(self) -> bool:
        # source-free flow implies sizing from the profile
        return (
            self.ct_use_profile_font
            or self.ct_source_free_flow
            or self.consistent_typography_mode
        )

    @property
    def ct_skip_line_xcorr(self) -> bool:
        return self.ct_disable_line_xcorr or self.consistent_typography_mode

    @property
    def ct_skip_band_mask(self) -> bool:
        return self.ct_disable_band_mask or self.consistent_typography_mode

    @property
    def ct_eq_pdflatex_only(self) -> bool:
        return self.ct_equation_pdflatex_only or self.consistent_typography_mode

    @property
    def ct_skip_table_grid(self) -> bool:
        return self.ct_disable_table_grid_detect or self.consistent_typography_mode

    @property
    def renderer(self) -> RendererConfig:
        # fresh snapshot on each access; see class docstring
        return RendererConfig(
            font_path=self.font_path,
            use_bitmap_renderer=self.use_bitmap_renderer,
            bitmap_templates_path=self.bitmap_templates_path,
            render_dilation_kernel=self.render_dilation_kernel,
            font_size_multiplier=self.font_size_multiplier,
            pitch_aware_font=self.pitch_aware_font,
            pitch_max_lines=self.pitch_max_lines,
            pitch_min_ratio=self.pitch_min_ratio,
            use_pdflatex_for_equations=self.use_pdflatex_for_equations,
            pdflatex_max_block_height=self.pdflatex_max_block_height,
            use_pdflatex_composite=self.use_pdflatex_composite,
            use_hybrid_glyph_stamping=self.use_hybrid_glyph_stamping,
            formula_cleanup_path=self.formula_cleanup_path,
            ct_use_profile_font=self.ct_use_profile_font,
            ct_disable_line_xcorr=self.ct_disable_line_xcorr,
            ct_source_free_flow=self.ct_source_free_flow,
            ct_equation_pdflatex_only=self.ct_equation_pdflatex_only,
            ct_disable_table_grid_detect=self.ct_disable_table_grid_detect,
            ct_heading_levels=self.ct_heading_levels,
            ct_leading=self.ct_leading,
            ct_kmeans_seed=self.ct_kmeans_seed,
            typography=self.typography,
        )

    @property
    def scoring(self) -> ScoringConfig:
        # fresh snapshot on each access; see class docstring
        return ScoringConfig(
            binarize_method=self.binarize_method,
            min_block_area_px=self.min_block_area_px,
            weight_ink_overlap=self.weight_ink_overlap,
            weight_ssim=self.weight_ssim,
            dt_threshold=self.dt_threshold,
            overlap_dilation_kernel=self.overlap_dilation_kernel,
            photo_credit_score=self.photo_credit_score,
            dt_threshold_full_page=self.dt_threshold_full_page,
            ct_tau_body_frac=self.ct_tau_body_frac,
        )

    @property
    def features(self) -> FeatureFlags:
        # fresh snapshot on each access; see class docstring
        return FeatureFlags(
            use_dt_overlap=self.use_dt_overlap,
            skip_formula_heavy=self.skip_formula_heavy,
            formula_min_text_ratio=self.formula_min_text_ratio,
            detect_diagrams=self.detect_diagrams,
            mask_mixed_content=self.mask_mixed_content,
            evaluate_tables=self.evaluate_tables,
            use_image_guided_tables=self.use_image_guided_tables,
            full_page_mode=self.full_page_mode,
            consistent_typography_mode=self.consistent_typography_mode,
            ct_disable_band_mask=self.ct_disable_band_mask,
            ct_figure_recreation=self.ct_figure_recreation,
            ct_capture_confusion=self.ct_capture_confusion,
        )

    @property
    def font_selection(self) -> FontSelectionConfig:
        # fresh snapshot on each access; see class docstring
        return FontSelectionConfig(
            font_selection_enabled=self.font_selection_enabled,
            font_selection_threshold=self.font_selection_threshold,
            font_selection_sample_pages=self.font_selection_sample_pages,
        )

    @property
    def overlay(self) -> OverlayConfig:
        # fresh snapshot on each access; see class docstring
        return OverlayConfig(
            generate_overlays=self.generate_overlays,
            overlay_max_pages=self.overlay_max_pages,
        )
