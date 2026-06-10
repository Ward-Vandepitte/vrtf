"""Per-book consistent-typography calibration for honest-mode VRTF.

Sibling of ``font_selection``: computes a single per-book ``TypographyProfile``
(body size, line leading, monospace pitch, per-level heading sizes) and returns
a config with it attached via ``dataclasses.replace``. The orchestrator calls
this once per book, BEFORE font selection, so both share the profile.

Design constraints (from the reviewed plan):

* **Global, not per-block.** Sizes are measured once over the whole book; the
  honest renderer then uses these fixed sizes so it can no longer resize per
  block to absorb bbox-detection error (the C1 leak).
* **Source-independent measurement.** Body size comes from the raw horizontal
  ink-projection band heights (a direct measurement of line ink extent), NOT
  from ``analyse_text_layout`` whose merge/filter/fallback heuristics were tuned
  to make the OLD per-block metric score well.
* **Same coordinate space as scoring (G0).** Blocks are cropped with the exact
  ``img_w / BBOX_SCALE`` scaling the evaluator uses, so ``body_px`` (pixels) and
  a scored block's ``h_px`` (pixels) live in one unit system. A runtime assert
  fires loudly if a book's bboxes are not in the expected ~0-1000 space.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vrtf import BBOX_SCALE, QualityEvaluationConfig, QualityEvaluationService
from vrtf.models import TypographyProfile

logger = logging.getLogger(__name__)

_BATCH_RE = re.compile(r"batch_(\d+)")

# Template aspect ratio (CELL_W / CELL_H = 36 / 64) — monospace advance / height.
_PITCH_HEIGHT_RATIO = 36.0 / 64.0

# Plausible single text-line band height, in per-mille of page height
# (resolution-independent; ~12 per-mille = a 16px line on a 1346px-tall scan).
_MIN_LINE_NORM = 3.0
_MAX_TEXT_LINE_NORM = 90.0
_MAX_HEADING_LINE_NORM = 160.0

_MIN_CROP_PX = 6               # skip crops thinner/shorter than this (pixels)
_MIN_BODY_SAMPLES = 30         # below this -> degenerate body fallback
_DEFAULT_BODY_NORM = 12.0      # fallback body height (per-mille)
_MIN_HEADINGS_FOR_CLUSTER = 10  # below this -> single heading size


def _binarize(crop_gray: np.ndarray) -> np.ndarray:
    """Otsu binarize a grayscale crop (ink=0, bg=255)."""
    if crop_gray.size == 0:
        return crop_gray
    _, bin_img = cv2.threshold(
        crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return bin_img


def _line_bands(crop_bin: np.ndarray) -> tuple[list[int], list[float]]:
    """Horizontal-projection text-line bands -> (band_heights, band_centers).

    The ink-per-row projection is Gaussian-smoothed before thresholding so that
    a single text line (whose ascenders / x-height / descenders leave small
    inter-row gaps) registers as ONE band rather than several stubs. Smoothing
    is standard line detection, not an F1-tuned heuristic -- we deliberately do
    NOT apply ``analyse_text_layout``'s downstream merge-gap / oversized-band /
    0.65-fallback logic, which was tuned to make the OLD per-block metric score.
    """
    if crop_bin.size == 0 or crop_bin.shape[0] < 2:
        return [], []
    ink_per_row = np.count_nonzero(crop_bin == 0, axis=1).astype(np.float32)
    h = crop_bin.shape[0]
    k = max(3, h // 20) | 1  # odd kernel, same scale as analyse_text_layout
    proj = cv2.GaussianBlur(ink_per_row.reshape(-1, 1), (1, k), 0).flatten()
    if proj.max() <= 0:
        return [], []
    proj /= proj.max()
    is_ink = proj > 0.10
    heights: list[int] = []
    centers: list[float] = []
    run_start: int | None = None
    for y, on in enumerate(is_ink):
        if on and run_start is None:
            run_start = y
        elif not on and run_start is not None:
            heights.append(y - run_start)
            centers.append((run_start + y) / 2.0)
            run_start = None
    if run_start is not None:
        end = len(is_ink)
        heights.append(end - run_start)
        centers.append((run_start + end) / 2.0)
    return heights, centers


def _dominant_mode(values: list[float], bin_w: float = 1.0) -> float:
    """Histogram-peak (modal) value, robust to a long tail of outliers."""
    if not values:
        return 0.0
    binned = Counter(round(v / bin_w) for v in values)
    peak_bin, _ = binned.most_common(1)[0]
    return peak_bin * bin_w


def _kmeans_1d(data: np.ndarray, k: int, seed: int, iters: int = 50) -> np.ndarray:
    """Tiny seeded 1-D k-means (no sklearn dependency). Returns sorted centers."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(data)
    if len(uniq) <= k:
        return np.sort(uniq.astype(float))
    # seed centers from evenly-spaced quantiles for determinism
    centers = np.quantile(data, [(i + 0.5) / k for i in range(k)])
    centers = centers + rng.normal(0, 1e-6, size=k)  # break exact ties, seed-stable
    for _ in range(iters):
        assign = np.abs(data[:, None] - centers[None, :]).argmin(axis=1)
        new = np.array([
            data[assign == j].mean() if np.any(assign == j) else centers[j]
            for j in range(k)
        ])
        if np.allclose(new, centers):
            break
        centers = new
    return np.sort(centers)


def _cluster_heading_sizes(
    heights_norm: list[float], k: int, seed: int, body_h_norm: float,
) -> tuple[tuple[float, float], ...]:
    """Cluster normalized heading band heights into <=k levels.

    Returns ``((threshold_norm, size_norm), ...)`` sorted by threshold
    DESCENDING, where a heading of a given normalized band height is assigned the
    first entry whose threshold it meets. Falls back to a single size when
    headings are sparse. All values are per-mille of page height.
    """
    heights = [h for h in heights_norm if _MIN_LINE_NORM <= h <= _MAX_HEADING_LINE_NORM]
    if not heights:
        return ()
    distinct = len({round(h, 1) for h in heights})
    if len(heights) < _MIN_HEADINGS_FOR_CLUSTER or distinct < k:
        size = max(body_h_norm, _dominant_mode(heights))
        # one level, threshold just below body so any heading matches
        return ((max(_MIN_LINE_NORM, body_h_norm), size),)

    data = np.asarray(heights, dtype=float)
    centers = _kmeans_1d(data, k, seed)
    assign = np.abs(data[:, None] - centers[None, :]).argmin(axis=1)
    levels: list[tuple[float, float]] = []
    for j in range(len(centers)):
        members = data[assign == j]
        if members.size == 0:
            continue
        threshold = round(float(members.min()), 2)   # lower bound of this cluster
        size = round(float(members.mean()), 2)
        levels.append((threshold, size))
    levels.sort(key=lambda t: t[0], reverse=True)
    deduped: list[tuple[float, float]] = []
    for thr, size in levels:
        if deduped and abs(deduped[-1][0] - thr) < 1e-6:
            continue
        deduped.append((thr, size))
    return tuple(deduped)


def _iter_pages(
    service: QualityEvaluationService,
    run_dir: Path,
    source_images: list[Path],
    batch_size: int,
):
    """Yield (img_path, blocks) per page, mirroring font_selection alignment."""
    content_lists = service._load_content_lists(run_dir)
    for page_idx, img_path in enumerate(source_images):
        batch_num = page_idx // batch_size
        local_idx = page_idx % batch_size
        page_blocks: list[dict] = []
        for cl_path, blocks in content_lists:
            pb = [b for b in blocks if b.get("page_idx", 0) == local_idx]
            m = _BATCH_RE.search(cl_path.name)
            if m:
                if int(m.group(1)) - 1 == batch_num and pb:
                    page_blocks = pb
                    break
            elif pb:
                page_blocks = pb
                break
        if page_blocks:
            yield img_path, page_blocks


def calibrate_typography(
    qe_config: QualityEvaluationConfig,
    run_dir: Path,
    source_images: list[Path],
    *,
    batch_size: int = 50,
    max_pages: int | None = None,
) -> QualityEvaluationConfig:
    """Measure a per-book TypographyProfile and attach it to the config.

    Args:
        qe_config: base config (read for ct_heading_levels / ct_kmeans_seed).
        run_dir: run directory with content lists (same loader as the evaluator).
        source_images: all source images for the book.
        batch_size: pages per batch (for page->batch alignment).
        max_pages: cap pages scanned (None = whole book; calibration is cheap
            relative to scoring but capping helps very large books).

    Returns:
        ``dataclasses.replace(qe_config, typography=profile)``.
    """
    service = QualityEvaluationService(qe_config)
    pages = _iter_pages(service, run_dir, source_images, batch_size)
    if max_pages is not None:
        from itertools import islice
        pages = islice(pages, max_pages)
    return calibrate_from_pages(qe_config, pages)


def calibrate_from_pages(
    qe_config: QualityEvaluationConfig,
    pages,
) -> QualityEvaluationConfig:
    """Build a TypographyProfile from pre-stitched (img_path, blocks) pages.

    Same measurement as ``calibrate_typography`` but over an arbitrary iterable
    of ``(image_path, blocks)`` pairs in global page order -- used to calibrate
    across the many per-batch run directories that make up a full book.
    """
    body_norms: list[float] = []     # per-mille of page height
    heading_norms: list[float] = []
    leadings: list[float] = []       # ratios (resolution-independent)
    x1_ratios: list[float] = []

    for img_path, blocks in pages:
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
                arr = np.array(im.convert("L"))
        except Exception:
            logger.warning("Calibration: cannot open %s", img_path)
            continue
        sx, sy = img_w / BBOX_SCALE, img_h / BBOX_SCALE
        to_norm = BBOX_SCALE / max(1, img_h)  # px height -> per-mille of page height

        for b in blocks:
            btype = b.get("type")
            bbox = b.get("bbox")
            if not bbox or btype not in ("text", "heading"):
                continue
            x0 = max(0, int(bbox[0] * sx))
            y0 = max(0, int(bbox[1] * sy))
            x1 = min(img_w, int(bbox[2] * sx))
            y1 = min(img_h, int(bbox[3] * sy))
            if btype == "text":
                x1_ratios.append(x1 / max(1, img_w))
            if (x1 - x0) < _MIN_CROP_PX or (y1 - y0) < _MIN_CROP_PX:
                continue
            crop_bin = _binarize(arr[y0:y1, x0:x1])
            heights, centers = _line_bands(crop_bin)
            if not heights:
                continue
            heights_norm = [h * to_norm for h in heights]
            if btype == "text":
                body_norms.extend(
                    h for h in heights_norm
                    if _MIN_LINE_NORM <= h <= _MAX_TEXT_LINE_NORM
                )
                if len(centers) >= 2:
                    pitches = np.diff(sorted(centers))
                    line_pitch = float(np.median(pitches))
                    med_h = float(np.median(heights))  # ratio: px/px, no normalize
                    if med_h > 0 and line_pitch > 0:
                        leadings.append(line_pitch / med_h)
            else:
                heading_norms.extend(heights_norm)

    # --- G0 coordinate sanity: text blocks should span most of the page width ---
    if x1_ratios:
        med_ratio = float(np.median(x1_ratios))
        if not (0.3 <= med_ratio <= 1.10):
            raise ValueError(
                "Typography calibration: bbox coordinate space looks wrong "
                f"(median text x1/img_w = {med_ratio:.2f}; expected ~0.5-1.0 for "
                "0-1000-normalized bboxes). Refusing to calibrate on mis-scaled "
                "coordinates. Check that content_lists are normalized (G0 gate)."
            )

    degenerate = False
    if len(body_norms) >= _MIN_BODY_SAMPLES:
        body_h_norm = max(_MIN_LINE_NORM, _dominant_mode(body_norms, bin_w=0.5))
    else:
        body_h_norm = _DEFAULT_BODY_NORM
        degenerate = True
        logger.warning(
            "Typography calibration: only %d body samples (<%d) -> degenerate "
            "fallback body_h_norm=%.1f",
            len(body_norms), _MIN_BODY_SAMPLES, body_h_norm,
        )

    if leadings:
        body_leading = float(np.clip(np.median(leadings), 1.0, 2.5))
    else:
        body_leading = 1.2
    if qe_config.ct_leading > 0:
        body_leading = qe_config.ct_leading

    heading_levels = _cluster_heading_sizes(
        heading_norms, qe_config.ct_heading_levels, qe_config.ct_kmeans_seed,
        body_h_norm,
    )

    profile = TypographyProfile(
        body_h_norm=float(body_h_norm),
        body_leading=float(body_leading),
        pitch_ratio=_PITCH_HEIGHT_RATIO,
        heading_levels=heading_levels,
        degenerate=degenerate,
    )
    logger.info(
        "Typography profile: body_h_norm=%.1f leading=%.2f headings=%s%s",
        profile.body_h_norm, profile.body_leading,
        profile.heading_levels, " (degenerate)" if degenerate else "",
    )
    return dataclasses.replace(qe_config, typography=profile)
