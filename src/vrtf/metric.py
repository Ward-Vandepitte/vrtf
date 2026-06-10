"""Image-level metrics for Visual Roundtrip Fidelity scoring.

Provides binarization, ink-overlap (DT and dilation), SSIM, band masking,
and alignment utilities extracted from QualityEvaluationService.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import fftconvolve
from skimage.metrics import structural_similarity

from vrtf.config import QualityEvaluationConfig


# ---------------------------------------------------------------------------
# Standalone alignment helpers (module-level, no state needed)
# ---------------------------------------------------------------------------

def xcorr_shift(
    orig: np.ndarray, rend: np.ndarray, max_shift: int = 30,
) -> tuple[int, int]:
    """Find (dy, dx) shift that best aligns *rend* to *orig* via FFT cross-correlation."""
    h, w = orig.shape
    oi = (orig == 0).astype(np.float32)
    ri = (rend == 0).astype(np.float32)
    if oi.sum() == 0 or ri.sum() == 0:
        return 0, 0
    corr = fftconvolve(oi, ri[::-1, ::-1], mode="full")
    cy, cx = h - 1, w - 1
    y0 = max(0, cy - max_shift)
    y1 = min(corr.shape[0], cy + max_shift + 1)
    x0 = max(0, cx - max_shift)
    x1 = min(corr.shape[1], cx + max_shift + 1)
    region = corr[y0:y1, x0:x1]
    peak = np.unravel_index(np.argmax(region), region.shape)
    return int(peak[0] + y0 - cy), int(peak[1] + x0 - cx)


def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift binary image by (*dy*, *dx*), filling with white (255)."""
    h, w = img.shape
    out = np.full_like(img, 255)
    sy0 = max(0, -dy); sy1 = min(h, h - dy)
    sx0 = max(0, -dx); sx1 = min(w, w - dx)
    dy0 = max(0, dy); dy1 = min(h, h + dy)
    dx0 = max(0, dx); dx1 = min(w, w + dx)
    sh = min(sy1 - sy0, dy1 - dy0)
    sw = min(sx1 - sx0, dx1 - dx0)
    if sh > 0 and sw > 0:
        out[dy0:dy0 + sh, dx0:dx0 + sw] = img[sy0:sy0 + sh, sx0:sx0 + sw]
    return out


# ---------------------------------------------------------------------------
# Metrics class — config-aware overlap / similarity helpers
# ---------------------------------------------------------------------------

class Metrics:
    """Image-level comparison metrics for VRTF scoring.

    Parameters
    ----------
    config : QualityEvaluationConfig
        Supplies thresholds and kernel sizes used by overlap methods.
    """

    def __init__(self, config: QualityEvaluationConfig) -> None:
        self.config = config

    # -- Binarization -------------------------------------------------------

    def binarize(self, image: np.ndarray, normalize_bg: bool = True) -> np.ndarray:
        """Binarize an image using Otsu's method with background normalization.

        Args:
            image: Grayscale or RGB numpy array.
            normalize_bg: If True, apply morphological background normalization
                before thresholding. Helps with dark/uneven paper.

        Returns:
            Binary image (0=ink, 255=background).
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Background normalization: estimate background via morphological closing,
        # then divide to normalize. This removes paper color/texture variation.
        if normalize_bg and gray.size > 0:
            bg_median = np.median(gray)
            # Only normalize if background is noticeably dark (< 240)
            if bg_median < 240:
                h, w = gray.shape
                # Kernel size ~5% of smallest dimension, at least 15px
                k = max(15, min(h, w) // 20) | 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                # Avoid division by zero
                background = np.maximum(background, 1)
                # Normalize: scale so background becomes ~255
                normalized = np.clip(
                    gray.astype(float) / background.astype(float) * 255,
                    0, 255,
                ).astype(np.uint8)
                gray = normalized

        # Otsu binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Degenerate check: if result is <2% or >98% ink, fallback to fixed threshold
        ink_ratio = np.count_nonzero(binary == 0) / max(binary.size, 1)
        if ink_ratio < 0.02 or ink_ratio > 0.98:
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        return binary

    # -- Ink overlap --------------------------------------------------------

    def compute_ink_overlap(
        self, orig_bin: np.ndarray, rend_bin: np.ndarray,
    ) -> float:
        """Compute ink overlap, dispatching to DT or dilation method."""
        if self.config.use_dt_overlap:
            return self.compute_ink_overlap_dt(orig_bin, rend_bin)
        return self.compute_ink_overlap_dilation(orig_bin, rend_bin)

    def compute_f1_overlap(
        self, orig_bin: np.ndarray, rend_bin: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute F1 ink overlap (recall + precision via DT).

        Returns (f1, recall, precision).

        In honest mode (``consistent_typography_mode``) this is the pixel
        confusion matrix (see ``compute_f1_pixel_confusion``) so the reported
        score equals the green/red/blue overlay exactly. Otherwise it is the
        original asymmetric overlap (recall over source ink, precision over
        rendered ink) -- kept byte-stable for the legacy metric.
        """
        if self.config.consistent_typography_mode:
            return self.compute_f1_pixel_confusion(orig_bin, rend_bin)
        recall = self.compute_ink_overlap(orig_bin, rend_bin)
        precision = self.compute_ink_overlap(rend_bin, orig_bin)
        if recall + precision == 0:
            return 0.0, recall, precision
        f1 = 2.0 * recall * precision / (recall + precision)
        return f1, recall, precision

    def compute_pixel_confusion_masks(
        self, orig_bin: np.ndarray, rend_bin: np.ndarray, tau: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boolean G/R/B masks of the pixel confusion partition at tolerance ``tau``.

        ``tau`` is explicit (not read from config) because the full-page path
        evaluates at an effective tau tied to the body size while
        ``config.dt_threshold`` holds the block-level value.

          * G = source ink within tau of rendered ink   (true positive)
          * R = source ink NOT within tau               (false negative)
          * B = rendered ink NOT within tau of source   (false positive)
        """
        orig_ink = orig_bin == 0
        rend_ink = rend_bin == 0
        if not orig_ink.any() and not rend_ink.any():
            z = np.zeros_like(orig_ink)
            return z, z.copy(), z.copy()

        dt_rend = cv2.distanceTransform(
            (rend_bin == 255).astype(np.uint8), cv2.DIST_L2, 3)
        dt_orig = cv2.distanceTransform(
            (orig_bin == 255).astype(np.uint8), cv2.DIST_L2, 3)
        g = orig_ink & (dt_rend <= tau)
        r = orig_ink & ~g
        b = rend_ink & (dt_orig > tau) & ~orig_ink
        return g, r, b

    def compute_f1_pixel_confusion(
        self, orig_bin: np.ndarray, rend_bin: np.ndarray,
    ) -> tuple[float, float, float]:
        """Pixel-confusion-matrix F1 at the DT tolerance ``dt_threshold``.

        A single coherent partition (the 3-color overlay):
          * green G = source ink within tau of rendered ink   (true positive)
          * red   R = source ink NOT within tau               (false negative)
          * blue  B = rendered ink NOT within tau of source   (false positive)
        Then recall = G/(G+R), precision = G/(G+B), F1 = 2G/(2G+R+B).
        The score therefore equals exactly what the overlay visualizes.

        Returns (f1, recall, precision).
        """
        orig_ink = orig_bin == 0
        rend_ink = rend_bin == 0
        oc = int(orig_ink.sum())
        rc = int(rend_ink.sum())
        if oc == 0 and rc == 0:
            return 1.0, 1.0, 1.0

        g, r, b = self.compute_pixel_confusion_masks(
            orig_bin, rend_bin, self.config.dt_threshold)
        G = int(g.sum())
        R = int(r.sum())
        B = int(b.sum())

        recall = G / (G + R) if (G + R) > 0 else 1.0
        precision = G / (G + B) if (G + B) > 0 else 1.0
        denom = 2 * G + R + B
        f1 = (2 * G) / denom if denom > 0 else 0.0
        return f1, recall, precision

    def compute_ink_overlap_dt(
        self, orig_bin: np.ndarray, rend_bin: np.ndarray,
    ) -> float:
        """Compute ink overlap using distance-transform tolerance.

        For each original ink pixel, checks if it is within dt_threshold
        pixels of any rendered ink pixel. More principled than dilation
        because it degrades gracefully with distance.

        Returns overlap / original_ink. Empty original returns 1.0.
        """
        orig_ink = (orig_bin == 0)
        orig_count = np.count_nonzero(orig_ink)
        if orig_count == 0:
            return 1.0

        # Distance transform of rendered background: distance from each
        # pixel to the nearest rendered ink pixel.
        rend_bg = (rend_bin == 255).astype(np.uint8)
        dt = cv2.distanceTransform(rend_bg, cv2.DIST_L2, 3)

        # Count original ink pixels within threshold distance of rendered ink
        overlap = np.count_nonzero(orig_ink & (dt <= self.config.dt_threshold))
        return overlap / orig_count

    def compute_ink_overlap_dilation(
        self, orig_bin: np.ndarray, rend_bin: np.ndarray,
    ) -> float:
        """Compute ink overlap using morphological dilation (old method).

        Dilates rendered ink by overlap_dilation_kernel, then counts
        original ink pixels that fall within the dilated region.

        Returns overlap / original_ink. Empty original returns 1.0.
        """
        orig_ink = (orig_bin == 0)
        orig_count = np.count_nonzero(orig_ink)
        if orig_count == 0:
            return 1.0

        kernel = np.ones(
            (self.config.overlap_dilation_kernel,) * 2, np.uint8,
        )
        rend_ink = (rend_bin == 0).astype(np.uint8)
        rend_dilated = cv2.dilate(rend_ink, kernel, iterations=1)
        overlap = np.count_nonzero(orig_ink & (rend_dilated > 0))
        return overlap / orig_count

    # -- SSIM ---------------------------------------------------------------

    @staticmethod
    def compute_ssim(
        orig_bin: np.ndarray, rend_bin: np.ndarray,
    ) -> float:
        """Compute SSIM between two binarized images.

        Returns 0.0 if either dimension is smaller than SSIM window size.
        """
        win_size = 7
        h, w = orig_bin.shape[:2]
        if h < win_size or w < win_size:
            return 0.0

        return float(structural_similarity(orig_bin, rend_bin, win_size=win_size))

    # -- Band masking -------------------------------------------------------

    @staticmethod
    def build_band_mask(
        bands: tuple[tuple[int, int], ...], h: int, w: int, pad: int = 4,
    ) -> np.ndarray:
        """Build a boolean mask that is True within band intervals +/- pad.

        Used to mask non-text regions (figures/equations) in the original
        before comparison, so they don't penalize the overlap score.
        """
        mask = np.zeros((h, w), dtype=bool)
        for start, end in bands:
            y0 = max(0, start - pad)
            y1 = min(h, end + pad)
            mask[y0:y1, :] = True
        return mask
