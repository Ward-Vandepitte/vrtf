"""Overlay and report generation for VRTF quality evaluation."""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from vrtf.config import QualityEvaluationConfig
from vrtf.models import BBOX_SCALE, BookEvaluation, PageScore
from vrtf.utils.font import FontCache
from vrtf.utils.latex import simplify_latex


class OverlayGenerator:
    """Generates side-by-side overlay images and markdown reports."""

    def __init__(self, config: QualityEvaluationConfig, fonts: FontCache):
        self.config = config
        self.fonts = fonts

    def generate_overlay(
        self,
        source_image_path: Path,
        page_score: PageScore,
        blocks: list[dict],
        output_path: Path,
    ) -> Path:
        """Generate a side-by-side overlay image at half resolution.

        Left: original with color-coded bbox outlines.
        Right: rendered text blocks.
        Green >80%, yellow 50-80%, red <50%.
        """
        with Image.open(source_image_path) as img:
            img_w, img_h = img.size

            # Half resolution
            half_w, half_h = img_w // 2, img_h // 2
            img_half = img.resize((half_w, half_h), Image.LANCZOS)

        # Create right panel (rendered text)
        render_panel = Image.new("RGB", (half_w, half_h), (255, 255, 255))
        render_draw = ImageDraw.Draw(render_panel)

        scale_x = img_w / BBOX_SCALE
        scale_y = img_h / BBOX_SCALE

        # Draw on left panel
        left_draw = ImageDraw.Draw(img_half)

        block_idx = 0
        for block in blocks:
            block_type = block.get("type", "unknown")
            text = block.get("text", "")
            bbox = block.get("bbox")

            if bbox is None:
                block_idx += 1
                continue

            # Map to pixel coords (full res), then halve
            px_x0 = int(bbox[0] * scale_x) // 2
            px_y0 = int(bbox[1] * scale_y) // 2
            px_x1 = int(bbox[2] * scale_x) // 2
            px_y1 = int(bbox[3] * scale_y) // 2

            if block_type != "text":
                # Gray outline for non-text
                left_draw.rectangle([px_x0, px_y0, px_x1, px_y1],
                                    outline=(128, 128, 128), width=1)
                block_idx += 1
                continue

            if not text or not text.strip():
                block_idx += 1
                continue

            # Find matching BlockScore
            score = None
            if block_idx < len(page_score.block_scores):
                score = page_score.block_scores[block_idx]

            # Color based on score
            if score and score.evaluated:
                pct = (self.config.weight_ink_overlap * score.ink_overlap
                       + self.config.weight_ssim * score.ssim) * 100
                if pct > 80:
                    color = (0, 200, 0)
                elif pct > 50:
                    color = (200, 200, 0)
                else:
                    color = (200, 0, 0)
            else:
                color = (128, 128, 128)

            left_draw.rectangle([px_x0, px_y0, px_x1, px_y1],
                                outline=color, width=2)

            # Render text on right panel (simplified, small font)
            w_px = max(1, px_x1 - px_x0)
            h_px = max(1, px_y1 - px_y0)
            try:
                font_size = max(6, min(h_px - 2, 14))
                font = self.fonts.get_font(font_size)
                simplified = simplify_latex(text)
                if '\n' in simplified:
                    wrapped = simplified  # preserve line structure
                else:
                    wrapped = textwrap.fill(simplified, width=max(1, w_px // max(1, font_size // 2)))
                render_draw.text((px_x0 + 2, px_y0 + 1), wrapped,
                                 fill=(0, 0, 0), font=font)
            except Exception:
                pass

            block_idx += 1

        # Combine side-by-side
        combined = Image.new("RGB", (half_w * 2, half_h), (255, 255, 255))
        combined.paste(img_half, (0, 0))
        combined.paste(render_panel, (half_w, 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.save(output_path, "PNG")
        return output_path

    def generate_report(
        self,
        evaluation: BookEvaluation,
        output_path: Path,
    ) -> Path:
        """Generate a markdown report from evaluation results."""
        lines = ["# OCR Quality Evaluation Report\n"]

        scored = [ps for ps in evaluation.page_scores if ps.evaluated_scores]
        lines.append(f"**Total pages:** {len(evaluation.page_scores)}")
        lines.append(f"**Pages with scores:** {len(scored)}")
        lines.append(f"**Average score:** {evaluation.average_score:.1f}%\n")

        # Worst pages
        worst = evaluation.worst_pages()
        if worst:
            lines.append("## Worst Pages\n")
            lines.append("| Page | Score | Text Quality | SSIM | Evaluated | Empty | Skipped |")
            lines.append("|------|-------|-------------|------|-----------|-------|---------|")
            for ps in worst:
                lines.append(
                    f"| {ps.page_idx} | {ps.score_percent():.1f}% "
                    f"| {ps.text_quality:.3f} | {ps.ssim_average:.3f} "
                    f"| {len(ps.evaluated_scores)} | {ps.empty_block_count} "
                    f"| {ps.skipped_block_count} |"
                )
            lines.append("")

        # All pages table
        lines.append("## All Pages\n")
        lines.append("| Page | Score | Text Quality | SSIM | Blocks | Empty | Skipped |")
        lines.append("|------|-------|-------------|------|--------|-------|---------|")
        for ps in evaluation.page_scores:
            lines.append(
                f"| {ps.page_idx} | {ps.score_percent():.1f}% "
                f"| {ps.text_quality:.3f} | {ps.ssim_average:.3f} "
                f"| {len(ps.evaluated_scores)} | {ps.empty_block_count} "
                f"| {ps.skipped_block_count} |"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
