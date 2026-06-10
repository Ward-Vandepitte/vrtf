"""Table rendering for VRTF quality evaluation.

Extracts and renders markdown/HTML tables into binarized images
for pixel-level comparison against original scans.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from vrtf.utils.latex import simplify_latex

if TYPE_CHECKING:
    from PIL import ImageFont

    from vrtf.config import QualityEvaluationConfig


from vrtf.utils.protocols import FontCacheProtocol as FontCache  # noqa: E402


class TableRenderer:
    """Render parsed tables into binarized images for VRTF scoring."""

    def __init__(
        self,
        config: QualityEvaluationConfig,
        fonts: FontCache,
    ) -> None:
        self.config = config
        self.fonts = fonts

    # -- Public entry point ---------------------------------------------------

    def render_table_in_bbox(
        self,
        text: str,
        w_px: int,
        h_px: int,
    ) -> np.ndarray | None:
        """Render markdown pipe table as PIL image with grid lines and cell text.

        Returns grayscale numpy array (white bg, black ink), or None if
        table is unparseable or too dense to render.
        """
        grid = self.parse_markdown_table(text)
        if not grid:
            return None
        return self.render_grid_in_bbox(grid, w_px, h_px)

    # -- Grid rendering -------------------------------------------------------

    def render_grid_consistent(
        self,
        grid: list[list[str]],
        w_px: int,
        h_px: int,
        *,
        font_px: int,
        leading: float,
    ) -> np.ndarray | None:
        """Honest table render: full grid at consistent typography, top-left.

        Uses the per-book body font size (NOT a bbox-derived size) and lays out
        rows at the body line height and columns at their content width, anchored
        at the bbox top-left. No source grid detection (that measures the answer
        from the source image) and no stretch-to-bbox. Ink that overflows the
        bbox is clipped (precision loss); under-fill stays blank (recall loss).
        """
        if not grid:
            return None
        n_rows = len(grid)
        n_cols = max(len(r) for r in grid)
        if n_cols == 0:
            return None

        font = self.fonts.get_font(font_px)
        row_h = max(font_px + 4, int(round(font_px * leading)))
        pad = 6
        # content-driven column widths (from OCR cell text + consistent font)
        col_ws: list[int] = []
        for j in range(n_cols):
            wmax = 12
            for i in range(n_rows):
                cell = grid[i][j] if j < len(grid[i]) else ""
                if cell:
                    wmax = max(wmax, int(font.getlength(cell)) + pad)
            col_ws.append(wmax)

        img = Image.new("L", (w_px, h_px), 255)
        draw = ImageDraw.Draw(img)
        col_xs = [0]
        for cw in col_ws:
            col_xs.append(col_xs[-1] + cw)
        total_w = col_xs[-1]
        total_h = n_rows * row_h

        # horizontal grid lines (clipped to canvas)
        for i in range(n_rows + 1):
            y = i * row_h
            if y >= h_px:
                break
            draw.line([(0, y), (min(total_w, w_px) - 1, y)], fill=0, width=1)
        # vertical grid lines
        for x in col_xs:
            if x >= w_px:
                break
            draw.line([(x, 0), (x, min(total_h, h_px) - 1)], fill=0, width=1)

        # cell text (clipped naturally by the canvas bounds)
        for i in range(n_rows):
            y = i * row_h
            if y >= h_px:
                break
            for j in range(n_cols):
                x = col_xs[j]
                if x >= w_px:
                    break
                cell = grid[i][j] if j < len(grid[i]) else ""
                if cell:
                    draw.text((x + 2, y + 2), cell, fill=0, font=font)

        return np.array(img)

    def render_grid_in_bbox(
        self,
        grid: list[list[str]],
        w_px: int,
        h_px: int,
        *,
        col_xs: list[int] | None = None,
        row_ys: list[int] | None = None,
        cell_types: list[list[str]] | None = None,
    ) -> np.ndarray | None:
        """Render a 2D grid of cell strings as PIL image with grid lines.

        When col_xs / row_ys are provided (from image-guided detection),
        uses detected positions for column widths and row heights.
        Falls back to equal distribution when None (backward compatible).

        cell_types: optional grid of 'solid'/'empty'/'text' per cell.
        'solid' cells get black fill, 'empty' cells skip text rendering.

        Returns grayscale numpy array (white bg, black ink), or None if
        grid is empty or too dense to render.
        """
        if not grid:
            return None

        n_rows = len(grid)
        n_cols = max(len(r) for r in grid)
        if n_cols == 0:
            return None

        # Row heights: from detected positions or equal distribution
        if row_ys is not None:
            row_hs = [row_ys[i + 1] - row_ys[i] for i in range(len(row_ys) - 1)]
        else:
            row_h = h_px // n_rows
            row_hs = [row_h] * n_rows
            row_hs[-1] = h_px - row_h * (n_rows - 1)

        # Guard: skip if too dense to render meaningfully
        if min(row_hs) < 8:
            return None

        # Column widths: from detected positions or equal distribution
        if col_xs is not None:
            col_ws = [col_xs[j + 1] - col_xs[j] for j in range(len(col_xs) - 1)]
        else:
            col_ws = [w_px // n_cols] * n_cols
            col_ws[-1] = w_px - sum(col_ws[:-1])

        # Font size: constrained by row height and narrowest text column
        # Filter to columns that have text content, minimum 12px
        text_col_ws = []
        for j in range(n_cols):
            has_text = False
            for i in range(n_rows):
                ct = "text"
                if cell_types and i < len(cell_types) and j < len(cell_types[i]):
                    ct = cell_types[i][j]
                if ct == "text" and i < len(grid) and j < len(grid[i]) and grid[i][j]:
                    has_text = True
                    break
            if has_text and col_ws[j] >= 12:
                text_col_ws.append(col_ws[j])
        min_col_w = min(text_col_ws) if text_col_ws else min(col_ws)

        min_row_h = min(row_hs)
        font_size = max(6, int(min_row_h * 0.7 * self.config.font_size_multiplier))
        font = self.fonts.get_font(font_size)
        while font_size > 6 and int(font.getlength("W")) + 4 > min_col_w:
            font_size -= 1
            font = self.fonts.get_font(font_size)
        if int(font.getlength("W")) + 4 > min_col_w:
            return None

        img = Image.new("L", (w_px, h_px), 255)
        draw = ImageDraw.Draw(img)

        # Draw grid borders at detected or computed positions
        y = row_ys[0] if row_ys is not None else 0
        for i in range(n_rows + 1):
            draw.line([(0, y), (w_px - 1, y)], fill=0, width=1)
            if i < n_rows:
                y += row_hs[i]
        x = col_xs[0] if col_xs is not None else 0
        for j in range(n_cols + 1):
            draw.line([(x, 0), (x, h_px - 1)], fill=0, width=1)
            if j < n_cols:
                x += col_ws[j]

        # Draw cell content
        y = row_ys[0] if row_ys is not None else 0
        for i, row in enumerate(grid):
            x = col_xs[0] if col_xs is not None else 0
            for j, cell in enumerate(row):
                cw = col_ws[j] if j < n_cols else col_ws[-1]
                rh = row_hs[i] if i < n_rows else row_hs[-1]

                # Get cell type
                ct = "text"
                if cell_types and i < len(cell_types) and j < len(cell_types[i]):
                    ct = cell_types[i][j]

                if ct == "solid":
                    draw.rectangle([(x, y), (x + cw, y + rh)], fill=0)
                elif ct != "empty" and cell:
                    while cell and font.getlength(cell) > cw - 4:
                        cell = cell[:-1]
                    pad_y = max(0, (rh - font_size) // 2)
                    draw.text((x + 2, y + pad_y), cell, fill=0, font=font)

                x += cw
            y += row_hs[i] if i < n_rows else row_hs[-1]

        return np.array(img)

    # -- Grid detection -------------------------------------------------------

    @staticmethod
    def detect_table_grid_lines(
        orig_bin: np.ndarray,
        expected_rows: int,
        expected_cols: int,
    ) -> tuple[list[int] | None, list[int] | None]:
        """Detect grid line positions from binarized table crop.

        Uses projection profiles with run-width filtering to find narrow
        ink bands that span a significant fraction of the image dimension
        (grid lines), while rejecting wide bands (solid fills, dense text).

        Returns (row_ys, col_xs) where each is a list of pixel positions
        for grid lines (len = expected + 1 for borders), or None per axis
        if detection fails for that axis.
        """
        h, w = orig_bin.shape[:2]

        H_THRESHOLD_FRAC = 0.5   # horizontal lines span >=50% of width
        V_THRESHOLD_FRAC = 0.3   # vertical lines span >=30% of height
        MAX_LINE_WIDTH = 8        # grid lines are narrow
        EDGE_SNAP_FRAC = 0.05    # snap to edge if within 5%

        def _find_grid_lines(projection: np.ndarray, threshold: int,
                             dim: int, expected_count: int) -> list[int] | None:
            """Find grid line positions in a 1D projection profile.

            Args:
                projection: ink count per row/col
                threshold: minimum ink to be considered a line candidate
                dim: total dimension (h or w) for edge snapping
                expected_count: expected number of grid lines (rows+1 or cols+1)

            Returns:
                List of pixel positions, or None if count doesn't match.
            """
            above = projection >= threshold
            if not np.any(above):
                return None

            # Find contiguous runs using diff + where
            padded = np.concatenate(([False], above, [False]))
            diffs = np.diff(padded.astype(np.int8))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]

            # Filter by run width: keep only narrow runs (grid lines)
            centers = []
            for s, e in zip(starts, ends):
                run_width = e - s
                if run_width <= MAX_LINE_WIDTH:
                    centers.append((s + e) // 2)

            if not centers:
                return None

            # Edge snapping: snap lines near edges to 0 / dim-1
            edge_threshold = int(dim * EDGE_SNAP_FRAC)
            if centers[0] <= edge_threshold:
                centers[0] = 0
            if centers[-1] >= dim - 1 - edge_threshold:
                centers[-1] = dim - 1

            # Edge inference: if short by 1-2 lines, add missing edges
            deficit = expected_count - len(centers)
            if 1 <= deficit <= 2:
                if centers[0] != 0:
                    centers.insert(0, 0)
                if len(centers) < expected_count and centers[-1] != dim - 1:
                    centers.append(dim - 1)

            # Sanity: count must match expected (borders = n_items + 1)
            if len(centers) != expected_count:
                return None

            # Sanity: reject if any cell is < 33% of mean size (lopsided detection)
            if len(centers) >= 2:
                spans = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
                mean_span = sum(spans) / len(spans)
                if mean_span > 0 and min(spans) < 0.33 * mean_span:
                    return None

            return centers

        # Horizontal grid lines: ink per row, threshold on width
        ink_per_row = np.count_nonzero(orig_bin == 0, axis=1)
        h_threshold = int(w * H_THRESHOLD_FRAC)
        row_ys = _find_grid_lines(ink_per_row, h_threshold, h, expected_rows + 1)

        # Vertical grid lines: ink per column, threshold on height
        ink_per_col = np.count_nonzero(orig_bin == 0, axis=0)
        v_threshold = int(h * V_THRESHOLD_FRAC)
        col_xs = _find_grid_lines(ink_per_col, v_threshold, w, expected_cols + 1)

        if row_ys is None and col_xs is None:
            return None, None
        return row_ys, col_xs

    # -- Cell classification --------------------------------------------------

    @staticmethod
    def classify_table_cells(
        orig_bin: np.ndarray,
        row_ys: list[int],
        col_xs: list[int],
    ) -> list[list[str]]:
        """Classify each cell as 'solid', 'empty', or 'text'.

        Uses ink density + uniformity within the cell region (excluding
        grid line borders via margin).
        """
        n_rows = len(row_ys) - 1
        n_cols = len(col_xs) - 1
        result: list[list[str]] = []

        # Estimate line width from grid line runs for margin calculation
        line_width_estimate = 2  # default

        for i in range(n_rows):
            row_result: list[str] = []
            for j in range(n_cols):
                top, bot = row_ys[i], row_ys[i + 1]
                left, right = col_xs[j], col_xs[j + 1]
                cell_h = bot - top
                cell_w = right - left
                margin = min(max(2, line_width_estimate), cell_h // 4, cell_w // 4)
                # Clamp to ensure non-empty crop
                t = top + margin
                b = bot - margin
                l = left + margin
                r = right - margin
                if b <= t or r <= l:
                    row_result.append("empty")
                    continue

                cell = orig_bin[t:b, l:r]
                total = cell.size
                if total == 0:
                    row_result.append("empty")
                    continue

                ink = np.count_nonzero(cell == 0)
                density = ink / total

                if density > 0.6:
                    # Uniformity check: solid fill vs dense text
                    row_densities = np.count_nonzero(cell == 0, axis=1) / cell.shape[1]
                    if np.std(row_densities) < 0.2:
                        row_result.append("solid")
                        continue

                if density < 0.01:
                    row_result.append("empty")
                else:
                    row_result.append("text")
            result.append(row_result)

        return result

    # -- Parsing --------------------------------------------------------------

    @staticmethod
    def parse_markdown_table(text: str) -> list[list[str]]:
        """Parse markdown pipe table into grid of cell strings.

        Skips separator rows (3+ consecutive dashes).
        Handles missing outer pipes and ragged columns.
        Runs simplify_latex on cell content. Returns empty list if no data rows.

        Column count is anchored to the separator row (if present) or the
        first data row.  Data rows with more pipes than the anchor get
        excess cells merged into the last column -- this handles embedded
        ``|`` characters in Docling's table markdown export.
        """
        sep_re = re.compile(r'^\|?[\s:]*-{3,}[\s:|-]*$')

        def _split_row(line: str) -> list[str]:
            cells = [c.strip() for c in line.split('|')]
            if cells and cells[0] == '':
                cells = cells[1:]
            if cells and cells[-1] == '':
                cells = cells[:-1]
            return cells

        # First pass: find anchor column count from separator row
        anchor_cols = 0
        for raw_line in text.split('\n'):
            line = raw_line.strip()
            if line and sep_re.match(line):
                anchor_cols = len(_split_row(line))
                break

        # If no separator, use the first data row
        if anchor_cols == 0:
            for raw_line in text.split('\n'):
                line = raw_line.strip()
                if line and not sep_re.match(line) and '|' in line:
                    anchor_cols = len(_split_row(line))
                    break

        if anchor_cols == 0:
            return []

        # Second pass: parse data rows, clamping to anchor_cols
        rows: list[list[str]] = []
        for raw_line in text.split('\n'):
            line = raw_line.strip()
            if not line or sep_re.match(line):
                continue
            cells = _split_row(line)
            if not cells:
                continue
            # Merge excess cells into the last column
            if len(cells) > anchor_cols:
                merged = ' | '.join(cells[anchor_cols - 1:])
                cells = cells[:anchor_cols - 1] + [merged]
            cells = [simplify_latex(c) for c in cells]
            rows.append(cells)

        if not rows:
            return []
        for r in rows:
            while len(r) < anchor_cols:
                r.append('')
        return rows

    @staticmethod
    def parse_html_table(html: str) -> list[list[str]]:
        """Parse HTML table into grid of cell strings.

        Handles <tr>, <td>/<th>, rowspan/colspan.
        Spanned positions are filled with empty strings.
        Tags are stripped from cell text content.
        """
        from html.parser import HTMLParser

        class _TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows: list[list[tuple[str, int, int]]] = []  # (text, rowspan, colspan)
                self._current_row: list[tuple[str, int, int]] = []
                self._in_cell = False
                self._cell_text = ""
                self._cell_rowspan = 1
                self._cell_colspan = 1

            def handle_starttag(self, tag, attrs):
                if tag == "tr":
                    self._current_row = []
                elif tag in ("td", "th"):
                    self._in_cell = True
                    self._cell_text = ""
                    self._cell_rowspan = 1
                    self._cell_colspan = 1
                    for name, value in attrs:
                        if name == "rowspan":
                            try:
                                self._cell_rowspan = int(value)
                            except (ValueError, TypeError):
                                pass
                        elif name == "colspan":
                            try:
                                self._cell_colspan = int(value)
                            except (ValueError, TypeError):
                                pass

            def handle_endtag(self, tag):
                if tag in ("td", "th") and self._in_cell:
                    self._current_row.append(
                        (self._cell_text.strip(), self._cell_rowspan, self._cell_colspan))
                    self._in_cell = False
                elif tag == "tr" and self._current_row:
                    self.rows.append(self._current_row)

            def handle_data(self, data):
                if self._in_cell:
                    self._cell_text += data

        parser = _TableParser()
        parser.feed(html)
        if not parser.rows:
            return []

        # Determine grid dimensions
        n_rows = len(parser.rows)
        n_cols = 0
        for row in parser.rows:
            cols = sum(cs for _, _, cs in row)
            n_cols = max(n_cols, cols)
        if n_cols == 0:
            return []

        # Expand spans into a 2D grid
        grid: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]
        # Track occupied cells (from rowspan/colspan)
        occupied: set[tuple[int, int]] = set()

        for ri, row in enumerate(parser.rows):
            ci = 0
            for text, rspan, cspan in row:
                # Find next unoccupied column
                while ci < n_cols and (ri, ci) in occupied:
                    ci += 1
                if ci >= n_cols:
                    break
                grid[ri][ci] = text
                # Mark spanned positions as occupied
                for dr in range(rspan):
                    for dc in range(cspan):
                        if ri + dr < n_rows and ci + dc < n_cols:
                            occupied.add((ri + dr, ci + dc))
                ci += cspan

        return grid
