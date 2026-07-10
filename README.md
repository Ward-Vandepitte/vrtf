# VRTF — Visual Roundtrip Fidelity

A **reference-free** metric of **structural-reconstruction fidelity** for document
OCR: render the parser's output back into an image — without consulting the source
pixels — and score how much of the original page's ink the reconstruction reproduces.

VRTF needs **no ground truth** (only the source image the parser already consumed),
is **parser-agnostic** (any block-level OCR output), and stays discriminative after
character error rates saturate.

Companion code for the paper *Beyond Edit Distance: Reference-Free
Structural-Reconstruction Fidelity for Document OCR* (Vandepitte, 2026; submitted to
engrXiv, DOI to follow on posting).

## How it works (the honest protocol)

1. **Calibrate** — once per document, measure the dominant body line-height and
   heading sizes (`vrtf.calibrate.calibrate_from_pages`). This book-global statistic
   is the protocol's single, deliberate exception to source-blindness: an aggregate
   over hundreds of pages cannot absorb any individual block's detection error.
2. **Render, source-blind** — every block is re-typeset at the calibrated typography
   and placed at its detected bounding box. The renderer is forbidden from per-block
   font resizing, cross-correlation alignment, or source-guided layout — the
   auto-compensations that would let a renderer hide the parser's errors. Equations
   compile via pdflatex at body scale; tables render as plain grids; figure blocks
   are not reconstructed and count as missed (the parser's output contains nothing
   to rebuild them from).
3. **Score as a pixel confusion matrix** — overlay the binarized reconstruction on
   the binarized scan: green = source ink matched within tolerance τ, red = source
   ink missed, blue = spurious rendered ink. The **VRTF of a page is the F1** of that
   partition, `2G/(2G+R+B)`; a document's VRTF is the mean over its pages. τ is tied
   to the calibrated body line-height (τ = 0.5·h_body), making scores comparable
   across scan resolutions.

## Installation

```bash
pip install vrtf
```

System packages for equation rendering (recommended):

```bash
sudo apt install texlive-latex-base texlive-latex-extra poppler-utils
```

Optional extras:

```bash
pip install "vrtf[figures]"    # matplotlib + cairosvg (figure-recreation avenue runs)
pip install "vrtf[equations]"  # pdfminer-six (hybrid glyph stamping)
pip install "vrtf[photos]"     # anthropic SDK (photo-description scoring; needs ANTHROPIC_API_KEY)
```

The default font is `Nimbus Mono PS Bold` at its Debian/Ubuntu path; on other
systems set `QualityEvaluationConfig.font_path` to any monospace font (a warning is
logged if the font is missing — scores are unreliable with the PIL fallback font).

## Quick start — reproducing the paper's protocol

```python
import dataclasses
from vrtf import QualityEvaluationConfig, QualityEvaluationService, calibrate_from_pages

# pages: list of (source_image_path, blocks) for calibration (a stride sample is fine)
base = QualityEvaluationConfig(full_page_mode=True, generate_overlays=False,
                               font_selection_enabled=False)
base = calibrate_from_pages(base, pages)          # per-document typography profile

cfg = dataclasses.replace(base,
                          consistent_typography_mode=True,  # the honest preset
                          ct_tau_body_frac=0.5)             # τ = 0.5 · body line-height
service = QualityEvaluationService(cfg)

ps = service.evaluate_page(source_image_path, blocks, page_idx=0)
print(f"page VRTF: {ps.full_page_f1:.3f}")
```

`consistent_typography_mode=True` is a preset over granular `ct_*` toggles (see
`config.py`) that disables every auto-compensation at once; `full_page_mode=True`
enables the full-page composite that the paper reports. The library also retains the
**legacy source-aware mode** (the default config) for ablation studies — it produces
*systematically inflated* scores by consulting the source during rendering and should
not be used for evaluation claims; the paper quantifies this inflation.

## Input format

A list of block dicts per page (the MinerU-style content-list interchange schema;
bounding boxes normalized to a 1000×1000 grid):

```python
{
    "type": "text",            # text | heading | equation | table | image | discarded
    "bbox": [x0, y0, x1, y1],  # normalized to [0, 1000]
    "text": "...",             # text/heading: plain text; equation: LaTeX
    "page_idx": 0,             # page index within the batch (default 0)
}
```

Optional fields consumed when present:

- `table_body` — table content (pipe-delimited or HTML) for table blocks;
- `figure_type` — `"photo" | "line_graph" | "technical_drawing"` on image blocks;
- `graph_extraction: {"status": "success", "data": {...}}` — parsed chart series
  (used only in the figure-recreation avenue runs, `ct_figure_recreation`);
- `drawing_extraction: {"status": "success", "svg": "..."}` — vectorized drawing
  (same avenue runs);
- `img_path` — key into `formula_cleanup_path` JSON for the formula
  compile-failure-recovery fallback.

Without the optional fields, figure blocks simply count as missed — which is the
paper's baseline protocol.

## Results from the paper (honest protocol, τ = 0.5·h_body)

| Document | Pages | Mean VRTF | Pages < 70 % |
|----------|------:|----------:|-------------:|
| Vandepitte *Berekening van Constructies* I (1979) | 560 | 72.2 % | 33 % |
| Vandepitte II (1979) | 691 | 67.5 % | 50 % |
| Vandepitte III (1982) | 692 | 73.1 % | 32 % |
| CUR 198 (1990s) | 258 | 68.6 % | 32 % |
| CUR 226 (2016) | 129 | 54.3 % | 88 % |
| Abbs 1988 | 5 | 65.2 % | 60 % |
| Bruce 1986 | 16 | 53.6 % | 94 % |

Parser: dots.ocr (1.7B). On the same corpus the character error rate is 0.0045 —
text recognition is saturated while structural reconstruction is not, and the two
are uncorrelated per page. See the paper for the decomposition of the gap by content
type, the capability ("avenue") runs, and the measured score inflation of
source-peeking rendering.

## Security note

Equation rendering shells out to `pdflatex` with `--no-shell-escape` and 5-second
timeouts. LaTeX from untrusted OCR output is still compiled; do not run the metric
on adversarial documents in a privileged environment.

## Citation

```
Vandepitte, W. (2026). Beyond Edit Distance: Reference-Free
Structural-Reconstruction Fidelity for Document OCR. engrXiv preprint.
```

(engrXiv DOI added on posting.)

## License

Apache-2.0
