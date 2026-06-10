# Per-page scores and experiment data

Complete per-page data behind the paper's tables (honest protocol,
τ = 0.5·h_body unless stated). Page indices refer to the natsorted scan
sequence of each (copyrighted, not redistributable) source document, so
results are verifiable against lawfully obtained copies.

- `structure_fullbook_*.json` — per-page VRTF for the seven corpus documents
  (paper Table 2).
- `structure_decomposition_deel_ii.json` — per-content-type G/R/B gap
  decomposition + first-order counterfactuals (Table 3).
- `structure_ablation_deel_ii.json` — avenue-run matrix: baseline, figure
  recreation (source-blind & peeking contrast), formula compile-failure
  recovery, combined; paired deltas, bootstrap CIs, counters (Table 4).
- `baseline_metric_comparison_deel_ii.json` — full-page SSIM and strict
  IoU (τ=0) vs VRTF on a 70-page stride sample (§3.4).
- `structure_fullbook_deel_ii_mineru.json` — second-parser column: archival
  MinerU pass, paired against dots.ocr (§5).
