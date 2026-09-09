#!/usr/bin/env bash
# Fetch the two Anthropic de novo protein-binder tables the analysis reads (~10 MB).
# CPU-only; no GPU, no auth. Source: HF Anthropic/claude-protein-binder-design.
set -euo pipefail
BASE="https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/resolve/main/data"
curl -sL -o /tmp/cofold.parquet         "$BASE/tables/insilico/cofold_predictions.parquet"
curl -sL -o /tmp/design_summary.parquet "$BASE/tables/design_summary.parquet"
for f in /tmp/cofold.parquet /tmp/design_summary.parquet; do
  [ "$(head -c4 "$f")" = "PAR1" ] || { echo "ERROR: $f is not a parquet"; exit 1; }
done
echo "ok: /tmp/cofold.parquet + /tmp/design_summary.parquet"
