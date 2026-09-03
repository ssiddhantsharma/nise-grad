#!/usr/bin/env bash
# Fetch the LigandMPNN weights needed by the rescue / projector / guided-diffusion scripts
# (rescue_backbone.py, project.py, guided_design.py). These are third-party model files from
# the official LigandMPNN repo (github.com/dauparas/LigandMPNN); they are not stored in this
# repo. The core STE panel and baselines do NOT need them -- they run on Boltz-2 alone.
#
# Boltz-2 weights (boltz2_conf.ckpt, boltz2_aff.ckpt) plus the CCD data (mols.tar, ccd.pkl)
# are handled separately: the boltz CLI downloads them to ~/.boltz on its first run. See README.
#
# Usage:
#   bash scripts/get_weights.sh            # into ./weights
#   WEIGHTS_DIR=/data/weights bash scripts/get_weights.sh
# Then export the two paths it prints (or add them to your shell profile).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
CKPT="$WEIGHTS_DIR/ligandmpnn_v_32_010_25.pt"

# Verified primary source:
#   checkpoint -> dauparas/LigandMPNN get_model_params.sh (hosted on the IPD file server)
# The torch reference module is NOT needed here: jigandmpnn (the JAX LigandMPNN this repo
# depends on) vendors it and does the torch->JAX conversion internally.
CKPT_URL="https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt"

mkdir -p "$WEIGHTS_DIR"

fetch() {  # $1 url  $2 dest
  if [ -s "$2" ]; then echo "  exists, skipping: $2"; return; fi
  echo "  fetching $1"
  if command -v curl >/dev/null 2>&1; then curl -fSL "$1" -o "$2"
  else wget -q "$1" -O "$2"; fi
}

echo "LigandMPNN checkpoint ->"
fetch "$CKPT_URL" "$CKPT"

# Sanity: the checkpoint is a torch pickle (>1 MB), not an HTML error page.
if [ "$(wc -c < "$CKPT")" -lt 1000000 ]; then
  echo "WARNING: $CKPT is under 1 MB -- the download may have failed. Check the URL/network." >&2
fi

cat <<EOF

Done. Set this before running rescue_backbone.py / project.py / guided_design.py:

  export LIGANDMPNN_CKPT="$CKPT"

(The core STE panel and baselines do not need it; they run on Boltz-2 alone.)
EOF
