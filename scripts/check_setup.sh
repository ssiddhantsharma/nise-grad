#!/usr/bin/env bash
# Doctor: verify the environment before a run, so failures surface up front instead of three hours in.
# For each artifact it reports present-on-disk OR whether the source URL still resolves, plus the Python
# version of each venv. Non-destructive and read-only; it never downloads anything and always exits 0.
#
#   bash scripts/check_setup.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ok=0; need=0
G="\033[32m"; R="\033[31m"; Y="\033[33m"; Z="\033[0m"
[ -t 1 ] || { G=""; R=""; Y=""; Z=""; }  # no color codes when piped or logged
pass(){ printf "  ${G}ok${Z}   %s\n" "$1"; ok=$((ok+1)); }
miss(){ printf "  ${R}MISS${Z} %s\n" "$1"; need=$((need+1)); }
note(){ printf "  ${Y}--${Z}   %s\n" "$1"; }
pyver(){ "$1" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "?"; }
url_ok(){ case "$(curl -sI -o /dev/null -w '%{http_code}' --max-time 15 "$1" 2>/dev/null)" in 2*|3*) return 0;; *) return 1;; esac; }
reach(){ url_ok "$1" && note "source reachable: $1" || note "SOURCE UNREACHABLE: $1"; }

echo "== nise-grad venv (Python 3.12) =="
if [ -x "$REPO/.venv/bin/python" ]; then
  v=$(pyver "$REPO/.venv/bin/python")
  [ "$v" = "3.12" ] && pass "venv Python $v" || miss "venv Python $v (want 3.12; run: uv sync)"
else miss ".venv missing (run: uv sync)"; fi

echo "== Boltz-2 (needed for every fold) =="
CACHE="${NISEGRAD_BOLTZ_CACHE:-$HOME/.boltz}"
for f in boltz2_conf.ckpt boltz2_aff.ckpt; do
  if [ -s "$CACHE/$f" ]; then pass "$CACHE/$f"
  else miss "$CACHE/$f absent"; reach "https://huggingface.co/boltz-community/boltz-2/resolve/main/$f"; fi
done
if [ -d "$CACHE/mols" ]; then pass "$CACHE/mols/ (CCD)"
else miss "$CACHE/mols/ absent (untar mols.tar here)"; reach "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"; fi

echo "== LigandMPNN (only for rescue/projector/guided) =="
if [ -n "${LIGANDMPNN_CKPT:-}" ] && [ -s "${LIGANDMPNN_CKPT:-/nonexistent}" ]; then pass "LIGANDMPNN_CKPT=$LIGANDMPNN_CKPT"
else note "LIGANDMPNN_CKPT unset/absent (run scripts/get_weights.sh; not needed for the core panel)"; fi

echo "== Protenix-v2 judge (scores every design; separate venv, Python 3.11) =="
if [ -n "${PROTENIX_DIR:-}" ] && [ -x "${PROTENIX_DIR:-/nonexistent}/.venv/bin/protenix" ]; then
  pass "PROTENIX_DIR=$PROTENIX_DIR"
  pv=$(pyver "$PROTENIX_DIR/.venv/bin/python")
  [ "$pv" = "3.11" ] && pass "Protenix venv Python $pv" || miss "Protenix venv Python $pv (want 3.11; cp311 fused CUDA kernels do not import under other versions)"
else note "PROTENIX_DIR unset or .venv/bin/protenix missing (a ByteDance Protenix checkout with its own 3.11 venv)"; fi
reach "https://huggingface.co/TMF001/protenix-v2-weights/resolve/main/protenix-v2.pt"

echo "== data-only tiers (optional; scored outputs already committed) =="
if [ -n "${CHEM_FAP_DIR:-}" ] && [ -d "${CHEM_FAP_DIR:-/nonexistent}" ]; then pass "CHEM_FAP_DIR=$CHEM_FAP_DIR (RFd3 arm)"
else note "CHEM_FAP_DIR unset (RFd3 backbone arm; skip it, scores are in scripts/data/rfd3_panel/)"; fi
if [ -n "${RF2AA_DIR:-}" ] && [ -d "${RF2AA_DIR:-/nonexistent}" ]; then pass "RF2AA_DIR=$RF2AA_DIR (independence judge)"
else note "RF2AA_DIR unset (RF2AA judge; skip it, scores are in scripts/data/rf2aa_scored.json)"; fi

echo
printf "%d ok, %d need attention. The data-only tiers can be skipped; their scored outputs are committed.\n" "$ok" "$need"
