#!/usr/bin/env bash
# SLURM batch template for the panel-wide real-backbone rescue on one H100.
# Edit the paths + the cluster directives (partition/account/gres name) for your site, then:
#   sbatch scripts/panel_rescue.slurm
#
#SBATCH --job-name=panel_rescue
#SBATCH --gres=gpu:h100:1          # one H100; some clusters use gpu:1 or a different type name
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00            # 14 ligands x 8 seeds; generous, an H100 finishes well inside this
#SBATCH --output=panel_rescue_%j.log
## --partition=gpu                 # <-- set your GPU partition
## --account=YOUR_ACCOUNT          # <-- set if your cluster meters by account
set -uo pipefail

# --- edit these for your cluster -------------------------------------------
REPO="$HOME/nise-grad"                                   # your nise-grad checkout
export PROTENIX_DIR="$HOME/Protenix"                     # ByteDance Protenix checkout (its venv MUST be Python 3.11)
export LIGANDMPNN_CKPT="$REPO/weights/ligandmpnn_v_32_010_25.pt"
export LIGMPNN_MODEL_DIR="$REPO/weights/jligandmpnn_reference"
# Boltz-2 weights are read from ~/.boltz (or wherever NISEGRAD_BOLTZ_CACHE points).
# module load cuda/12.4                                  # uncomment if your cluster uses modules
# ---------------------------------------------------------------------------

cd "$REPO"
# SLURM already restricts the job to its allocated GPU, which the job sees as index 0.
GPU=0 NSEEDS=8 LIG_START=0 LIG_END=14 bash scripts/panel_rescue.sh
