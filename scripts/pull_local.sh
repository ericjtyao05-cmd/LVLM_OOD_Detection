#!/usr/bin/env bash
# Laptop-side: fetch a LOCAL copy of the private dataset (from HF) and the
# results (from the git `results` branch). Because the HF dataset is private and
# can't be shared directly, this gives you a local copy to keep / hand to a
# collaborator. Run from the repo root.
#
#   HF_TOKEN must be available (in ./.env or the environment).
#   ./scripts/pull_local.sh [config]
set -euo pipefail

CONFIG="${1:-configs/experiment.yaml}"
[ -f .env ] && { set -a; . ./.env; set +a; }
: "${HF_TOKEN:?export HF_TOKEN (or put it in ./.env) to read the private dataset}"

echo "[pull] dataset snapshot from HF -> ./data"
python -m src.dataset_hub pull --config "$CONFIG" --dest .

echo "[pull] results from git 'results' branch -> ./results"
git fetch -q origin results
mkdir -p results
git archive origin/results results 2>/dev/null | tar -x -f - 2>/dev/null \
  || echo "[pull] (no results branch yet)"

echo "[pull] done -> ./data  ./results"
