#!/usr/bin/env bash
# Laptop-side auto-destroy (option A). Keeps your VAST API key OFF the rented box.
# Polls the results branch for the DONE marker, pulls results, then destroys the
# instance. Run this on your own machine after launching the vast instance.
#
#   export VAST_API_KEY=...           # stays on your laptop only
#   ./scripts/destroy_watcher.sh <INSTANCE_ID> [POLL_SECONDS]
set -euo pipefail

INSTANCE_ID="${1:?usage: destroy_watcher.sh <INSTANCE_ID> [poll_seconds]}"
POLL="${2:-120}"
REPO="${REPO:-https://github.com/ericjtyao05-cmd/LVLM_OOD_Detection.git}"
: "${VAST_API_KEY:?export VAST_API_KEY (never put this on the rented box)}"

command -v vastai >/dev/null || pip install --quiet vastai
vastai set api-key "$VAST_API_KEY" >/dev/null

echo "[watch] waiting for DONE on results branch of $REPO (poll ${POLL}s)"
while true; do
  if git ls-remote "$REPO" refs/heads/results | grep -q .; then
    if git archive --remote="$REPO" results results/DONE >/dev/null 2>&1; then
      echo "[watch] DONE detected."
      break
    fi
  fi
  sleep "$POLL"
done

echo "[watch] pulling results into ./results-download/"
tmp=$(mktemp -d)
git clone --branch results --depth 1 "$REPO" "$tmp/r" >/dev/null 2>&1 || true
mkdir -p results-download && cp -r "$tmp/r/results/." results-download/ 2>/dev/null || true

echo "[watch] destroying instance $INSTANCE_ID"
vastai destroy instance "$INSTANCE_ID"
echo "[watch] done. billing stopped."
