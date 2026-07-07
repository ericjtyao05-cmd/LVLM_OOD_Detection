#!/usr/bin/env bash
# Laptop-side auto-destroy. Keeps your VAST API key OFF the rented box.
# Polls the git `results` branch for the DONE marker, then destroys the instance.
# Results are already on GitHub (and pullable via scripts/pull_local.sh), so this
# does NOT download anything -- it only tears the box down.
#
#   export VAST_API_KEY=...            # stays on your laptop only
#   ./scripts/destroy_watcher.sh <INSTANCE_ID> [POLL_SECONDS]
set -euo pipefail

INSTANCE_ID="${1:?usage: destroy_watcher.sh <INSTANCE_ID> [poll_seconds]}"
POLL="${2:-120}"
REPO="${REPO:-https://github.com/ericjtyao05-cmd/LVLM_OOD_Detection.git}"
: "${VAST_API_KEY:?export VAST_API_KEY (never put this on the rented box)}"

command -v vastai >/dev/null || pip install --quiet vastai
vastai set api-key "$VAST_API_KEY" >/dev/null

echo "[watch] waiting for DONE on results branch (poll ${POLL}s)"
while true; do
  if git archive --remote="$REPO" results results/DONE >/dev/null 2>&1; then
    echo "[watch] DONE detected."; break
  fi
  sleep "$POLL"
done

echo "[watch] destroying instance $INSTANCE_ID (results are on GitHub)"
vastai destroy instance "$INSTANCE_ID"
echo "[watch] done. billing stopped. run scripts/pull_local.sh for local copies."
