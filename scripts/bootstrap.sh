#!/usr/bin/env bash
# One-shot unattended bootstrap for a freshly rented vast.ai GPU.
# Paste into the vast "onstart" field. Secrets come from /workspace/.env
# (HF_TOKEN, GITHUB_TOKEN) -- see .env.example. Nothing secret is committed.
#
# Data flow (set by `data.reuse` in the config):
#   reuse=false : build data (prepare_data + generate_fakes) -> PUSH to private HF
#   reuse=true  : PULL the prebuilt snapshot from HF (byte-exact via hf.revision)
# Results (raw + paper-ready) are always PUSHED to the git `results` branch.
# Teardown is off-box: scripts/destroy_watcher.sh (keeps the vast key off here).
set -euo pipefail

: "${GITHUB_TOKEN:?}"; : "${HF_TOKEN:?}"
REPO_URL="${REPO_URL:-https://github.com/ericjtyao05-cmd/LVLM_OOD_Detection.git}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
WORK="${WORK:-/workspace}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export HF_HOME="$WORK/hf" HF_HUB_DOWNLOAD_TIMEOUT=120 PYTHONUNBUFFERED=1
mkdir -p "$WORK" "$HF_HOME"; cd "$WORK"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

AUTH_REMOTE="https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}"
[ -d repo/.git ] || git clone "$AUTH_REMOTE" repo
cd repo
git config user.email bot@vast; git config user.name vast-runner
ln -sf "$WORK/.env" .env 2>/dev/null || true

log "installing deps"; pip install -q -r requirements.txt
python -c "import torch; assert torch.cuda.is_available()" || { log "no CUDA"; exit 1; }

REUSE=$(python -c "import yaml;print(str(yaml.safe_load(open('$CONFIG'))['data'].get('reuse',False)).lower())")

# ---------------- data ----------------
if [ "$REUSE" = "true" ]; then
  log "reuse=true -> pulling dataset snapshot from HF"
  python -m src.dataset_hub pull --config "$CONFIG" --dest .
else
  log "reuse=false -> building dataset"
  python -m src.prepare_data --config "$CONFIG"
  # Contaminant pool: category-aligned fakes only (injected into id_train;
  # test sets always stay clean real -- design guardrail).
  python src/generate_fakes.py aligned --n-per-class 240 --batch-size 8 --out data/fake_id \
    --classes tabby_cat labrador_retriever goldfish bald_eagle african_elephant \
              zebra tiger brown_bear ostrich sports_car school_bus airliner \
              mountain_bike grand_piano steam_locomotive || log "WARN aligned fakes skipped"
  log "pushing dataset snapshot to HF (private)"
  python -m src.dataset_hub push --config "$CONFIG"    # prints revision to pin
fi

# ------------- build + run -------------
python -m src.build_from_config --config "$CONFIG" --out-dir manifests
python -m src.run_all  --config "$CONFIG" --manifests manifests --results "results/$RUN_ID" \
  2>&1 | tee "results/$RUN_ID.log" || true
mkdir -p "results/$RUN_ID"; mv "results/$RUN_ID.log" "results/$RUN_ID/run.log" 2>/dev/null || true
python -m src.paperize --results "results/$RUN_ID" --out "results/$RUN_ID/paper"

# ------------- push results to git -------------
log "pushing results + manifests to git results branch"
git fetch origin results 2>/dev/null && git checkout results || git checkout --orphan results
git add -f results/ manifests/ configs/ && git commit -m "results: run $RUN_ID" || log "nothing to commit"
echo "$RUN_ID" > results/DONE && git add -f results/DONE && git commit -m "DONE $RUN_ID"
git push "$AUTH_REMOTE" HEAD:results
log "DONE $RUN_ID -- results on GitHub; laptop watcher will destroy this box"
