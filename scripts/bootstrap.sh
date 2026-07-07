#!/usr/bin/env bash
# One-shot unattended bootstrap for a freshly rented vast.ai GPU (4090).
# Paste this (or a curl to it) into the vast instance "onstart" field, and set
# the env vars GITHUB_TOKEN / HF_TOKEN / REPO_URL there too. Secrets stay in the
# instance environment; nothing is written into the git repo.
#
# Flow:  env -> clone -> deps -> data -> build conditions -> run -> upload -> DONE
# Auto-destroy is handled OFF-box by scripts/destroy_watcher.sh (option A): this
# script only pushes a DONE marker; your laptop watcher destroys the instance.
set -euo pipefail

# ---- required env (injected via vast onstart, NEVER committed) ---------------
: "${GITHUB_TOKEN:?set GITHUB_TOKEN (fine-grained PAT, contents:rw on the repo)}"
: "${HF_TOKEN:?set HF_TOKEN (HuggingFace token with ImageNet-1k access)}"
REPO_URL="${REPO_URL:-https://github.com/ericjtyao05-cmd/LVLM_OOD_Detection.git}"
WORK="${WORK:-/workspace}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG="${CONFIG:-configs/experiment.yaml}"

export HF_HOME="$WORK/hf"
export HF_TOKEN
mkdir -p "$WORK" "$HF_HOME"
cd "$WORK"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ---- clone code (token only used in-memory for the authed remote) -----------
AUTH_REMOTE="https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}"
if [ ! -d repo/.git ]; then
  git clone "$AUTH_REMOTE" repo
fi
cd repo
git config user.email "bot@vast"; git config user.name "vast-runner"

# ---- system + python deps ---------------------------------------------------
log "installing deps"
apt-get update -y && apt-get install -y git tmux rsync jq >/dev/null 2>&1 || true
pip install -q -r requirements.txt
pip install -q "transformers>=4.43" accelerate bitsandbytes sentencepiece \
               protobuf scikit-learn matplotlib pyyaml huggingface_hub

python - <<'PY'
import torch; assert torch.cuda.is_available(), "no CUDA!"
print("[env] GPU:", torch.cuda.get_device_name(0))
PY

# ---- pull datasets (ID gated via HF_TOKEN) ----------------------------------
log "pulling datasets"
python -m src.prepare_data --config "$CONFIG"     # ID(ImageNet)+OOD(DTD)+WHOOPS
python src/generate_fakes.py aligned  --classes tabby_cat labrador_retriever goldfish \
        bald_eagle african_elephant zebra tiger brown_bear ostrich sports_car \
        school_bus airliner mountain_bike grand_piano steam_locomotive \
        --out data/fake_id || log "WARN: aligned fake gen skipped"
python src/generate_fakes.py freeform --n 500 --out data/fake_ood || log "WARN: freeform skipped"

# ---- build conditions -------------------------------------------------------
log "building manifests"
python -m src.build_from_config --config "$CONFIG" --out-dir manifests

# ---- run every condition + write results ------------------------------------
log "running experiments"
python -m src.run_all --config "$CONFIG" --manifests manifests \
       --results "results/$RUN_ID" 2>&1 | tee "results_${RUN_ID}.log"

# ---- paper-ready artifacts --------------------------------------------------
python -m src.paperize --results "results/$RUN_ID" --out "results/$RUN_ID/paper"

# ---- upload to the results branch -------------------------------------------
log "uploading results"
cp "results_${RUN_ID}.log" "results/$RUN_ID/run.log"
git fetch origin results || true
git checkout results 2>/dev/null || git checkout --orphan results
git add results/ manifests/ configs/experiment.yaml
git commit -m "results: run $RUN_ID" || log "nothing to commit"
echo "$RUN_ID" > results/DONE
git add results/DONE && git commit -m "DONE $RUN_ID"
git push "$AUTH_REMOTE" HEAD:results

log "DONE $RUN_ID -- laptop watcher will destroy this instance"
