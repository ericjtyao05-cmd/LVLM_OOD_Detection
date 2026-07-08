# Onboarding guide — get running on a rented GPU in ~30 min

For a new collaborator joining the LVLM-OOD-vs-fake project. The happy path
**reuses the prebuilt dataset** (no rebuild), so you go from zero to results fast.
Deeper vast.ai detail lives in [vast_ai_guide.md](vast_ai_guide.md).

> **⚙️ One-time, done by the repo owner (Eric):** add the collaborator's
> HuggingFace account as a **read** member of the private dataset
> `EricY05/lvlm-ood-fake-data` (HF → dataset → Settings → Members). Without this,
> the collaborator's token can't pull the images.

## 0. What you need (accounts + tokens)
| Thing | Where | Used for |
|---|---|---|
| **GitHub access** to the code repo | ericjtyao05-cmd/LVLM_OOD_Detection | clone code; (optional) push results |
| **HuggingFace account** (added as collaborator above) + **read token** | huggingface.co/settings/tokens | pull the private dataset snapshot |
| **GitHub fine-grained PAT** (Contents: R/W) — optional | github.com/settings/tokens | push your results to the `results` branch |
| **vast.ai account** + **API key** | vast.ai → Account | rent the GPU; auto-destroy |
| **SSH keypair** | your laptop | connect to the box |

You do **not** need ImageNet access — reuse mode pulls the already-built images from HF.

## 1. SSH key (one time)
```bash
ssh-keygen -t ed25519 -C "vast" -f ~/.ssh/vast_ed25519      # if you don't have one
# add the PUBLIC key to vast: Account → SSH Keys → paste ~/.ssh/vast_ed25519.pub
ssh-add --apple-use-keychain ~/.ssh/vast_ed25519            # macOS; loads the (passphrased) key into the agent
```
> **Gotcha we hit:** if your key has a passphrase, load it into the ssh-agent (`ssh-add`)
> *before* connecting. A locked key gives `Server accepts key … Permission denied`.

## 2. Rent the box
- vast.ai → **Create**. **GPU: 1× RTX 4090 (24 GB)** is plenty.
- **Template: an official PyTorch/CUDA image** (e.g. `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`).
  **Do NOT use the NVIDIA NGC image** (`nvcr.io/nvidia/pytorch`) — its entrypoint breaks vast's
  SSH provisioning (we lost an instance to this).
- **Disk ≥ 80 GB**, reliability > 0.98, decent Inet-down. Enable **Direct SSH**.
- Connect (host/port from the console's Connect panel):
  ```bash
  ssh -i ~/.ssh/vast_ed25519 -p <PORT> root@<HOST>
  ```

## 3. Clone code + install deps (on the box, inside tmux)
```bash
tmux new -s ood
cd /workspace && git clone https://github.com/ericjtyao05-cmd/LVLM_OOD_Detection.git
cd LVLM_OOD_Detection && pip install -r requirements.txt
python -c "import torch;print('CUDA', torch.cuda.is_available())"     # expect True
```

## 4. Secrets → one file on the box
Never paste tokens into chat/logs. From **your laptop**, pipe them straight to the box
(`.env` is git-ignored; template is `.env.example`):
```bash
ssh -p <PORT> root@<HOST> 'umask 077; cat > /workspace/.env' <<'EOF'
export HF_TOKEN=hf_your_read_token
export GITHUB_TOKEN=github_pat_your_token   # only if you'll push results
EOF
```

## 5. Get the data — reuse the pinned snapshot (no rebuild)
`configs/experiment.yaml` already has `data.reuse: true`-ready settings and the pinned
`hf.revision`. Pull the exact images everyone else used:
```bash
cd /workspace/LVLM_OOD_Detection && set -a && . /workspace/.env && set +a
python -m src.dataset_hub pull --config configs/experiment.yaml --dest .
# -> recreates ./data/ (4500 ID + 3000 OOD + 1200 fakes) at revision 05cfa5e…
```

## 6. Run
```bash
set -a && . /workspace/.env && set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.build_from_config --config configs/experiment.yaml --out-dir manifests
python -m src.run_all  --config configs/experiment.yaml --manifests manifests --results results/mine
python -m src.paperize --results results/mine --out results/mine/paper
```
Watch it work: `watch -n1 nvidia-smi` (GPU should sit near 100% during extraction).
Results: `results/mine/summary.json` + `results/mine/paper/` (table + heatmaps).

**Or fully unattended:** set `HF_TOKEN`+`GITHUB_TOKEN` in the box's onstart field and let
`scripts/bootstrap.sh` do env → pull → build → run → push results in one shot.

## 7. Get results off the box, then destroy
```bash
# from your laptop
rsync -a -e "ssh -i ~/.ssh/vast_ed25519 -p <PORT>" root@<HOST>:/workspace/LVLM_OOD_Detection/results/mine/ ./results/mine/
```
Then **Destroy** the instance in the vast console (billing stops only on Destroy, not Stop). Or use
the laptop-side watcher (keeps your vast key off the box):
```bash
export VAST_API_KEY=...   # laptop only
./scripts/destroy_watcher.sh <INSTANCE_ID>
```

## 8. Change what you run (no pipeline edits)
Everything swappable is in `configs/experiment.yaml`:
- **Detectors:** `detector.methods: [msp, energy, mahalanobis]` — add/remove a name.
- **Conditions:** `conditions.id_fake_ratios` / `ood_fake_ratios`.
- **Model:** `model.name` (a second LVLM registers in `src/models/` and is selected by its key).

## Common snags (all pre-solved in the code, listed so you recognise them)
- **`Permission denied (publickey)`** → key not in agent (`ssh-add`) or not on your vast account.
- **NGC image** → SSH won't provision; rent an official PyTorch template.
- **CUDA OOM** → keep `batch_size: 8`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **DataLoader hang / "too many open files"** → already handled (`file_system` sharing strategy).
- **SDXL import error** → only relevant if you *rebuild* fakes; use the isolated venv recipe in
  `scripts/` (reuse mode skips fake generation entirely).
