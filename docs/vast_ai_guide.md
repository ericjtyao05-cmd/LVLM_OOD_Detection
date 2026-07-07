# Hands-on: running this project on a rented vast.ai GPU

A practical, copy-pasteable walkthrough: pick a box → connect → build the env →
move data → run → pull results → **destroy** (so you stop paying).

## 0. What GPU to rent (for this project)

Phase-1 baseline (ViT-B/16 feature extraction + linear probe) is light:
* **1× RTX 3090 / 4090 (24 GB)** is plenty and cheap (~$0.20–0.40/hr).
* SDXL fake generation also fits comfortably on a 24 GB card.
* You do **not** need an A100/H100 for the baseline. Rent up only if you move to
  full ViT fine-tuning or LVLM judging.

Filter tips on the web console:
* Sort by **$/hr**; require **On-demand** (not Interruptible) for stability while
  learning; Interruptible is cheaper once you checkpoint.
* **Reliability > 0.98**, **Inet down > 200 Mbps** (dataset pulls), **Disk ≥ 60 GB**.
* Pick an image with CUDA preinstalled — the template
  **`pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime`** or vast's "PyTorch" template.

## 1. Account + SSH key (one time)

1. Sign up at vast.ai, add credit.
2. Create a key locally if you don't have one:
   ```bash
   ssh-keygen -t ed25519 -C "vast" -f ~/.ssh/vast_ed25519
   ```
3. Console → **Account → SSH Keys** → paste `~/.ssh/vast_ed25519.pub`.

Optional CLI (nice for scripting; the `vast-gpu` skill wraps this):
```bash
pip install vastai
vastai set api-key <YOUR_API_KEY>          # from Account page
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 rentable=true' -o 'dph+'
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime \
       --disk 60 --ssh
```

## 2. Rent + connect

Web console → **Create** on an offer → set the **PyTorch** template and **Disk**
(≥60 GB) → **Rent**. When it shows **running**, open **Instances**, click the
`>_` (or "Connect") to get the SSH command, e.g.:

```bash
ssh -i ~/.ssh/vast_ed25519 -p 41234 root@ssh5.vast.ai
```

* The `-p <port>` and host come from the console — they differ per instance.
* First connection: accept the host key.
* If it hangs, the instance may still be pulling the image — wait 1–2 min.

Add a `~/.ssh/config` block so you can just `ssh vast`:
```
Host vast
    HostName ssh5.vast.ai
    Port 41234
    User root
    IdentityFile ~/.ssh/vast_ed25519
    ServerAliveInterval 30
```

## 3. Build the environment

Always work inside **tmux** so a dropped connection doesn't kill your run:
```bash
tmux new -s ood            # later: tmux attach -t ood ; detach with Ctrl-b then d
```

Then:
```bash
# The PyTorch image already has torch+CUDA. Verify:
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# get the code
apt-get update && apt-get install -y git tmux rsync
git clone <your-repo-url> LVLM_OOD_Detection && cd LVLM_OOD_Detection
# (or rsync it up from your laptop — see §4)

# install project deps on top of the image's torch
pip install -r requirements.txt
```

If the image lacks torch (bare Ubuntu template), install the CUDA build:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 4. Move data & code

**Push code/data up** (from your laptop; note the port):
```bash
rsync -avP -e "ssh -i ~/.ssh/vast_ed25519 -p 41234" \
      ./LVLM_OOD_Detection/ root@ssh5.vast.ai:/root/LVLM_OOD_Detection/ \
      --exclude data --exclude manifests
```
Big datasets: prefer pulling **on the box** (fast datacenter link) rather than
uploading from home:
```bash
# on the instance
python src/prepare_sources.py whoops --out data/fake_ood --limit 500   # HF download
# ImageNet-200 / OOD sets: follow OpenOOD's download scripts here
```
HuggingFace cache lives in `~/.cache/huggingface`; set `HF_HOME=/workspace/hf` if
you want it on the big disk.

## 5. Run

Fully automated (what `bootstrap.sh` runs): env → data → build → run → upload.
To drive a stage by hand instead:
```bash
python -m src.prepare_data       --config configs/experiment.yaml           # ImageNet+DTD+WHOOPS
python src/generate_fakes.py aligned  --classes tabby_cat labrador_retriever ... --out data/fake_id
python src/generate_fakes.py freeform --n 500 --out data/fake_ood
python -m src.build_from_config  --config configs/experiment.yaml --out-dir manifests
python -m src.run_all            --config configs/experiment.yaml --manifests manifests \
                                 --results results/manual
```
Watch the GPU in another pane (`Ctrl-b "` to split): `watch -n1 nvidia-smi`.

## 6. Pull results back, then DESTROY

```bash
# from laptop
rsync -avP -e "ssh -i ~/.ssh/vast_ed25519 -p 41234" \
      root@ssh5.vast.ai:/root/LVLM_OOD_Detection/manifests/ ./manifests/
```
Then in the console click **Destroy** on the instance (Stop only pauses billing
for compute but you still pay storage; **Destroy** ends all charges). Confirm the
instance is gone from **Instances**.

## 7. Cost & gotchas checklist

* **Always Destroy when done** — idle rented GPUs bill by the second.
* **tmux** everything; SSH drops are common.
* Persist to `/workspace` (large volume); `/root` may be small.
* If `torch.cuda.is_available()` is `False`, you rented a CPU offer or the driver
  mismatches the image — destroy and pick a proper PyTorch/CUDA template.
* Interruptible instances can be reclaimed anytime — only use once you checkpoint.
* Keep a `setup.sh` with §3 commands so re-provisioning a fresh box is one line.
```

> This repo also ships a `run-experiment` / `vast-gpu` skill integration — you can
> let the assistant rent, deploy, and destroy for you once the manual flow makes
> sense to you.
