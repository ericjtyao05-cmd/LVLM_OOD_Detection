# LVLM OOD Detection vs. *Fake* (physically-impossible) Images

Do LVLM-based OOD detectors flag **physically-impossible** images
(a cat with wings, Godzilla, an upward waterfall) — or is impossibility a blind
spot? We read a **large vision-language model's hidden states** on ID/OOD/fake
images and measure how post-hoc detectors (MSP / Energy / **Mahalanobis**)
respond as fakes are mixed in at controlled ratios.

- **Model:** LLaVA-1.5-7B (open LVLM, hidden states via HF `transformers`, fits a 4090).
- **Detectors:** MSP (baseline) + Energy + Mahalanobis on the last-token hidden
  state / class-restricted logits — swappable via `configs/experiment.yaml`.

Full write-up: **[docs/DESIGN.md](docs/DESIGN.md)**.
GPU walkthrough: **[docs/vast_ai_guide.md](docs/vast_ai_guide.md)**.

## Configuration — two files, one place each
- **`configs/experiment.yaml`** — every *non-secret* knob: `model`, `detector`
  methods, `data` (+ `reuse`), condition grid, `hf.dataset_repo`, results `upload`.
  Swap the model or detector by editing one line.
- **`.env`** (git-ignored; template in **`.env.example`**) — every *secret*:
  `HF_TOKEN`, `GITHUB_TOKEN`, `VAST_API_KEY`. All code reads secrets only from here.

## Artifact split (right tool per artifact)
| Artifact | Home | Why |
|---|---|---|
| Code, **manifests** (splits), configs | this git repo | small, diff-able, versioned |
| **Images** (`data/`) | **private HF dataset** (`hf.dataset_repo`) | large + ImageNet license → not in git |
| **Results** (json/tables/heatmaps) | git **`results` branch** | small, want them in the repo |

## Layout
```
configs/experiment.yaml    # non-secret single source of truth (+ experiment.smoke.yaml)
.env.example               # secrets template -> copy to .env
src/
  registry.py              # string-keyed MODEL_REGISTRY / DETECTOR_REGISTRY (swap = 1 line)
  config.py  metrics.py    # yaml loader; AUROC/FPR95 (rank-based)
  prepare_data.py          # ImageNet-1k(gated) + DTD + WHOOPS -> data/
  generate_fakes.py        # SDXL: category-aligned + freeform fakes (GPU)
  build_datasets.py / build_from_config.py   # mixing pipeline -> CSV manifests
  dataset_hub.py           # push/pull the private HF image snapshot
  models/llava.py          # @register_model: LLaVA-1.5-7B -> {hidden, logits}
  detectors.py             # @register_detector: msp / energy / mahalanobis
  run_all.py  paperize.py  # score conditions -> results + LaTeX/heatmaps
scripts/
  bootstrap.sh             # one-shot unattended run on a fresh vast.ai box
  pull_local.sh            # laptop: pull dataset (HF) + results (git) locally
  destroy_watcher.sh       # laptop-side auto-destroy (keeps VAST key off the box)
docs/DESIGN.md, docs/vast_ai_guide.md
```

## Workflow
**First run** (builds data): `bootstrap.sh` → build ImageNet+DTD+WHOOPS+SDXL →
**push images to private HF** → build manifests → run → **push results to git**.
Pin the printed HF revision in `hf.revision` for exact reruns.
**Later runs** (`data.reuse: true`): **pull the HF snapshot** (byte-exact) →
build → run → push results. No rebuild, identical data.
**Local copies**: `scripts/pull_local.sh` pulls the private dataset + results to
your laptop (the HF dataset is private, so this is how you keep/share a copy).

Set `GITHUB_TOKEN` + `HF_TOKEN` in the vast **onstart** field and the box self-runs
env → data → build → conditions → results → DONE. See
[docs/vast_ai_guide.md](docs/vast_ai_guide.md).

Baseline method = **MSP** (Hendrycks & Gimpel 2017); headline hidden-state method
= **Mahalanobis** (Lee et al. 2018). For the full ~40-method zoo, cross-reference
[OpenOOD](https://github.com/Jingkang50/OpenOOD).
