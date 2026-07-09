# Does fake-image contamination of training data hurt LVLM OOD detection?

Training data increasingly contains AI-generated, physically-impossible images
(a cat with wings, Godzilla). We contaminate an OOD detector's **ID training
set** with such fakes at controlled ratios and measure whether it gets worse at
its actual job: separating **real ID** from **real OOD** images.

**Design guardrail:** contamination happens **only in training**. The test sets
— clean real `id_test` vs clean real `ood_test` — are identical across all
conditions (enforced by an assertion in `run_all`). There is deliberately no
fake-containing test set: real-vs-fake separation is confounded (all fake pools,
including WHOOPS!, are AI-generated) and is not measured here.

- **Model:** LLaVA-1.5-7B (hidden states via HF `transformers`; fits a 4090).
- **Detectors:** MSP · Energy · Mahalanobis, post-hoc, fit only on the
  (contaminated) ID training set — swappable via `configs/experiment.yaml`.
- **Pilot result:** up to 25% contamination, real-vs-real OOD AUROC is flat
  (~0.97–0.99 for all three detectors). See **[docs/REPORT.md](docs/REPORT.md)**.

Design: **[docs/DESIGN.md](docs/DESIGN.md)** ·
GPU walkthrough: **[docs/vast_ai_guide.md](docs/vast_ai_guide.md)** ·
New collaborator: **[docs/onboarding_guide.md](docs/onboarding_guide.md)**

## Configuration — two files, one place each
- **`configs/experiment.yaml`** — every *non-secret* knob: model, detector
  methods, data (+ `reuse`), contamination ratios, `hf.dataset_repo`, upload.
- **`.env`** (git-ignored; template **`.env.example`**) — every *secret*:
  `HF_TOKEN`, `GITHUB_TOKEN`, `VAST_API_KEY`.

## Artifact split
| Artifact | Home | Why |
|---|---|---|
| Code, **manifests** (splits), configs | this git repo | small, diff-able, versioned |
| **Images** (`data/`) | **private HF dataset** (`hf.dataset_repo`, pinned revision) | large + ImageNet license |
| **Results** | git **`results` branch** | small, versioned with the code |

## Layout
```
configs/experiment.yaml    # non-secret single source of truth (+ experiment.smoke.yaml)
.env.example               # secrets template -> copy to .env
src/
  registry.py              # string-keyed MODEL_REGISTRY / DETECTOR_REGISTRY (swap = 1 line)
  config.py  metrics.py    # yaml loader; AUROC/FPR95 (rank-based)
  prepare_data.py          # ImageNet-1k(gated) + DTD -> data/
  generate_fakes.py        # SDXL category-aligned contaminants -> data/fake_id (GPU)
  build_datasets.py / build_from_config.py   # contamination sweep -> CSV manifests
  dataset_hub.py           # push/pull the private HF image snapshot
  models/llava.py          # @register_model: LLaVA-1.5-7B -> hidden states (parallel)
  detectors.py             # @register_detector: msp / energy / mahalanobis
  run_all.py  paperize.py  # fit on id_train -> score clean tests -> tables + sweep plots
scripts/
  bootstrap.sh             # one-shot unattended run on a fresh vast.ai box
  pull_local.sh            # laptop: pull dataset (HF) + results (git) locally
  destroy_watcher.sh       # laptop-side auto-destroy (keeps VAST key off the box)
docs/DESIGN.md, docs/REPORT.md, docs/vast_ai_guide.md, docs/onboarding_guide.md
```

## Workflow
**First run** (`data.reuse: false`): build ImageNet+DTD+SDXL contaminants →
**push images to private HF** (pin the printed revision) → build manifests →
run → **push results to git**.
**Later runs** (`data.reuse: true`): **pull the pinned HF snapshot** → build →
run → push results. Identical data, no rebuild.
**Local copies:** `scripts/pull_local.sh`.

Set `GITHUB_TOKEN` + `HF_TOKEN` in the vast **onstart** field and the box
self-runs end to end. See [docs/vast_ai_guide.md](docs/vast_ai_guide.md).
