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

## Layout
```
configs/
  experiment.yaml        # single source of truth: model, detectors, data, grid, upload
  datasets.example.json  # low-level mixing-pipeline config
src/
  registry.py            # string-keyed MODEL_REGISTRY / DETECTOR_REGISTRY (swap = 1 line)
  config.py  metrics.py  # yaml loader; AUROC/FPR95 (rank-based)
  prepare_data.py        # ImageNet-1k(gated) + DTD + WHOOPS -> data/
  generate_fakes.py      # SDXL: category-aligned + freeform fakes (GPU)
  build_datasets.py      # mixing pipeline core -> CSV manifests
  build_from_config.py   # experiment.yaml -> mixing pipeline
  models/llava.py        # @register_model: LLaVA-1.5-7B -> {hidden, logits}
  detectors.py           # @register_detector: msp / energy / mahalanobis
  run_all.py             # loop conditions -> score -> results/<run>/*.json
  paperize.py            # results -> CSV + LaTeX table + AUROC heatmaps
scripts/
  bootstrap.sh           # one-shot unattended run on a fresh vast.ai box
  destroy_watcher.sh     # laptop-side auto-destroy (keeps VAST key off the box)
docs/DESIGN.md, docs/vast_ai_guide.md
```

## Data layout (built automatically by the pipeline)
```
data/
  id_real/<class>/*.jpg   # real ID (ImageNet-1k, 15 classes — see experiment.yaml)
  ood_real/*.jpg          # real OOD (Textures/DTD)
  fake_id/<class>/*.jpg   # category-aligned fakes (generate_fakes.py aligned)
  fake_ood/*.jpg          # WHOOPS! + freeform fakes
```

## Automated run (fresh server)
Set `GITHUB_TOKEN` + `HF_TOKEN` in the vast **onstart** field, then it self-runs:
env → data → build → all conditions → upload results → DONE. See
[docs/vast_ai_guide.md](docs/vast_ai_guide.md). Locally you can still run stages
by hand (`build_datasets.py`, etc.).

Baseline method = **MSP** (Hendrycks & Gimpel 2017); headline hidden-state method
= **Mahalanobis** (Lee et al. 2018). For the full ~40-method zoo, cross-reference
[OpenOOD](https://github.com/Jingkang50/OpenOOD).
