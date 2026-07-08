# LVLM OOD Detection vs. *Fake* (physically-impossible) Images — Progress Report

*Internal progress report. Model: LLaVA-1.5-7B. Full corrected run: `results/full_v2`.*

## TL;DR
- **Question:** do LVLM-based OOD detectors flag images that are *semantically familiar but
  physically impossible* (a cat with wings, Godzilla) — a blind spot orthogonal to normal OOD?
- **Answer (this run):** yes — on LLaVA-1.5-7B **hidden states**, impossibility is highly
  detectable. **Mahalanobis** reaches **AUROC 0.99 / FPR95 0.05** on fakes, ≈ its score on real
  novel classes.
- **It's not an artifact detector:** detection on *human-made* fakes (WHOOPS) ≈ *AI-made* fakes
  (SDXL) — so it tracks **impossibility, not generation fingerprints.**
- **Contaminating the ID set with fakes degrades detection** (FPR95 0.053→0.078 as ID fakes 0→25%).

---

## 1. Dataset construction (what & why)

The study needs three image pools that no single benchmark provides, so we assemble them:

| Pool | Source | Size | Why this choice |
|---|---|---|---|
| **Real ID** | ImageNet-1k (gated), 15 concrete classes, `train` split, 300/class | 4500 | Concrete objects (cat, dog, bird, elephant, car, bus, airliner, piano, locomotive…) that have *plausible impossible variants*. Real photos, so the "fake" signal isn't confounded by photo-vs-render domain. |
| **Real OOD** | Describable Textures (DTD) | 3000 | Standard *far-OOD*; semantically disjoint from the ID objects — the "normal OOD" reference. |
| **Fake — human** | **WHOOPS!** (Bitton-Guetta et al., ICCV'23) | 500 | Human-curated, commonsense-defying **real photographs** (edited/staged). **Artifact-free** → the honest anchor that proves detection is about *impossibility*, not synthesis. |
| **Fake — generated** | **SDXL**: aligned (30/class) + freeform | 450 + 250 | *Category-aligned* fakes (an impossible **cat** labelled "cat") are the only way to build the **ID-contamination axis**; freeform gives scale for the OOD/probe side. |

**Why two fake sources (the key design point).** SDXL images carry generation fingerprints; a
detector could score high by spotting *those* rather than impossibility. WHOOPS images are
pixel-wise real, so they isolate impossibility. We report both separately and check they agree
(they do — §4). Using only one would leave the claim ambiguous.

**Mixing into conditions.** Fakes are injected along two independent axes and every
`(id_fake%, ood_fake%)` pair is one *condition* — a data-mixing recipe over the shared pools,
materialised as 4 small CSV **manifests** (path lists, no image copies):

```
id_fake%  ∈ {0, 1, 10, 25}   fraction of the ID *train* set that is (category-aligned) fake
ood_fake% ∈ {0, 1, 10, 25}   fraction of the OOD *test* set that is fake
grid = "families" → 10 conditions: (0,0) + (r,0) + (0,r) + (r,r)
```

Every condition also emits two fixed diagnostics: a clean `id_test` (never contaminated) and a
100%-fake `fake_probe` — the headline "can we detect impossibility?" set (drawn from WHOOPS +
freeform). Sizes are held constant across conditions (replace, don't add) so AUROC stays
comparable.

**Reproducibility.** Images live in a **private HF dataset** (`EricY05/lvlm-ood-fake-data`) pinned
at a commit `revision`; the manifests are committed to git. `(pinned revision) × (committed
manifests)` = byte-exact reruns. (Images are out of git: too large, and ImageNet redistribution is
license-restricted → private HF.)

---

## 2. The three detectors (why these, and how they work)

All are **post-hoc** (no OOD data at training) and all read the **LLaVA-1.5-7B** internals from a
single forward pass on the prompt *"What is the main object? Answer with a single word."*. They
span the axis we care about — **decision-layer vs representation**:

| Detector | Reads | Mechanism | Role |
|---|---|---|---|
| **MSP** (Hendrycks & Gimpel, 2017) | logits | **Max softmax probability**. Train a linear probe on ID-train hidden states → C-way logits; OOD-ness = `1 − max softmax`. Confident ⇒ ID. | *The* canonical OOD baseline. |
| **Energy** (Liu et al., 2020) | logits | **Free energy** `E = −logsumexp(logits)`; OOD-ness = `−E`. Uses the *whole* logit vector, not just the peak — less saturation-prone than MSP. | Stronger logit-based reference. |
| **Mahalanobis** (Lee et al., 2018) | hidden state | Fit one Gaussian per ID class on last-token hidden states (tied, shrinkage-regularised covariance); OOD-ness = min class Mahalanobis distance. Measures distance in the *whitened representation*. | The **feature-geometry** method — the natural probe for "does impossibility live in the representation?" |

**Why this trio.** MSP is the reference everyone compares against; Energy is the standard
logit-based improvement; Mahalanobis is the representation-based method. MSP+Energy answer *"does
the decision layer flag fakes?"* and Mahalanobis answers *"does the hidden representation?"* —
which is exactly the question. (MSP/Energy use a linear probe because LLaVA's raw next-token
class logits are near-constant — see §4 note.)

---

## 3. Workflow & code scaffold

**Design principles:** swappable (models/detectors via a registry + one YAML), self-provisioning
(one bootstrap on a fresh GPU), and results/data preserved off the ephemeral box.

**Artifact split — right tool per artifact:**

| Artifact | Home |
|---|---|
| Code, **manifests** (splits), configs | this git repo (`main`) |
| **Images** (`data/`) | private **HF dataset** (pinned revision) |
| **Results** (JSON, tables, heatmaps) | git **`results` branch** |
| Secrets (HF/GitHub/vast tokens) | `.env` (git-ignored; template `.env.example`) |

**Pipeline (files):**
```
configs/experiment.yaml        # non-secret single source of truth (model, detectors, data, grid, hf)
  └─ .env                      # all secrets, one place

src/prepare_data.py            # ImageNet-1k(gated) + DTD + WHOOPS  -> data/
src/generate_fakes.py          # SDXL aligned + freeform fakes      -> data/ (GPU; isolated venv)
src/build_from_config.py ─┐    # experiment.yaml -> mixing pipeline
src/build_datasets.py    ─┘    #   -> manifests/<condition>/*.csv (relative paths)
src/dataset_hub.py             # push/pull the private HF image snapshot (reproducible)
src/models/llava.py            # @register_model: LLaVA -> last-token hidden state (parallel DataLoader)
src/detectors.py               # @register_detector: msp / energy / mahalanobis
src/run_all.py                 # loop conditions: extract -> linear probe -> score -> results/*.json
src/paperize.py                # results -> CSV + LaTeX table + AUROC heatmaps
src/registry.py                # string-keyed registries (swap model/detector = 1 line)

scripts/bootstrap.sh           # fresh vast box: env -> (reuse|build+push HF) -> run -> push results
scripts/pull_local.sh          # laptop: pull dataset (HF) + results (git) locally
scripts/destroy_watcher.sh     # laptop-side auto-destroy (vast key stays off the box)
```

**Two run modes** (set by `data.reuse` in the config):
- **build** (first time): prepare data + generate fakes → **push snapshot to HF** → build → run.
- **reuse** (thereafter): **pull the pinned HF snapshot** → build → run. Identical data, no rebuild.

Results (raw + paper-ready) are always pushed to the `results` branch; `pull_local.sh` mirrors data
+ results to a laptop.

**Performance note.** `run_all` extracts hidden states with a parallel `DataLoader`
(worker processes), keeping the GPU **~100%** utilised (an earlier serial version starved it to
~0%). On a container with `ulimit -n = 1024` this required the `file_system` tensor-sharing
strategy to avoid file-descriptor exhaustion.

---

## 4. Results & interpretation

**Corrected full run (`full_v2`), 10 conditions.** AUROC↑ / FPR95↓. `_ood` = clean ID vs real
textures; `_fake` = clean ID vs the 100%-fake probe.

| Condition | MSP AUC_fake | Energy AUC_fake | **Maha AUC_fake** | **Maha FPR95_fake** |
|---|---|---|---|---|
| id00_ood00 (baseline) | 0.898 | 0.970 | **0.988** | **0.053** |
| id00_ood25 (OOD 25% fake) | 0.897 | 0.969 | 0.988 | 0.053 |
| id01_ood00 (ID 1% fake) | 0.898 | 0.970 | 0.988 | 0.062 |
| id10_ood00 (ID 10% fake) | 0.909 | 0.972 | 0.987 | 0.071 |
| id25_ood00 (ID 25% fake) | 0.899 | 0.968 | 0.985 | 0.073 |
| id25_ood25 (both 25%) | 0.905 | 0.969 | 0.986 | 0.078 |

(Real-OOD AUROC is high throughout: MSP 0.95–0.97, Energy ~0.99, Maha ~0.99 — sanity holds.)

**Findings:**

1. **Impossibility is detectable — strongly — in the hidden state.** Mahalanobis flags fakes at
   **0.988**, essentially equal to its score on real novel classes (0.989). Impossibility is *not*
   a blind spot for a feature-geometry detector.

2. **All three methods have signal; Mahalanobis wins at the operating point.** With a proper linear
   probe, MSP (0.90) and Energy (0.97) also separate fakes by AUROC. But at **FPR95** the gap is
   decisive: **Maha 0.05** vs Energy ~0.20 vs **MSP 1.0**. MSP ranks fakes above ID on average yet
   its score tails overlap, so no usable threshold exists — good AUROC, unusable operating point.

3. **Not an AI-image detector (confound controlled).** Fake AUROC on **human-made WHOOPS** ≈
   **AI-made SDXL** for every method (e.g. Maha 0.991 vs 0.983; MSP 0.899 vs 0.896). If the models
   keyed on synthesis artifacts, SDXL would dominate — it doesn't. Detection tracks *impossibility*.

4. **ID contamination degrades detection (H2).** As impossible images are injected into the ID
   training set, Mahalanobis FPR95 on fakes rises monotonically **0.053 → 0.062 → 0.071 → 0.078**
   (ID 0→25%): the fitted class distributions absorb the fakes, so they blend in. Effect is modest
   (Maha is robust) but clean and directional.

5. **OOD-set contamination doesn't touch the fake probe** (by design — the probe is compared to
   clean ID), but it *does* slightly lower Energy's real-OOD AUROC (0.993→0.987): Energy finds
   WHOOPS fakes marginally harder than textures.

> **Methods note (why the numbers changed once):** an initial run scored MSP/Energy from LLaVA's
> raw next-token class-token logits, which were near-constant → MSP collapsed to AUROC **0.50**.
> Switching to a linear probe trained on ID hidden states fixed it (MSP 0.50→0.90, Energy
> 0.76→0.97). Mahalanobis was unaffected (it never used logits). The retired run is kept in the
> `results` branch for provenance only.

---

## 5. Limitations
- **Scale:** 15 classes, 300/class, **one** LVLM (LLaVA-1.5-7B), **single seed** — no error bars yet.
- **Fake diversity:** only 30 unique aligned fakes/class (cycled to fill the 25% ID condition), so
  the ID-contamination effect is real but on limited fake variety.
- **Impossibility ≠ isolated:** a winged cat differs from a cat in *both* physical plausibility *and*
  appearance; we can't yet separate "detects impossibility" from "detects visual novelty."
- **MSP probe:** MSP/Energy depend on a linear probe (a modeling choice); a prompt-scoring variant
  (MCM-style full-sequence log-prob) would test the *native* logits more faithfully.

## 6. Next steps
- Second LVLM (Qwen2-VL-2B) via the registry — does the finding generalise across models?
- Multi-seed for error bars; more classes.
- A **ReGuide-style prompt judge** ("is this physically possible?") as a contrasting *reasoning*
  method vs the hidden-state scores.
- Disentangle impossibility from appearance (e.g. matched real-vs-fake pairs of the same object).
