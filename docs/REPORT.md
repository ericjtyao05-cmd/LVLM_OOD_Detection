# Fake-image contamination of OOD-detector training — Progress Report

*Internal progress report. Model: LLaVA-1.5-7B. Valid run: `results/contamination_v1`
(cleaned from run `full_v2`; see the retraction note in §5).*

## TL;DR
- **Question:** if fake (physically-impossible, AI-generated) images pollute the
  **ID training set**, does an LVLM-based OOD detector get worse at separating
  **real ID** from **real OOD** images? Test sets are always clean real images.
- **Answer (pilot):** essentially **no** — up to 25% contamination, real-vs-real
  OOD AUROC stays ~0.97–0.99 for all three detectors (MSP / Energy / Mahalanobis).
- **Why (likely):** post-hoc detectors are majority-driven robust estimators, and
  category-aligned fakes (a winged cat is still cat-shaped) barely move the
  fitted class statistics.
- **Honest scope:** one LVLM, 15 classes, single seed, ≤25%, post-hoc detectors
  only — a pilot signal, not a replicated result. An earlier "fake detection"
  framing was **retracted** as confounded (§5).

---

## 1. Dataset construction (what & why)

| Pool | Source | Size | Role |
|---|---|---|---|
| **Real ID** | ImageNet-1k, 15 concrete classes (cat, dog, eagle, elephant, zebra, sports car, school bus, airliner, grand piano, …), 300/class | 4500 | training (contaminated) + **clean test** |
| **Real OOD** | Describable Textures (DTD) | 3000 | **clean test** only |
| **Contaminants** | SDXL **category-aligned** impossible variants (winged / giant / transparent / two-headed / floating versions of each ID class) | 30/class | injected into `id_train` **only** |

**Why these choices:**
- *Concrete ImageNet classes* → each has plausible "impossible variants," and real
  photos keep the domain consistent.
- *Category-aligned contaminants* → the fake must credibly carry the class label
  (an impossible **cat** labelled "cat"). That is the realistic contamination
  scenario — polluted training data, not random noise.
- *Replace, don't add* → training-set size is constant across conditions, so any
  change is attributable to contamination, not set size.

**The design guardrail.** Contamination exists **only in training**. `id_test`
(900 clean real ID) and `ood_test` (3000 clean real OOD) are identical in every
condition. One condition = one contamination ratio ∈ {0, 1, 10, 25}% →
conditions `id00, id01, id10, id25`.

**Reproducibility.** Images live in a private HF dataset
(`EricY05/lvlm-ood-fake-data`, pinned revision `05cfa5e…`); the CSV manifests
defining every split are committed to git. Pinned snapshot × committed manifests
= byte-exact reruns.

---

## 2. The three detectors (why these, and how they work)

All are **post-hoc**: they fit statistics on the (possibly contaminated) ID
training set and never see OOD data. Contamination reaches them through those
fitted statistics — that is the causal path under test. All read LLaVA-1.5-7B's
last-token hidden state from one forward pass.

| Detector | Fits on id_train | Scores by | Why included |
|---|---|---|---|
| **MSP** (Hendrycks & Gimpel 2017) | linear probe (logistic head on hidden states) | max softmax probability — confident ⇒ ID | *The* canonical OOD baseline |
| **Energy** (Liu et al. 2020) | same probe | `−logsumexp(logits)` — uses the whole logit vector, less saturation-prone | Standard logit-based reference |
| **Mahalanobis** (Lee et al. 2018) | class means + tied shrinkage covariance | min class Mahalanobis distance in feature space | Representation-geometry method; directly exposes whether fakes shift the class statistics |

Together they cover both ways contamination could bite: corrupted **probe
weights** (MSP/Energy) and corrupted **class geometry** (Mahalanobis).

*Note:* the probe exists because LLaVA's raw next-token class logits are
near-constant — scoring MSP on them collapses to AUROC 0.5. This was found and
fixed during the project (see §5).

---

## 3. Workflow & code scaffold

Principles: **swappable** (models/detectors behind a string-keyed registry + one
YAML), **self-provisioning** (one bootstrap on a fresh rented GPU),
**preserved** (data on HF, results on git — nothing lives only on the box).

| Artifact | Home |
|---|---|
| Code · manifests · configs | git `main` |
| Images (`data/`) | private HF dataset, pinned revision |
| Results (JSON/tables/plots) | git `results` branch |
| Secrets | `.env` (git-ignored; template `.env.example`) |

```
configs/experiment.yaml       # every non-secret knob (model, detectors, ratios, hf repo)
src/prepare_data.py           # ImageNet + DTD -> data/
src/generate_fakes.py         # SDXL category-aligned contaminants -> data/fake_id
src/build_from_config.py      # yaml -> build_datasets -> manifests/<idNN>/*.csv
src/dataset_hub.py            # push/pull the private HF image snapshot
src/models/llava.py           # LLaVA -> last-token hidden states (parallel DataLoader)
src/detectors.py              # msp / energy / mahalanobis (registry)
src/run_all.py                # fit on id_train -> score CLEAN id_test vs CLEAN ood_test
src/paperize.py               # results -> CSV + LaTeX + contamination-sweep plots
scripts/bootstrap.sh          # fresh box: env -> data(reuse|build) -> run -> push results
scripts/pull_local.sh         # laptop copies of dataset + results
scripts/destroy_watcher.sh    # laptop-side instance teardown
```

`run_all` **asserts test sets are clean** (`is_fake == 0`) before scoring — the
guardrail is enforced in code, not just documented. It also records `id_acc`,
the probe's classification accuracy on clean ID images, so "identifying real
samples" covers both OOD separation *and* classification.

Two run modes: `data.reuse: false` builds the data and pushes a snapshot to HF;
`data.reuse: true` pulls the pinned snapshot — identical data, no rebuild.

---

## 4. Results & interpretation

**Run `contamination_v1`** (LLaVA-1.5-7B; clean real `id_test` vs clean real
`ood_test`, identical across conditions; AUROC↑):

| ID-train fake % | MSP | Energy | Mahalanobis |
|---|---|---|---|
| 0 (baseline) | 0.970 | 0.993 | 0.989 |
| 1 | 0.971 | 0.993 | 0.989 |
| 10 | 0.972 | 0.993 | 0.989 |
| 25 | 0.973 | 0.992 | 0.988 |

**Finding: contamination is (nearly) harmless in this regime.** Replacing up to
a quarter of the ID training set with impossible images moves real-vs-real OOD
AUROC by at most ~0.005 — within single-seed noise. The tiny MSP uptick is not
meaningful.

**Likely mechanism (stated as hypothesis, not proof):**
1. *Robust estimators.* Class means, a tied covariance, and a logistic probe are
   majority-driven; a ≤25% minority shifts them only slightly.
2. *Near-class contaminants.* A winged cat is still mostly cat-shaped, so its
   hidden state lies near the real "cat" cluster — injected fakes barely drag
   the fitted statistics toward the OOD region.

Both halves are testable (see §6): a detector that can overfit, and contaminants
that are *not* near-class, would each stress the respective half.

**What this does *not* yet show:** whether *impossible* contaminants behave any
differently from ordinary label noise or from AI-generated *possible* images —
the controls that would make "fakes barely hurt" a statement about fakes
specifically.

---

## 5. Retraction note (methods honesty)

Two corrections were made during this project, both worth recording:

1. **Fake-detection framing retracted.** An earlier revision also measured
   detection of a 100%-fake test probe and contaminated the OOD *test* set,
   reading the results as "LVLMs can detect impossibility" (Mahalanobis AUROC
   0.988). This is **confounded**: all fake pools are AI-generated — including
   **WHOOPS!**, whose images are made by designers *using Midjourney/DALL-E/
   Stable Diffusion* — while ID/OOD pools are real photos. Real-vs-fake
   separation therefore cannot be attributed to impossibility rather than
   AI-generation artifacts. All fake-containing test metrics were removed; the
   retracted runs remain in `results`-branch history for provenance. The
   contamination result above is immune (its test comparison is real-vs-real).
2. **Degenerate MSP fixed.** LLaVA's raw class-token logits are near-constant
   (MSP AUROC 0.50); MSP/Energy now score a linear probe trained on ID hidden
   states.

## 6. Limitations & next steps

**Limitations:** single seed (no error bars); 15 classes; one LVLM; ≤25%
contamination; post-hoc (robust) detectors only; `id_acc` not recorded in this
run (now emitted); only 30 unique contaminants per class (cycled at 25%).

**Next experiments, in order of value:**
1. **Contamination-type controls** — matched ratios of (a) real wrong-class
   images and (b) AI-generated *possible* images vs (c) the impossible fakes.
   Distinguishes "fakes are special" from generic label noise.
2. **A detector that can overfit** — fine-tuned head / LoRA on the contaminated
   set; robust estimators may mask effects that learned detectors show.
3. **Multi-seed + higher ratios** (50/75%) — error bars and a breaking point.
4. **OOD-training contamination** (original families 3–4) — needs an
   outlier-exposure-style detector that trains on OOD data.
5. **Second LVLM** (Qwen2-VL) via the registry.
