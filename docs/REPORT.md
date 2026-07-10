# Fake-image contamination of OOD-detector training — Progress Report

*Internal progress report. Model: LLaVA-1.5-7B (frozen). Current run:
`results/sweep_v2` (contamination sweep 0→100%, fake_id v2). Supersedes the
earlier ≤25% pilot `contamination_v1`.*

## TL;DR
- **Question:** if fake (physically-impossible, AI-generated) images pollute the
  **ID training set**, does an LVLM-based OOD detector get worse at separating
  **real ID** from **real OOD** images? Test sets are always clean real images.
- **Answer:** **remarkably robust — even at 100%.** Across a 0→100% sweep,
  real-vs-real OOD AUROC stays ~0.97–0.99 and clean-ID accuracy ~0.98; only a
  mild bend appears at 100%. A detector fit *entirely* on impossible images still
  separates real ID from real OOD (Mahalanobis 0.985 AUROC / 0.033 FPR95).
- **Why (and the key caveat):** LLaVA is **frozen** — contamination only perturbs
  a thin post-hoc head (linear probe + class Gaussians) fit on top of fixed
  features. Robustness is therefore partly *by construction*. See §5.

---

## 1. Dataset construction (what & why)

| Pool | Source | Size | Role |
|---|---|---|---|
| **Real ID** | ImageNet-1k, 15 concrete classes, 300/class | 4500 | training (contaminated) + **clean test** |
| **Real OOD** | Describable Textures (DTD) | 3000 | **clean test** only |
| **Fake contaminants** | SDXL, **240/class**, per-class-group templates (below) | 3600 | injected into `id_train` **only** |

**Design guardrail.** Contamination is **only in training**. `id_test` (~900
clean real ID) and `ood_test` (~3000 clean real OOD) are identical across all
conditions (enforced by an assertion in `run_all`). One condition = one
contamination ratio ∈ **{0, 10, 25, 50, 75, 100}%** → conditions `id00…id100`.
Replace-not-add keeps the training-set size constant.

**Fake generation — what makes a clean contaminant (fake_id v2).** The fakes must
be **unambiguously impossible** *and* **recognisable as their class**. Two rounds
of sample review taught us that SDXL renders different impossibilities per class
type, so templates are class-group-specific:

| Class group | Impossibility that works | Why others fail |
|---|---|---|
| **Mammals** (cat, dog, elephant, zebra, tiger, bear) | attached **wings** (feathered/bat/butterfly) → winged mammal | — |
| **Birds / fish** (eagle, ostrich, goldfish) | **giant-with-scale** (building-sized, cars/people for reference) | wings just make a normal bird / a fish-bird hybrid |
| **Ground objects** (car, bus, bike, piano, locomotive) | **feathered wings** → winged vehicle | creature wings (bat/dragon) spawn a *separate* animal |
| **Airliner** | **inverse scale** (car-sized plane among real cars) | wings → a bird; giant → a normal (already-big) plane |

Ambiguous "impossibilities" that can occur in reality (on fire, glass sculpture,
floating, melting) were **excluded** — they'd pollute the pool with real-looking
images. Residual per-seed noise remains (some giant-ostriches render normal-sized,
goldfish come out fish-vehicle hybrids) and is accepted as realistic contamination.

**Reproducibility.** Images live in the HF dataset
`EricY05/lvlm-ood-fake-data`, pinned revision **`6d8de16…`** (currently public,
per the owner's choice); split manifests are committed to git.

---

## 2. Model & detectors

**Model: LLaVA-1.5-7B, used frozen** — one forward pass per image
(`output_hidden_states=True`, `no_grad`) yields the last-token hidden state; its
weights never change. All detectors are **post-hoc** and fit **only** on the
(possibly contaminated) ID training set's hidden states:

| Detector | Fits | Scores by | Contamination path |
|---|---|---|---|
| **MSP** (Hendrycks & Gimpel 2017) | linear probe on hidden states | max softmax | corrupted probe weights |
| **Energy** (Liu et al. 2020) | same probe | `−logsumexp(logits)` | corrupted probe weights |
| **Mahalanobis** (Lee et al. 2018) | class means + tied covariance | min class distance | shifted class geometry |

`id_acc` (probe accuracy on clean ID images) is also recorded, so "identifying
real samples" covers both OOD separation and classification.

---

## 3. Workflow & code scaffold

Principles: swappable (registry + one YAML), self-provisioning (bootstrap on a
fresh GPU), preserved (images on HF, results on a git branch).

| Artifact | Home |
|---|---|
| Code · manifests · configs | git `main` |
| Images (`data/`) | HF dataset (pinned revision) |
| Results (JSON/tables/plots) | git `results` branch |
| Secrets | `.env` (git-ignored; template `.env.example`) |

```
configs/experiment.yaml       # non-secret knobs (model, detectors, ratios, hf repo)
src/prepare_data.py           # ImageNet + DTD -> data/
src/generate_fakes.py         # SDXL per-class-group contaminants (batched) -> data/fake_id
src/build_from_config.py      # yaml -> manifests/<idNN>/*.csv (clean tests enforced)
src/dataset_hub.py            # push/pull the HF image snapshot
src/models/llava.py           # frozen LLaVA -> hidden states (parallel DataLoader)
src/detectors.py              # msp / energy / mahalanobis
src/run_all.py                # fit on id_train -> score CLEAN id_test vs CLEAN ood_test + id_acc
src/paperize.py               # results -> CSV + LaTeX + contamination-sweep plots
scripts/{bootstrap,pull_local,destroy_watcher}.sh
```

**Notes from this run.** SDXL generation runs in an isolated venv; the sweep ran
on an RTX 5090 (Blackwell) — which needed a fresh torch build and a C compiler
(triton JIT). The parallel DataLoader uses the `file_system` tensor-sharing
strategy to survive the container's low `ulimit -n`.

---

## 4. Results

**Run `sweep_v2`** — clean real `id_test` vs clean real `ood_test`, identical
across conditions. AUROC↑ / FPR95↓ / id_acc↑.

| Fake % | id_acc | MSP AUC | Energy AUC | **Maha AUC** | MSP FPR95 | Energy FPR95 | **Maha FPR95** |
|---|---|---|---|---|---|---|---|
| 0 | 0.998 | 0.975 | 0.995 | 0.990 | 0.017 | 0.008 | 0.009 |
| 10 | 0.997 | 0.972 | 0.991 | 0.990 | 0.023 | 0.011 | 0.012 |
| 25 | 0.996 | 0.972 | 0.992 | 0.989 | 0.023 | 0.006 | 0.012 |
| 50 | 0.996 | 0.971 | 0.991 | 0.990 | 1.000 | 0.010 | 0.016 |
| 75 | 0.996 | 0.959 | 0.988 | 0.988 | 1.000 | 0.014 | 0.019 |
| **100** | **0.976** | **0.882** | **0.971** | **0.985** | 1.000 | 0.084 | 0.033 |

**Findings:**
1. **Contamination is nearly harmless, and the curve is flat to 75%.** Even
   replacing three-quarters of the ID training set with impossible images leaves
   real-vs-real OOD AUROC essentially unchanged (~0.99) for the feature-geometry
   and energy detectors, and clean-ID accuracy at ~0.996.
2. **The `id100` near-class test passes.** Fit *entirely* on impossible images,
   Mahalanobis still separates real ID from real OOD at **0.985 / FPR95 0.033**,
   and the probe still classifies real IDs at **97.6%** — because LLaVA's frozen
   representation of a winged cat is still cat-shaped, so the class statistics fit
   on fakes still land near the real class.
3. **A mild bend appears only at 100%,** and the weakest method breaks first: MSP
   AUROC drops to 0.88 and its FPR95 collapses to 1.0 from 50% onward (its
   confidence loses calibration under heavy contamination), while Mahalanobis
   stays usable throughout.

---

## 5. The central caveat (why robustness is partly built-in)

**LLaVA is frozen.** The contamination never touches the representation — only a
**linear probe + class Gaussians** fit on top of fixed features. Those are
majority-driven robust estimators over an already-excellent representation, so a
few (or even a majority of) near-class fakes barely move them. In that sense
"contamination is harmless" is *partly by construction*.

So this study legitimately shows:
> *An OOD detector reading a **frozen** LVLM's features and fitting a light
> post-hoc head on contaminated ID data is robust to fake contamination up to
> 100%.*

It does **not** show that *training an LVLM on fakes is harmless* — for that the
detector must **learn from** the contaminated data. That is the next experiment.

## 6. Limitations & next steps

**Limitations:** frozen backbone + post-hoc detectors (see §5); single seed (no
error bars); 15 classes; one LVLM; per-seed noise in the fake pool.

**Next experiments, in order of value:**
1. **LoRA fine-tune (or trainable head) on the contaminated `id_train`,** swept
   0→100% — the version where fakes can actually reshape the model and the answer
   isn't semi-predetermined.
2. **Contamination-type controls** — real wrong-class images and AI-generated
   *possible* images at matched ratios, to isolate whether *impossible*
   contaminants are special vs generic label noise.
3. **Multi-seed** for error bars; a **second LVLM** (Qwen2-VL) via the registry.

## Appendix — retraction (kept for honesty)
An earlier revision measured detection of a 100%-fake test probe and read it as
"LVLMs detect impossibility." That was **retracted as confounded**: all fake
pools (incl. WHOOPS!) are AI-generated while ID/OOD are real photos, so real-vs-
fake separation cannot be attributed to impossibility rather than AI-generation.
Also fixed en route: LLaVA's raw class-token logits are degenerate (MSP→0.50), so
MSP/Energy score a linear probe instead.
