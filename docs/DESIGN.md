# Can ViT-based OOD detectors catch *fake* images? — Experiment Design

## 1. Research question

Classical Out-of-Distribution (OOD) detection asks: *is this test image from a
semantic class the model was trained on?* OOD samples are usually **real** images
of **novel categories** (e.g. train on ImageNet animals, test with textures).

We ask a different question, orthogonal to the usual one:

> When a test image is *semantically familiar* but **physically impossible**
> (a cat **with wings**, a floating waterfall, Godzilla), do existing OOD
> detectors flag it? Or does "impossibility" live in a blind spot?

This connects to **Physical AI / physical plausibility**: a model may recognize
"cat" with high confidence and therefore call the winged cat *in-distribution*,
even though no such thing exists in reality. We measure whether standard scores
(MSP, Energy, KNN) separate real-ID from fake images, and how that separation
**degrades under contamination** of the ID or OOD pools with fakes.

**Scope note (model).** The topic is **LVLM-based** OOD detection, so the model is
a large vision-language model, not a vision-only encoder. Default = **LLaVA-1.5-7B**:
open weights, hidden states exposed via HF `transformers` (`output_hidden_states=True`),
and it fits a single 4090 (fp16 ≈14 GB, inference-only). The detector reads the
model's **internals** — the last-token hidden state and class-restricted next-token
logits — so model and method are coupled (see §4). Model is swappable via
`configs/experiment.yaml` (`model.name`); a second LVLM (Qwen2-VL-2B) is the
planned ablation. Prompt-/generation-based LVLM-OOD methods (ReGuide) are a
separate family we treat as a Phase-2 comparison, not the hidden-state baseline.

## 2. Variables

| Axis | Meaning | Values |
|------|---------|--------|
| `id_fake_ratio`  | fraction of the **ID training** pool replaced by category-aligned fakes (winged cat labelled "cat") | 0, 0.01, 0.10, 0.25 |
| `ood_fake_ratio` | fraction of the **OOD test** set that is fake | 0, 0.01, 0.10, 0.25 |

The four requested families are points on this 2-D grid:

```
Family 1  baseline      (id=0,   ood=0)
Family 2  ID fake       (id=r,   ood=0)     r∈{.01,.10,.25}
Family 3  OOD fake      (id=0,   ood=r)
Family 4  both (matched) (id=r,  ood=r)
```

Total = **1 + 3 + 3 + 3 = 10 conditions** (`grid: "families"`), or the full
4×4 = 16 cross-product with `grid: "full"`.

To keep AUROC comparable across conditions we **replace** rather than add, so set
sizes stay constant. Every condition also emits two *fixed* diagnostic sets:

* `id_test`    — clean real ID images (never contaminated), the reference.
* `fake_probe` — 100%-fake test set. **`id_test` vs `fake_probe` AUROC is the
  headline number**: it answers "can this detector flag impossibility at all?"

## 3. Datasets (small but meaningful first baseline)

### Real ID — ImageNet-200 subset (recommended) or a 10–20 class mini-set
* Use the **ImageNet-200** class list from OpenOOD, or a hand-picked 10–20
  "concrete object" classes that have plausible impossible variants
  (cat, dog, bird, horse, elephant, car, …). Cap at `max_per_class: 300`
  → ~3–6k images. High-res natural photos match the fakes' domain.
* Why not CIFAR? 32×32 doesn't match photorealistic fakes (domain gap would
  confound the fake signal). Use 224×224 natural images.

### Real OOD — standard far-OOD sets
* Any of OpenOOD's far-OOD sets: **iNaturalist**, **SUN**, **Places**,
  **Textures (DTD)**, **OpenImage-O**. Start with one (e.g. iNaturalist or
  Textures) capped at `ood_test_size: 3000`.

### Fake images — two complementary sources
1. **WHOOPS!** (Bitton-Guetta et al., ICCV 2023) — ~500 *human-curated*
   commonsense-defying / physically-impossible images. HF `nlphuji/whoops`.
   This is the gold anchor for `fake_ood` / `fake_probe`.
   → `python src/prepare_sources.py whoops --out data/fake_ood`
2. **SDXL-generated** (`src/generate_fakes.py`) — for **scale** and **category
   alignment** (fakes that belong to an ID class, needed for `id_fake_ratio`):
   * `aligned`  → `data/fake_id/<class>/` (winged cat, giant dog, …)
   * `freeform` → `data/fake_ood/` (Godzilla, upward waterfall, …)

> Keep a held-out **human-curated (WHOOPS-only)** fake test split so the headline
> result isn't just "detect SDXL artifacts". Report generated-fake and
> WHOOPS-fake AUROC separately — if they differ a lot, the detector is keying on
> generation artifacts, not impossibility. This is the key confound to control.

### Directory layout the pipeline expects
```
data/
  id_real/<class>/*.jpg     # real ID, one folder per class
  ood_real/*.jpg            # real OOD (flat)
  fake_id/<class>/*.jpg     # category-aligned fakes (SDXL aligned)
  fake_ood/*.jpg            # any fakes (WHOOPS + SDXL freeform)
```

## 4. Method under test — read the LVLM's internals

One forward pass of LLaVA-1.5-7B per image yields two signals; all detectors are
**post-hoc** (no OOD data at train time):

* **MSP** — Maximum Softmax Probability (Hendrycks & Gimpel, ICLR 2017), on the
  next-token logits restricted to the ID class names. *The* canonical baseline.
* **Energy** — Liu et al., NeurIPS 2020, on the same class-restricted logits.
* **Mahalanobis** — Lee et al., NeurIPS 2018, on the **last-token hidden state**:
  fit one Gaussian per ID class on ID-train hidden states, score by min class
  distance. This is the *hidden-state* method — the one coupled to the LVLM — and
  our headline "does impossibility live in the representation?" probe.

Model↔method coupling is the point: the score is computed on *this* model's
hidden states/logits, so swapping the LVLM re-computes everything. `src/detectors.py`
registers all three; new scores (KNN, ViM, ReAct) drop into the same registry.
Cross-reference **OpenOOD** for standardized method implementations.

## 5. Metrics & expected reads

* **AUROC** (↑ better) and **FPR@95%TPR** (↓ better), computed for:
  * `id_test` vs `ood_test`   — standard OOD performance (sanity).
  * `id_test` vs `fake_probe` — **the fake-detection headline**.
* Report each method across all 10 conditions as two heat-maps
  (AUROC over the id×ood grid).

Hypotheses to look for:
* **H1** Fakes are *harder* than real-OOD: `AUROC(fake) < AUROC(ood)` at (0,0) —
  impossibility is a blind spot because fakes stay near ID classes.
* **H2** ID contamination hurts: AUROC drops as `id_fake_ratio` ↑ (the model
  learns to *accept* impossible images as ID).
* **H3** The hidden-state method (Mahalanobis) beats logit-only ones (MSP/Energy)
  on fakes, or *doesn't* — either way that localizes *where* impossibility is (or
  isn't) encoded: representation vs decision layer.

## 6. Extensions (phase 2)

* **Second LVLM (ablation):** Qwen2-VL-2B behind the same registry interface —
  does a different LVLM expose impossibility better, and does scale matter?
* **Prompt/generation-based method:** ReGuide-style — prompt the LVLM "Is this
  image physically possible?" and use its reject-rate as the detector. Contrasts
  a *reasoning* judge against the *hidden-state* scores (does explicit reasoning
  catch fakes the representation misses?). Uses generation, so it complements —
  not replaces — the hidden-state track.

## 7. Run order

**Automated (intended path):** set `GITHUB_TOKEN` + `HF_TOKEN` in the vast onstart
field; `scripts/bootstrap.sh` does env → data → build → all conditions → upload →
DONE, and `scripts/destroy_watcher.sh` tears the box down. See docs/vast_ai_guide.md.

**Manual / local (for debugging a single stage):**
```bash
pip install -r requirements.txt
python -m src.prepare_data       --config configs/experiment.yaml   # ImageNet+DTD+WHOOPS
python src/generate_fakes.py aligned  --classes tabby_cat ... --out data/fake_id   # GPU
python src/generate_fakes.py freeform --n 500 --out data/fake_ood                  # GPU
python -m src.build_from_config  --config configs/experiment.yaml --out-dir manifests
python -m src.run_all            --config configs/experiment.yaml --manifests manifests \
                                 --results results/local
python -m src.paperize           --results results/local --out results/local/paper
```
