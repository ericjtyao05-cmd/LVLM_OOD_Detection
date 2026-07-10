# Does fake-image contamination of the training set hurt OOD detection? — Design

## 1. Research question

Training data increasingly contains AI-generated and physically-impossible
images scraped from the web. We ask:

> If a fraction of an LVLM-based OOD detector's **ID training set** is replaced
> by fake (physically-impossible, AI-generated) images — a cat with wings
> labelled "cat" — does the detector get worse at its actual job: separating
> **real ID** images from **real OOD** images?

**Design guardrail (the core of the study):** contamination happens **only in
training**. The test sets — clean real `id_test` vs clean real `ood_test` — are
**identical across all conditions**. The dependent variables are real-vs-real
OOD detection (AUROC/FPR95) and clean-ID classification accuracy. Nothing is
ever measured against a fake-containing test set.

### Retracted: the "fake detection" framing
An earlier revision also injected fakes into the OOD *test* set and measured
detection of a 100%-fake probe ("can LVLMs flag impossibility?"). That framing
is **confounded and was retracted**: every available fake pool — including
WHOOPS!, whose images are created by designers *with Midjourney/DALL-E/Stable
Diffusion* (it is a *synthetic*-image benchmark) — is AI-generated, while the
ID/OOD pools are real photos. Real-vs-fake separation therefore cannot be
attributed to *impossibility* rather than *AI-generation artifacts*. Answering
the impossibility question requires an AI-generated-but-*possible* control set
(future work, §6). The current study is immune to this confound because its
test comparison is real-vs-real.

## 2. Variables

| | |
|---|---|
| **Independent** | `id_fake_ratio` ∈ {0, 0.10, 0.25, 0.50, 0.75, 1.0} — fraction of the ID training set replaced (not added) by category-aligned fakes |
| **Dependent** | `auroc_ood` / `fpr95_ood` (clean real ID vs clean real OOD), `id_acc` (probe accuracy on clean ID) |
| **Fixed** | model, detectors, prompt, test sets, seed |

One condition = one ratio → 6 conditions: `id00 … id100`. The sweep goes to
**100%** on purpose: ≥50% is where robust estimators should break if ever, and
`id100` = "train the detector *only* on impossible {class}s, test on real" — the
near-class test. Replacement (not addition) keeps the training set size constant.

## 3. Data

| Pool | Source | Size | Role |
|---|---|---|---|
| Real ID | ImageNet-1k (gated), 15 concrete classes, 300/class | 4500 | train (contaminated) + clean test |
| Real OOD | Describable Textures (DTD) | 3000 | clean test only |
| Contaminants | SDXL **category-aligned** fakes, **240/class** (needed so 100% isn't a few images cycled) | 3600 | injected into `id_train` only |

Category alignment matters: the contaminant must plausibly carry the class
label (an impossible *cat* labelled "cat"), otherwise it is generic label noise
rather than the contamination scenario we care about.

**Fake generation is class-group-specific (fake_id v2).** Sample review showed
SDXL renders different impossibilities per class type: **mammals** → attached
wings; **birds/fish** → giant-with-scale (wings just make a bird); **ground
objects** → feathered wings (creature wings spawn a separate animal);
**airliner** → inverse scale (a car-sized plane; wings→bird, giant→normal plane).
Ambiguous "impossibilities" that can occur in reality (fire, glass, floating,
melting) are excluded; some per-seed noise remains and is accepted as realistic
contamination. WHOOPS! / freeform images stay in the snapshot but are unused here
(reserved for §6 controls).

Images live in the HF dataset `EricY05/lvlm-ood-fake-data`, pinned revision
`6d8de16…` (currently public, per the owner's choice); the split **manifests**
are committed to git → (pinned revision × committed manifests) = byte-exact reruns.

## 4. Model & detectors

**Model: LLaVA-1.5-7B, used FROZEN** — one forward pass per image
(`output_hidden_states=True`, `no_grad`; the prompt *"What is the main object?
Answer with a single word."*) yields the last-token hidden state. Weights never
change; contamination cannot touch the representation, only the head fit on top
(this is the study's central caveat — see §5).

**Detectors** (all post-hoc; all fit **only** on the — possibly contaminated —
ID training set):

* **MSP** (Hendrycks & Gimpel 2017) — max softmax of a linear probe trained on
  ID-train hidden states. The canonical baseline.
* **Energy** (Liu et al. 2020) — `−logsumexp` of the same probe logits.
* **Mahalanobis** (Lee et al. 2018) — min class-conditional Mahalanobis distance
  on raw hidden states (tied, shrinkage-regularised covariance).

The probe is used because LLaVA's raw next-token class logits are
near-constant (MSP degenerates to AUROC 0.5 on them). Contamination reaches the
detectors through the statistics they fit: class means/covariance
(Mahalanobis) and probe weights (MSP/Energy).

## 5. Result (run `sweep_v2`, 0→100%)

Real-vs-real OOD AUROC is **flat to 75% and only mildly bends at 100%**
(full table + `id_acc` and FPR95 in `docs/REPORT.md §4`):

| ID fake % | MSP | Energy | Mahalanobis | id_acc |
|---|---|---|---|---|
| 0 | 0.975 | 0.995 | 0.990 | 0.998 |
| 25 | 0.972 | 0.992 | 0.989 | 0.996 |
| 75 | 0.959 | 0.988 | 0.988 | 0.996 |
| 100 | 0.882 | 0.971 | **0.985** | 0.976 |

**The `id100` near-class test passes:** fit *entirely* on impossible images, the
detector still separates real ID from real OOD (Maha 0.985) and classifies real
IDs at 97.6% — because LLaVA's *frozen* representation of a winged cat is still
cat-shaped, so class statistics fit on fakes still land near the real class.

**Central caveat (why robustness is partly built-in):** LLaVA is frozen, so
contamination only perturbs a linear probe + class Gaussians over fixed features
— majority-driven robust estimators over an already-good representation. This
study shows a *frozen-feature post-hoc detector* is robust to contamination; it
does **not** show that *training an LVLM on fakes* is harmless. Single seed.

## 6. What's missing / next experiments

1. **LoRA fine-tune (or trainable head) on the contaminated `id_train`,** swept
   0→100% — the version where fakes can actually reshape the model. The frozen
   post-hoc result above cannot show this; it's the highest-value follow-up.
2. **Contamination-type controls** — same ratios with (a) real wrong-class
   images (label noise) and (b) AI-generated *possible* images. Only this
   comparison can say whether *impossible* contaminants are special.
3. **OOD-training contamination** — the original families 3–4 need an
   outlier-exposure-style method that *trains on* OOD data.
4. **Rigor** — multiple seeds (error bars), a second LVLM (Qwen2-VL) via the registry.

## 7. Run order

**Automated:** set `HF_TOKEN` + `GITHUB_TOKEN` on the box; `scripts/bootstrap.sh`
does env → data (build+push HF, or pull pinned snapshot when `data.reuse: true`)
→ manifests → run → push results to the git `results` branch.

**Manual:**
```bash
pip install -r requirements.txt
python -m src.prepare_data       --config configs/experiment.yaml
python src/generate_fakes.py aligned --classes tabby_cat ... --out data/fake_id   # GPU
python -m src.build_from_config  --config configs/experiment.yaml --out-dir manifests
python -m src.run_all            --config configs/experiment.yaml --manifests manifests \
                                 --results results/run1
python -m src.paperize           --results results/run1 --out results/run1/paper
```
