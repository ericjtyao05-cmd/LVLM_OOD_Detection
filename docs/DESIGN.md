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
| **Independent** | `id_fake_ratio` ∈ {0, 0.01, 0.10, 0.25} — fraction of the ID training set replaced (not added) by category-aligned fakes |
| **Dependent** | `auroc_ood` / `fpr95_ood` (clean real ID vs clean real OOD), `id_acc` (probe accuracy on clean ID) |
| **Fixed** | model, detectors, prompt, test sets, seed |

One condition = one ratio → 4 conditions: `id00`, `id01`, `id10`, `id25`.
Replacement (not addition) keeps the training set size constant across
conditions.

## 3. Data

| Pool | Source | Size | Role |
|---|---|---|---|
| Real ID | ImageNet-1k (gated), 15 concrete classes, 300/class | 4500 | train (contaminated) + clean test |
| Real OOD | Describable Textures (DTD) | 3000 | clean test only |
| Contaminants | SDXL **category-aligned** fakes (impossible variants of each ID class: winged, giant, transparent, two-headed, …) | 30/class | injected into `id_train` only |

Category alignment matters: the contaminant must plausibly carry the class
label (an impossible *cat* labelled "cat"), otherwise it is generic label noise
rather than the contamination scenario we care about.

WHOOPS! and SDXL freeform images remain in the data snapshot but are **not used
in the current design** (reserved for future controls; see §6).

Images live in a **private HF dataset** (`EricY05/lvlm-ood-fake-data`, pinned
`revision`) — private because ImageNet redistribution is restricted. The split
**manifests** (CSV path lists) are committed to git; pinned revision ×
committed manifests = byte-exact reruns.

## 4. Model & detectors

**Model:** LLaVA-1.5-7B (open LVLM; hidden states via HF `transformers`; fits a
24 GB 4090 in fp16, inference-only). One forward pass per image with the prompt
*"What is the main object in this image? Answer with a single word."* yields the
last-token hidden state.

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

## 5. Result to date (run `contamination_v1`, cleaned from full_v2)

Real-vs-real OOD AUROC is **flat under contamination up to 25%**:

| ID fake % | MSP | Energy | Mahalanobis |
|---|---|---|---|
| 0 | 0.970 | 0.993 | 0.989 |
| 1 | 0.971 | 0.993 | 0.989 |
| 10 | 0.972 | 0.993 | 0.989 |
| 25 | 0.973 | 0.992 | 0.988 |

Interpretation and caveats live in `docs/REPORT.md`. Short version: post-hoc
detectors are majority-driven robust estimators, and category-aligned fakes
stay near their host class — so moderate contamination barely moves the fitted
statistics. Single seed; `id_acc` not recorded in this run (now emitted by
`run_all`).

## 6. What's missing / next experiments

1. **Contamination-type controls** — same ratios with (a) real wrong-class
   images (label noise) and (b) AI-generated *possible* images. Only this
   comparison can say whether *impossible* contaminants are special.
2. **A detector that can overfit** — fine-tune a head (or LoRA) on the
   contaminated set; robust estimators may hide effects that learned ones show.
3. **OOD-training contamination** — the original families 3–4 need an
   outlier-exposure-style method that *trains on* OOD data; post-hoc methods
   have no OOD-training slot.
4. **Rigor** — multiple seeds (error bars), higher ratios (50/75%) to find a
   breaking point, a second LVLM (Qwen2-VL) via the registry.
5. **The impossibility question, done right** — an AI-possible control set
   would de-confound real-vs-fake separation and revive the retracted question.

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
