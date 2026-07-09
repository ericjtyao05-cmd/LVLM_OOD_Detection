"""Run every condition: LVLM forward pass -> detectors -> metrics -> results.

The study: does contaminating the ID *training* set with fakes degrade
REAL-vs-REAL OOD detection? Accordingly, for each condition dir we:
  1. extract hidden states for id_train (contaminated) / id_test / ood_test
     (both test sets are CLEAN REAL images, identical across conditions)
  2. fit Mahalanobis stats + a linear probe on the (contaminated) id_train
  3. score clean id_test (negatives) vs clean ood_test (positives)
     -> auroc_ood / fpr95_ood per detector
  4. compute clean-ID classification accuracy of the probe (id_acc)
  5. write results/<run>/<condition>.json ; finally an aggregate summary.json

There is deliberately NO fake-containing test set: real-vs-fake separation is
confounded (all fake pools are AI-generated), so it is not measured here.

id_test/ood_test repeat across conditions, so a path->feature cache avoids
re-running the LVLM on them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import models  # noqa: F401  registers models
from . import detectors as det_mod  # noqa: F401  registers detectors
from .config import load_config, id_class_names
from .detectors import fit_stats
from .metrics import summarize
from .registry import get_model, get_detector


def read_manifest(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


class FeatureCache:
    """Caches per-image hidden states (by path) so repeats aren't recomputed."""
    def __init__(self, model):
        self.model = model
        self.h: dict[str, np.ndarray] = {}

    def features_for(self, rows):
        paths = [r["path"] for r in rows]
        todo = [p for p in paths if p not in self.h]
        if todo:
            out = self.model.extract(todo)          # paths -> hidden (parallel)
            for i, p in enumerate(todo):
                self.h[p] = out["hidden"][i]
        return np.stack([self.h[p] for p in paths])


def train_probe(H_train, y_train, n_classes, epochs=150, lr=1e-2):
    """Linear probe (multinomial logistic) on ID-train hidden states, in torch.

    Gives well-defined C-way logits for MSP/Energy -- the raw class-token logits
    were near-constant (MSP collapsed to AUROC 0.5). Trained on ID only, so it's a
    legitimate post-hoc OOD setup. Pure torch (no sklearn dep). Runs on CPU to
    avoid contending with the resident LLaVA model on GPU. Features are
    standardized (fit on train) for stable convergence.
    Returns a function hidden -> logits[N,C].
    """
    import torch
    import torch.nn.functional as F
    X = torch.as_tensor(np.asarray(H_train), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    head = torch.nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(Xn), y).backward()
        opt.step()
    head.eval()
    W = head.weight.detach(); b = head.bias.detach()

    def logits(H):
        h = (torch.as_tensor(np.asarray(H), dtype=torch.float32) - mu) / sd
        with torch.no_grad():
            return (h @ W.t() + b).numpy()
    return logits


def score_condition(cdir: Path, cache: FeatureCache, methods, n_classes):
    """Fit on (contaminated) id_train; evaluate on CLEAN id_test vs CLEAN ood_test."""
    tr = read_manifest(cdir / "id_train.csv")
    idt = read_manifest(cdir / "id_test.csv")
    ood = read_manifest(cdir / "ood_test.csv")

    n_tr_fake = sum(int(r.get("is_fake", 0)) for r in tr)
    assert all(int(r.get("is_fake", 0)) == 0 for r in idt + ood), \
        f"{cdir.name}: test sets must be clean real images (design guardrail)"

    Htr = cache.features_for(tr)
    ytr = np.array([int(r["label"]) for r in tr])
    Hid = cache.features_for(idt)
    Hood = cache.features_for(ood)

    stats = fit_stats(Htr, ytr, n_classes)     # Mahalanobis (raw hidden states)
    probe = train_probe(Htr, ytr, n_classes)   # MSP/Energy logits
    Lid, Lood = probe(Hid), probe(Hood)

    # clean-ID classification accuracy of the (contaminated) probe
    yid = np.array([int(r["label"]) for r in idt])
    id_acc = float((Lid.argmax(axis=1) == yid).mean())

    out = {"condition": cdir.name,
           "n": {"id_train": len(tr), "id_train_fake": n_tr_fake,
                 "id_test": len(idt), "ood_test": len(ood)},
           "id_acc": id_acc,
           "methods": {}}
    for m in methods:
        fn = get_detector(m)
        s_id = fn(Lid, Hid, stats)
        s_ood = fn(Lood, Hood, stats)
        out["methods"][m] = {f"{k}_ood": v for k, v in summarize(s_id, s_ood).items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mroot = Path(args.manifests)
    rroot = Path(args.results); rroot.mkdir(parents=True, exist_ok=True)
    classes = id_class_names(cfg)
    n_classes = len(classes)

    model = get_model(cfg["model"]["name"], class_names=classes,
                      dtype=cfg["model"].get("dtype", "float16"),
                      prompt=cfg["model"].get("prompt"))
    cache = FeatureCache(model)
    methods = cfg["detector"]["methods"]

    conds = sorted(d for d in mroot.iterdir()
                   if d.is_dir() and (d / "id_train.csv").exists())
    summary = {"model": cfg["model"]["name"], "methods": methods,
               "classes": classes, "conditions": []}
    for cdir in conds:
        print(f"[run] {cdir.name}")
        res = score_condition(cdir, cache, methods, n_classes)
        json.dump(res, open(rroot / f"{cdir.name}.json", "w"), indent=2)
        summary["conditions"].append(res)
        print(f"    id_acc={res['id_acc']:.3f}")
        for m, rec in res["methods"].items():
            print(f"    {m:12s} AUROC(ood)={rec['auroc_ood']:.3f} "
                  f"FPR95(ood)={rec['fpr95_ood']:.3f}")

    json.dump(summary, open(rroot / "summary.json", "w"), indent=2)
    print(f"[done] results -> {rroot}/")


if __name__ == "__main__":
    main()
