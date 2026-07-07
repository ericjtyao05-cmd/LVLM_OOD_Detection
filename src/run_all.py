"""Run every condition: LVLM forward pass -> detectors -> AUROC/FPR95 -> results.

For each condition dir under --manifests we:
  1. extract {hidden, logits} for id_train / id_test / ood_test / fake_probe
  2. fit Mahalanobis stats on id_train hidden states
  3. score id_test (negatives) vs ood_test and vs fake_probe (positives)
  4. write results/<run>/<condition>.json ; finally an aggregate summary.json

id_test and fake_probe are identical across conditions, so a path->feature cache
avoids re-running the LVLM on them (~9x saving over the 10-condition grid).
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
    """Caches per-image {hidden, logits} so repeated images aren't recomputed."""
    def __init__(self, model):
        self.model = model
        self.h: dict[str, np.ndarray] = {}
        self.l: dict[str, np.ndarray] = {}

    def features_for(self, rows):
        from PIL import Image
        paths = [r["path"] for r in rows]
        todo = [p for p in paths if p not in self.h]
        if todo:
            imgs = [Image.open(p).convert("RGB") for p in todo]
            out = self.model.extract(imgs)
            for i, p in enumerate(todo):
                self.h[p] = out["hidden"][i]
                self.l[p] = out["logits"][i]
        H = np.stack([self.h[p] for p in paths])
        L = np.stack([self.l[p] for p in paths])
        return H, L


def score_condition(cdir: Path, cache: FeatureCache, methods, n_classes):
    tr = read_manifest(cdir / "id_train.csv")
    idt = read_manifest(cdir / "id_test.csv")
    ood = read_manifest(cdir / "ood_test.csv")
    fake = read_manifest(cdir / "fake_probe.csv")

    Htr, _ = cache.features_for(tr)
    ytr = np.array([int(r["label"]) for r in tr])
    Hid, Lid = cache.features_for(idt)
    Hood, Lood = cache.features_for(ood)
    Hfk, Lfk = cache.features_for(fake)

    stats = fit_stats(Htr, ytr, n_classes)

    # split fake_probe by source for the artifact-vs-impossibility confound check
    is_whoops = np.array(["whoops" in Path(r["path"]).name for r in fake])

    out = {"condition": cdir.name,
           "n": {"id_test": len(idt), "ood_test": len(ood), "fake_probe": len(fake)},
           "methods": {}}
    for m in methods:
        fn = get_detector(m)
        s_id = fn(Lid, Hid, stats)
        s_ood = fn(Lood, Hood, stats)
        s_fk = fn(Lfk, Hfk, stats)
        rec = {}
        rec.update({f"{k}_ood": v for k, v in summarize(s_id, s_ood).items()})
        rec.update({f"{k}_fake": v for k, v in summarize(s_id, s_fk).items()})
        if is_whoops.any():
            rec.update({f"{k}_fake_whoops": v
                        for k, v in summarize(s_id, s_fk[is_whoops]).items()})
        if (~is_whoops).any():
            rec.update({f"{k}_fake_gen": v
                        for k, v in summarize(s_id, s_fk[~is_whoops]).items()})
        out["methods"][m] = rec
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
        for m, rec in res["methods"].items():
            print(f"    {m:12s} AUROC(ood)={rec['auroc_ood']:.3f} "
                  f"AUROC(fake)={rec['auroc_fake']:.3f} "
                  f"FPR95(fake)={rec['fpr95_fake']:.3f}")

    json.dump(summary, open(rroot / "summary.json", "w"), indent=2)
    print(f"[done] results -> {rroot}/")


if __name__ == "__main__":
    main()
