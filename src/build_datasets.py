#!/usr/bin/env python3
"""Build ID-training-contamination manifests. TEST SETS ARE ALWAYS CLEAN.

Research question
-----------------
Does mixing fake (physically-impossible, AI-generated) images into the ID
*training* set degrade a detector's ability to separate REAL ID from REAL OOD
images?

Design guardrail (do not change without revisiting the research question):
  * Contamination happens ONLY in `id_train` -- the set detectors fit on.
  * `id_test` and `ood_test` are ALWAYS clean real images, identical across
    conditions. The dependent variable is real-vs-real OOD detection (and
    clean-ID classification accuracy), never "fake detection".
  * A condition = one contamination ratio. Nothing else varies.

(An earlier revision also injected fakes into the OOD test set and emitted a
100%-fake probe. That drifted the study toward "can we detect fakes?", which is
confounded -- all fake pools, including WHOOPS!, are AI-generated, so real-vs-
fake separation cannot be attributed to impossibility. Removed on purpose.)

This script produces reproducible CSV *manifests* (paths only, no pixel copies):

    manifests/<idNN>/id_train.csv   # (1-r) real + r category-aligned fakes
    manifests/<idNN>/id_test.csv    # clean real ID   (fixed across conditions)
    manifests/<idNN>/ood_test.csv   # clean real OOD  (fixed across conditions)

Expected source layout (symlinks are fine)::

    data/
      id_real/<class_name>/*.jpg     # real ID images, one folder per class
      ood_real/*.jpg                 # real OOD images (flat; label ignored)
      fake_id/<class_name>/*.jpg     # category-aligned fakes (contaminant pool)

Usage::

    python src/build_datasets.py \
        --data-root data \
        --out-dir manifests \
        --config configs/datasets.example.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Source discovery
# --------------------------------------------------------------------------- #
def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def scan_class_dir(root: Path) -> dict[str, list[Path]]:
    """root/<class>/*.jpg -> {class: [paths]}."""
    out: dict[str, list[Path]] = {}
    if not root.is_dir():
        return out
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = _list_images(cls)
        if imgs:
            out[cls.name] = imgs
    return out


def scan_flat_dir(root: Path) -> list[Path]:
    """root/*.jpg (recursive) -> [paths]."""
    return _list_images(root)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class BuildConfig:
    id_fake_ratios: list[float] = field(default_factory=lambda: [0.0, 0.01, 0.10, 0.25])
    id_test_frac: float = 0.2       # held-out clean ID test fraction
    max_per_class: int | None = None  # cap ID images per class (speed)
    ood_test_size: int | None = None  # cap OOD test size (None = all)
    seed: int = 0

    @staticmethod
    def load(path: str | None) -> "BuildConfig":
        if not path:
            return BuildConfig()
        with open(path) as f:
            raw = json.load(f)
        return BuildConfig(**raw)


# --------------------------------------------------------------------------- #
# Splitting & contamination
# --------------------------------------------------------------------------- #
def split_id(id_real: dict[str, list[Path]], cfg: BuildConfig, rng: random.Random):
    """Per-class train/test split of real ID images (test is always clean)."""
    train: dict[str, list[Path]] = {}
    test: dict[str, list[Path]] = {}
    for cls, imgs in id_real.items():
        imgs = list(imgs)
        rng.shuffle(imgs)
        if cfg.max_per_class:
            imgs = imgs[: cfg.max_per_class]
        n_test = max(1, int(round(len(imgs) * cfg.id_test_frac)))
        test[cls] = imgs[:n_test]
        train[cls] = imgs[n_test:]
    return train, test


def contaminate_id_train(id_train, fake_id, ratio, rng):
    """Replace `ratio` of each class's train images with category-aligned fakes.

    Returns list of (path, label, is_fake). Total size per class is preserved
    (replace, not add) so conditions stay size-matched. Falls back to a shared
    fake pool if a class has no aligned fakes.
    """
    shared_fakes = [p for ps in fake_id.values() for p in ps]
    rows = []
    for cls, real in id_train.items():
        n = len(real)
        n_fake = int(round(n * ratio))
        pool = list(fake_id.get(cls, [])) or list(shared_fakes)
        rng.shuffle(pool)
        rng.shuffle(real)
        fakes = [(pool[i % len(pool)], cls, 1) for i in range(n_fake)] if pool else []
        reals = [(p, cls, 0) for p in real[: n - len(fakes)]]
        rows.extend(reals + fakes)
    rng.shuffle(rows)
    return rows


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
FIELDS = ["path", "label", "role", "is_fake", "split"]


def write_manifest(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def cond_name(id_r: float) -> str:
    return f"id{int(round(id_r * 100)):02d}"


def run(data_root, out_dir, cfg: BuildConfig):
    """Build all condition manifests. Reusable by the CLI and build_from_config."""
    root = Path(data_root)
    out = Path(out_dir)

    id_real = scan_class_dir(root / "id_real")
    ood_real = scan_flat_dir(root / "ood_real")
    fake_id = scan_class_dir(root / "fake_id")

    print(f"[scan] id_real classes={len(id_real)} imgs={sum(map(len, id_real.values()))}")
    print(f"[scan] ood_real imgs={len(ood_real)}")
    print(f"[scan] fake_id classes={len(fake_id)} imgs={sum(map(len, fake_id.values()))}")
    if not id_real:
        raise SystemExit(f"No ID images under {root/'id_real'} (need <class>/*.jpg).")
    if not ood_real:
        raise SystemExit(f"No OOD images under {root/'ood_real'}.")

    # class -> integer index (stable, sorted)
    classes = sorted(id_real.keys())
    cls2idx = {c: i for i, c in enumerate(classes)}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "classes.json", "w") as f:
        json.dump({"classes": classes, "cls2idx": cls2idx}, f, indent=2)

    # Fixed clean test sets: identical rows in EVERY condition (the guardrail).
    base_rng = random.Random(cfg.seed)
    id_train, id_test = split_id(id_real, cfg, base_rng)

    ood_pool = list(ood_real)
    base_rng.shuffle(ood_pool)
    if cfg.ood_test_size:
        ood_pool = ood_pool[: cfg.ood_test_size]

    rows_id_test = [dict(path=str(p), label=cls2idx[c], role="id", is_fake=0, split="test")
                    for c, ps in id_test.items() for p in ps]
    rows_ood_test = [dict(path=str(p), label=-1, role="ood", is_fake=0, split="test")
                     for p in ood_pool]

    ratios = sorted(set(cfg.id_fake_ratios))
    print(f"[plan] {len(ratios)} conditions: " + ", ".join(cond_name(r) for r in ratios))

    for id_r in ratios:
        cdir = out / cond_name(id_r)
        rng = random.Random(cfg.seed + int(id_r * 1000) * 131)

        train_rows = contaminate_id_train(id_train, fake_id, id_r, rng)
        write_manifest(cdir / "id_train.csv", [
            dict(path=str(p), label=cls2idx[c], role="id", is_fake=fk, split="train")
            for p, c, fk in train_rows])
        write_manifest(cdir / "id_test.csv", rows_id_test)
        write_manifest(cdir / "ood_test.csv", rows_ood_test)

        n_tr_fake = sum(fk for _, _, fk in train_rows)
        print(f"[build] {cond_name(id_r)}: train={len(train_rows)} (fake={n_tr_fake}) "
              f"id_test={len(rows_id_test)} ood_test={len(rows_ood_test)} (both clean)")

    print(f"[done] manifests written under {out}/")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="manifests")
    ap.add_argument("--config", default=None, help="JSON BuildConfig (optional)")
    args = ap.parse_args()
    run(args.data_root, args.out_dir, BuildConfig.load(args.config))


if __name__ == "__main__":
    main()
