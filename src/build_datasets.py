#!/usr/bin/env python3
"""Build mixed ID/OOD datasets with controllable fake-image contamination.

Research question
-----------------
Can existing ViT-based OOD detectors flag *physically-impossible* ("fake")
images -- e.g. a cat with wings, or Godzilla? Fake images can be semantically
close to in-distribution (ID) classes yet violate physical/common-sense reality.

This script produces reproducible CSV *manifests* (it does NOT copy pixels) that
downstream loaders (OpenOOD, or a plain torch Dataset) can consume. Fake images
are injected along two independent axes:

  * id_fake_ratio  : fraction of the ID *training* pool replaced by fake images
                     (category-aligned fakes, e.g. a winged cat labelled "cat").
  * ood_fake_ratio : fraction of the OOD *test* set replaced by fake images.

Each (id_fake_ratio, ood_fake_ratio) pair is one experiment "condition". The
four families requested map onto the (id, ood) grid:

    Family 1 (baseline)  : (0.00, 0.00)
    Family 2 (ID fake)   : (r, 0.00)      r in {0.01, 0.10, 0.25}
    Family 3 (OOD fake)  : (0.00, r)      r in {0.01, 0.10, 0.25}
    Family 4 (both)      : (r, r)         r in {0.01, 0.10, 0.25}  (matched)
                           or full cross-product with --grid full

Every condition also emits two *fixed* diagnostic test sets so results stay
comparable across conditions:
    id_test    : clean real ID test images (never contaminated)
    fake_probe : 100%-fake test set (the headline "can we detect fakes?" probe)

Expected source layout (symlinks are fine)::

    data/
      id_real/<class_name>/*.jpg     # real ID images, one folder per class
      ood_real/*.jpg                 # real OOD images (flat; label ignored)
      fake_id/<class_name>/*.jpg     # category-aligned fakes (SDXL-generated)
      fake_ood/*.jpg                 # any fakes (WHOOPS! + generated), flat

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
import os
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
    ood_fake_ratios: list[float] = field(default_factory=lambda: [0.0, 0.01, 0.10, 0.25])
    grid: str = "families"          # "families" (1+3+3+3) or "full" (cross-product)
    id_test_frac: float = 0.2       # held-out clean ID test fraction
    max_per_class: int | None = None  # cap ID images per class (speed)
    ood_test_size: int | None = None  # cap OOD test size (None = all)
    fake_probe_size: int | None = 500  # size of the 100%-fake diagnostic set
    seed: int = 0

    @staticmethod
    def load(path: str | None) -> "BuildConfig":
        if not path:
            return BuildConfig()
        with open(path) as f:
            raw = json.load(f)
        return BuildConfig(**raw)


def enumerate_conditions(cfg: BuildConfig) -> list[tuple[float, float]]:
    idr = sorted(set(cfg.id_fake_ratios))
    oodr = sorted(set(cfg.ood_fake_ratios))
    if cfg.grid == "full":
        return [(a, b) for a in idr for b in oodr]
    # "families": baseline + ID-only + OOD-only + matched-both
    nz_id = [r for r in idr if r > 0]
    nz_ood = [r for r in oodr if r > 0]
    conds = [(0.0, 0.0)]
    conds += [(r, 0.0) for r in nz_id]
    conds += [(0.0, r) for r in nz_ood]
    conds += [(r, r) for r in nz_id if r in nz_ood]
    # de-dup preserving order
    seen, out = set(), []
    for c in conds:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Splitting & mixing
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

    Returns list of (path, label, is_fake). Total size per class is preserved.
    Falls back to a shared fake pool if a class has no aligned fakes.
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


def build_ood_test(ood_real, fake_ood, ratio, size, rng):
    """OOD test set = (1-ratio) real OOD + ratio fake OOD. Returns (path, is_fake).

    The `ratio` is always honored exactly. Total size is `size` (or all real when
    None), but is reduced if the real-OOD pool is too small to supply the
    (1-ratio) real portion -- so ratios stay comparable across conditions instead
    of silently inflating when real images run out. Fakes are cycled with
    replacement, so they never limit size.
    """
    ood_real = list(ood_real)
    rng.shuffle(ood_real)
    n = size if size is not None else len(ood_real)
    # cap n so the real portion fits the available real pool
    if ratio < 1.0:
        n = min(n, int(len(ood_real) / (1.0 - ratio)))
    n_fake = int(round(n * ratio))
    n_real = n - n_fake
    if not fake_ood:
        n_fake, n_real = 0, min(n, len(ood_real))
    pool = list(fake_ood)
    rng.shuffle(pool)
    reals = [(p, 0) for p in ood_real[:n_real]]
    fakes = [(pool[i % len(pool)], 1) for i in range(n_fake)] if pool else []
    rows = reals + fakes
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


def cond_name(id_r: float, ood_r: float) -> str:
    return f"id{int(round(id_r * 100)):02d}_ood{int(round(ood_r * 100)):02d}"


def run(data_root, out_dir, cfg: BuildConfig):
    """Build all condition manifests. Reusable by the CLI and build_from_config."""
    root = Path(data_root)
    out = Path(out_dir)

    id_real = scan_class_dir(root / "id_real")
    ood_real = scan_flat_dir(root / "ood_real")
    fake_id = scan_class_dir(root / "fake_id")
    fake_ood = scan_flat_dir(root / "fake_ood")

    print(f"[scan] id_real classes={len(id_real)} imgs={sum(map(len, id_real.values()))}")
    print(f"[scan] ood_real imgs={len(ood_real)}")
    print(f"[scan] fake_id classes={len(fake_id)} imgs={sum(map(len, fake_id.values()))}")
    print(f"[scan] fake_ood imgs={len(fake_ood)}")
    if not id_real:
        raise SystemExit(f"No ID images under {root/'id_real'} (need <class>/*.jpg).")

    # class -> integer index (stable, sorted)
    classes = sorted(id_real.keys())
    cls2idx = {c: i for i, c in enumerate(classes)}
    (out).mkdir(parents=True, exist_ok=True)
    with open(out / "classes.json", "w") as f:
        json.dump({"classes": classes, "cls2idx": cls2idx}, f, indent=2)

    # Fixed splits: same clean id_test + fake_probe reused by every condition.
    base_rng = random.Random(cfg.seed)
    id_train, id_test = split_id(id_real, cfg, base_rng)

    fake_probe = list(fake_ood)
    base_rng.shuffle(fake_probe)
    if cfg.fake_probe_size:
        fake_probe = fake_probe[: cfg.fake_probe_size]

    def rows_id_test():
        return [dict(path=str(p), label=cls2idx[c], role="id", is_fake=0, split="test")
                for c, ps in id_test.items() for p in ps]

    def rows_fake_probe():
        return [dict(path=str(p), label=-1, role="ood", is_fake=1, split="test")
                for p in fake_probe]

    conditions = enumerate_conditions(cfg)
    print(f"[plan] {len(conditions)} conditions: "
          + ", ".join(cond_name(*c) for c in conditions))

    for id_r, ood_r in conditions:
        cdir = out / cond_name(id_r, ood_r)
        rng = random.Random(cfg.seed + int(id_r * 1000) * 131 + int(ood_r * 1000))

        train_rows = contaminate_id_train(id_train, fake_id, id_r, rng)
        ood_rows = build_ood_test(ood_real, fake_ood, ood_r, cfg.ood_test_size, rng)

        write_manifest(cdir / "id_train.csv", [
            dict(path=str(p), label=cls2idx[c], role="id", is_fake=fk, split="train")
            for p, c, fk in train_rows])
        write_manifest(cdir / "ood_test.csv", [
            dict(path=str(p), label=-1, role="ood", is_fake=fk, split="test")
            for p, fk in ood_rows])
        write_manifest(cdir / "id_test.csv", rows_id_test())
        write_manifest(cdir / "fake_probe.csv", rows_fake_probe())

        n_tr_fake = sum(fk for _, _, fk in train_rows)
        n_ood_fake = sum(fk for _, fk in ood_rows)
        print(f"[build] {cond_name(id_r, ood_r)}: "
              f"train={len(train_rows)} (fake={n_tr_fake}) "
              f"ood_test={len(ood_rows)} (fake={n_ood_fake}) "
              f"id_test={sum(map(len, id_test.values()))} fake_probe={len(fake_probe)}")

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
