"""Materialize real ID (ImageNet-1k, gated) + real OOD (DTD) + WHOOPS into data/.

  data/id_real/<class>/*.jpg   real ID  (ImageNet-1k via HF_TOKEN)
  data/ood_real/*.jpg          real OOD (Describable Textures / DTD)
  data/fake_ood/*.jpg          WHOOPS!  (human-curated impossible images)

SDXL fakes (data/fake_id, extra data/fake_ood) come from generate_fakes.py.

ImageNet class selection is done by matching a *distinctive substring* of the HF
label name (robust to whatever integer index HF assigns), so we never hardcode
class indices. We stream the split and bucket-fill each class up to max_per_class
in a single pass, with a scan cap so a fresh box can't run unbounded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config

# yaml class-name -> distinctive substring present in exactly one ImageNet label
IMAGENET_KEYWORDS = {
    "tabby_cat": "tabby",
    "labrador_retriever": "Labrador retriever",
    "goldfish": "goldfish",
    "bald_eagle": "bald eagle",
    "african_elephant": "African elephant",
    "zebra": "zebra",
    "tiger": "Panthera tigris",          # avoids tiger cat/shark/beetle
    "brown_bear": "brown bear",
    "ostrich": "ostrich",
    "sports_car": "sports car",
    "school_bus": "school bus",
    "airliner": "airliner",
    "mountain_bike": "mountain bike",
    "grand_piano": "grand piano",
    "steam_locomotive": "steam locomotive",
}


def _resolve_indices(label_names, wanted):
    """wanted: {yaml_name: keyword} -> {label_index: yaml_name}, asserting uniqueness."""
    idx_to_name = {}
    for yaml_name, kw in wanted.items():
        hits = [i for i, n in enumerate(label_names) if kw.lower() in n.lower()]
        if len(hits) != 1:
            raise ValueError(f"keyword '{kw}' for {yaml_name} matched {len(hits)} "
                             f"labels (need exactly 1): {[label_names[i] for i in hits]}")
        idx_to_name[hits[0]] = yaml_name
        print(f"[imagenet] {yaml_name:20s} -> idx {hits[0]:4d}  ({label_names[hits[0]]})")
    return idx_to_name


def prepare_imagenet(cfg, out_root: Path, max_scan: int):
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (ImageNet-1k is gated).")

    idc = cfg["data"]["id"]
    split = idc.get("split", "train")
    per_class = idc.get("max_per_class", 300)
    wanted = {c["name"]: IMAGENET_KEYWORDS[c["name"]] for c in idc["classes"]}

    # peek label names (non-streaming features are cheap)
    feats = load_dataset("imagenet-1k", split=split, streaming=True, token=token)
    label_names = feats.features["label"].names
    idx_to_name = _resolve_indices(label_names, wanted)

    buckets = {n: 0 for n in wanted}
    for n in wanted:
        (out_root / "id_real" / n).mkdir(parents=True, exist_ok=True)

    print(f"[imagenet] streaming '{split}', {per_class}/class, scan cap {max_scan}")
    for i, ex in enumerate(feats):
        if i >= max_scan:
            print(f"[imagenet] hit scan cap at {i}"); break
        lbl = ex["label"]
        if lbl not in idx_to_name:
            continue
        name = idx_to_name[lbl]
        if buckets[name] >= per_class:
            continue
        ex["image"].convert("RGB").save(
            out_root / "id_real" / name / f"{name}_{buckets[name]:04d}.jpg")
        buckets[name] += 1
        if all(v >= per_class for v in buckets.values()):
            print(f"[imagenet] all buckets full at scan {i}"); break
    print("[imagenet] collected:", {k: v for k, v in buckets.items()})


def prepare_dtd(cfg, out_root: Path):
    """Real OOD = Describable Textures (DTD), pulled from a fast HF mirror.

    torchvision's DTD downloads from thor.robots.ox.ac.uk which is heavily
    throttled (~20 KB/s); the HF mirror is orders of magnitude faster.
    """
    from datasets import load_dataset
    size = cfg["data"]["ood_real"].get("size", 3000)
    dst = out_root / "ood_real"; dst.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("tanganke/dtd", split="train")   # 3760 imgs, cols image/label
    n = min(size, len(ds))
    for i in range(n):
        ds[i]["image"].convert("RGB").save(dst / f"dtd_{i:04d}.jpg")
    print(f"[dtd] {n} texture images (HF tanganke/dtd) -> {dst}")


def prepare_whoops(cfg, out_root: Path):
    from datasets import load_dataset
    w = cfg["data"]["fake"].get("whoops", {})
    if not w.get("enabled", True):
        return
    dst = out_root / "fake_ood"; dst.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("nlphuji/whoops", split="test")
    n = min(w.get("limit", 500), len(ds))
    for i in range(n):
        ds[i]["image"].convert("RGB").save(dst / f"whoops_{i:04d}.jpg")
    print(f"[whoops] {n} images -> {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--max-scan", type=int, default=200_000,
                    help="max ImageNet examples to stream before giving up")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["imagenet", "dtd", "whoops"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.data_root)
    if "imagenet" not in args.skip:
        prepare_imagenet(cfg, out, args.max_scan)
    if "dtd" not in args.skip:
        prepare_dtd(cfg, out)
    if "whoops" not in args.skip:
        prepare_whoops(cfg, out)
    print("[done] data prepared under", out)


if __name__ == "__main__":
    main()
