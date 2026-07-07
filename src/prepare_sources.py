#!/usr/bin/env python3
"""Materialize source pools into the data/ layout expected by build_datasets.py.

Currently handles the WHOOPS! dataset (human-curated physically-impossible /
common-sense-defying images) -> data/fake_ood/.

  WHOOPS!  Bitton-Guetta et al., ICCV 2023.  HF: nlphuji/whoops

For real ID / real OOD pools, see docs/DESIGN.md (ImageNet-200 recipe). This
script only automates the parts that have a clean programmatic source.

Example::

    python src/prepare_sources.py whoops --out data/fake_ood --limit 500
"""

from __future__ import annotations

import argparse
from pathlib import Path


def run_whoops(args):
    from datasets import load_dataset
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("nlphuji/whoops", split="test")
    n = min(args.limit, len(ds)) if args.limit else len(ds)
    for i in range(n):
        img = ds[i]["image"].convert("RGB")
        img.save(out / f"whoops_{i:04d}.png")
        if i % 50 == 0:
            print(f"[whoops] {i+1}/{n}")
    print(f"[done] {n} WHOOPS images -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="src", required=True)
    w = sub.add_parser("whoops")
    w.add_argument("--out", default="data/fake_ood")
    w.add_argument("--limit", type=int, default=500)
    w.set_defaults(func=run_whoops)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
