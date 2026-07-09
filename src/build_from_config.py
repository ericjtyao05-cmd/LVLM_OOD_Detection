"""Translate experiment.yaml -> BuildConfig and run the (tested) mixing pipeline."""

from __future__ import annotations

import argparse

from .build_datasets import BuildConfig, run
from .config import load_config


def to_build_config(cfg: dict) -> BuildConfig:
    conds = cfg["conditions"]
    idc = cfg["data"]["id"]
    return BuildConfig(
        id_fake_ratios=conds["id_fake_ratios"],
        id_test_frac=idc.get("test_frac", 0.2),
        max_per_class=idc.get("max_per_class"),
        ood_test_size=conds.get("ood_test_size"),
        seed=cfg.get("seed", 0),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="manifests")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(args.data_root, args.out_dir, to_build_config(cfg))


if __name__ == "__main__":
    main()
