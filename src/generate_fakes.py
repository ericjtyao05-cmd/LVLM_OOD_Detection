#!/usr/bin/env python3
"""Generate category-aligned *fake* (physically-impossible) images with SDXL.

Produces fakes for the `fake_id/<class>/` and `fake_ood/` pools consumed by
build_datasets.py. Run this on a GPU (see docs/vast_ai_guide.md).

Two prompt strategies:
  * aligned  : for each ID class, render impossible variants  ->  fake_id/<class>/
               e.g. class "cat"  -> "a photo of a cat with large feathered wings"
  * freeform : render generic impossible scenes                ->  fake_ood/
               e.g. "Godzilla walking through a modern city, photorealistic"

You still want WHOOPS! (real human-curated impossible images) as an anchor set;
see docs/DESIGN.md. This script is for *scale* and *category alignment*.

Example::

    python src/generate_fakes.py aligned \
        --classes cat dog bird horse \
        --n-per-class 60 --out data/fake_id

    python src/generate_fakes.py freeform \
        --n 500 --out data/fake_ood
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

# Impossible-transformation templates for a known object class.
ALIGNED_TEMPLATES = [
    "a photo of a {c} with large feathered wings, photorealistic",
    "a photo of a {c} with a transparent glass body, studio lighting",
    "a photo of a {c} floating upside down in mid-air, defying gravity",
    "a photo of a giant {c} taller than a building in a city street",
    "a photo of a {c} made entirely of water, photorealistic",
    "a photo of a two-headed {c}, natural lighting",
    "a photo of a {c} on fire but completely unharmed, photorealistic",
]

# Freeform impossible / physics-violating scenes (class-agnostic OOD fakes).
FREEFORM_PROMPTS = [
    "Godzilla walking through a modern city, photorealistic",
    "a waterfall flowing upward into the sky, photorealistic",
    "a floating island held up by nothing in a blue sky, photorealistic",
    "a person casting no shadow under bright noon sun, photorealistic",
    "a melting clock draped over a tree branch, photorealistic",
    "a staircase that loops back into itself, impossible geometry, photoreal",
    "an elephant balancing on a single soap bubble, photorealistic",
    "a river of glowing lava frozen mid-splash, photorealistic",
    "a car with square wheels driving on a highway, photorealistic",
    "a candle burning underwater with a bright flame, photorealistic",
]

NEG = "cartoon, drawing, illustration, low quality, blurry, watermark, text"


def _pipe(model_id: str, steps: int):
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe, steps


def _gen(pipe, steps, prompt, seed):
    import torch
    g = torch.Generator("cuda").manual_seed(seed)
    return pipe(prompt=prompt, negative_prompt=NEG,
                num_inference_steps=steps, guidance_scale=6.5, generator=g).images[0]


def run_aligned(args):
    pipe, steps = _pipe(args.model, args.steps)
    rng = random.Random(args.seed)
    for cls in args.classes:
        outdir = Path(args.out) / cls
        outdir.mkdir(parents=True, exist_ok=True)
        for i in range(args.n_per_class):
            tmpl = ALIGNED_TEMPLATES[i % len(ALIGNED_TEMPLATES)]
            prompt = tmpl.format(c=cls)
            img = _gen(pipe, steps, prompt, rng.randint(0, 2**31))
            img.save(outdir / f"{cls}_{i:04d}.png")
            if i % 10 == 0:
                print(f"[aligned] {cls} {i+1}/{args.n_per_class}")
    print("[done] aligned fakes ->", args.out)


def run_freeform(args):
    pipe, steps = _pipe(args.model, args.steps)
    rng = random.Random(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    prompts = itertools.cycle(FREEFORM_PROMPTS)
    for i in range(args.n):
        prompt = next(prompts)
        img = _gen(pipe, steps, prompt, rng.randint(0, 2**31))
        img.save(outdir / f"fake_{i:04d}.png")
        if i % 20 == 0:
            print(f"[freeform] {i+1}/{args.n}")
    print("[done] freeform fakes ->", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("aligned")
    a.add_argument("--classes", nargs="+", required=True)
    a.add_argument("--n-per-class", type=int, default=60)
    a.add_argument("--out", default="data/fake_id")
    a.set_defaults(func=run_aligned)

    f = sub.add_parser("freeform")
    f.add_argument("--n", type=int, default=500)
    f.add_argument("--out", default="data/fake_ood")
    f.set_defaults(func=run_freeform)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
