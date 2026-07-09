#!/usr/bin/env python3
"""Generate category-aligned *fake* (physically-impossible) contaminants with SDXL.

Produces the `fake_id/<class>/` pool injected into id_train by build_datasets.py.
Run on a GPU (see docs/vast_ai_guide.md).

Design principle (matches the study's guardrail): each image must be
**unambiguously impossible** -- no real photo, sculpture, forced perspective,
costume, or VFX could produce it -- while staying **recognisable as its class**
(a winged *cat*, a *school bus* on chicken legs). Ambiguous "impossibilities"
that can occur in reality (on fire, made of glass, floating, melting) are
deliberately excluded: they'd pollute the fake pool with real-looking images.
Render style is straight photoreal; only the *content* is impossible.

Templates are split by class type -- absurd anatomy for animals, absurd
appendages/scale for objects -- so the impossibility stays coherent.

Example::

    python src/generate_fakes.py aligned \
        --classes tabby_cat sports_car ... \
        --n-per-class 240 --batch-size 8 --out data/fake_id
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

# Classes rendered with the ANIMAL template set; everything else -> OBJECT set.
ANIMAL_CLASSES = {
    "tabby_cat", "labrador_retriever", "goldfish", "bald_eagle",
    "african_elephant", "zebra", "tiger", "brown_bear", "ostrich",
}

# Unambiguously-impossible transformations, kept photoreal.
ANIMAL_TEMPLATES = [
    "a {c} with enormous feathered wings, flying high above the rooftops",
    "a {c} with five heads, every head clearly visible",
    "a {c} with eight long spider legs",
    "a {c} the size of a skyscraper towering over a city, tiny people fleeing below",
    "a {c} with a dozen tails fanned out",
    "a {c} with a long elephant trunk and giant bat wings",
]
OBJECT_TEMPLATES = [
    "a {c} with giant feathered wings, flapping as it flies over the city",
    "a {c} walking on four giant chicken legs instead of wheels",
    "a {c} the size of a skyscraper towering over a tiny city, people fleeing",
    "a {c} with giant crab claws and octopus tentacles sprouting from its sides",
    "a {c} with dozens of human legs underneath, walking down the street",
    "a {c} covered in hundreds of blinking eyes",
]

# Appended to every prompt for photoreal quality.
QUALITY = (", photorealistic, highly detailed, sharp focus, natural lighting, "
           "professional photography, shot on DSLR, 4k")

# Negative prompt -- style/quality only. Deliberately does NOT negate extra
# limbs/heads/eyes: those are the target impossibilities.
NEG = ("cartoon, anime, illustration, painting, drawing, sketch, cgi, 3d render, "
       "low quality, blurry, jpeg artifacts, watermark, text, signature, border, frame")

# Freeform pool is unused by the current (contamination) design; kept for the
# reserved OOD-fake controls. See docs/DESIGN.md.
FREEFORM_PROMPTS = [
    "a housecat the size of a skyscraper towering over a city, photorealistic",
    "a car with giant feathered wings flying above a highway, photorealistic",
    "a fish walking on four long legs down a city street, photorealistic",
    "a bus with dozens of human legs walking down the road, photorealistic",
]


def _pipe(model_id: str, steps: int):
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_vae_slicing()   # cheaper VAE decode for batched generation
    return pipe, steps


def _gen_batch(pipe, steps, prompts, seeds, guidance):
    """Generate len(prompts) images in one batched call (reproducible per seed)."""
    import torch
    gens = [torch.Generator("cuda").manual_seed(s) for s in seeds]
    return pipe(prompt=list(prompts), negative_prompt=[NEG] * len(prompts),
                num_inference_steps=steps, guidance_scale=guidance,
                generator=gens).images


def run_aligned(args):
    pipe, steps = _pipe(args.model, args.steps)
    rng = random.Random(args.seed)
    for cls in args.classes:
        outdir = Path(args.out) / cls           # folder keeps underscores
        outdir.mkdir(parents=True, exist_ok=True)
        readable = cls.replace("_", " ")        # prompt says 'tabby cat'
        templates = ANIMAL_TEMPLATES if cls in ANIMAL_CLASSES else OBJECT_TEMPLATES
        jobs = [(templates[i % len(templates)].format(c=readable) + QUALITY,
                 rng.randint(0, 2**31)) for i in range(args.n_per_class)]
        idx = 0
        for b in range(0, len(jobs), args.batch_size):
            chunk = jobs[b:b + args.batch_size]
            imgs = _gen_batch(pipe, steps, [p for p, _ in chunk],
                              [s for _, s in chunk], args.guidance)
            for img in imgs:
                img.save(outdir / f"{cls}_{idx:04d}.png"); idx += 1
            print(f"[aligned] {cls} {idx}/{args.n_per_class}")
    print("[done] aligned fakes ->", args.out)


def run_freeform(args):
    pipe, steps = _pipe(args.model, args.steps)
    rng = random.Random(args.seed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    prompts = itertools.cycle(FREEFORM_PROMPTS)
    jobs = [(next(prompts) + QUALITY, rng.randint(0, 2**31)) for _ in range(args.n)]
    idx = 0
    for b in range(0, len(jobs), args.batch_size):
        chunk = jobs[b:b + args.batch_size]
        imgs = _gen_batch(pipe, steps, [p for p, _ in chunk],
                          [s for _, s in chunk], args.guidance)
        for img in imgs:
            img.save(outdir / f"fake_{idx:04d}.png"); idx += 1
        print(f"[freeform] {idx}/{args.n}")
    print("[done] freeform fakes ->", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("aligned")
    a.add_argument("--classes", nargs="+", required=True)
    a.add_argument("--n-per-class", type=int, default=240)
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
