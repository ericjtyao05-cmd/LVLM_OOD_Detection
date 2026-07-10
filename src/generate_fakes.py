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

# Per-class-group templates, chosen from SDXL sample tests (see docs/DESIGN.md):
#  - MAMMALS: attached wings render cleanly -> winged mammal (+ giant for variety).
#  - BIRDS/FISH: wings fail (a bird with wings is a bird; a fish becomes a bird),
#    so use giant-with-scale -- a building-sized eagle/goldfish is unmistakable.
#  - GROUND OBJECTS: feathered wings attach (creature/insect wings spawn a
#    separate animal); some seeds still spawn a bird -> mild noise.
#  - AIRLINER: already flies + has wings, so wings->bird and giant doesn't help;
#    use inverse scale (a car-sized airliner among real cars) -- plane stays
#    recognisable in an impossible context.
MAMMALS = {"tabby_cat", "labrador_retriever", "african_elephant", "zebra", "tiger", "brown_bear"}
BIG_ANIMALS = {"goldfish", "bald_eagle", "ostrich"}
WINGED_OBJECTS = {"sports_car", "school_bus", "mountain_bike", "grand_piano", "steam_locomotive"}
AIRLINERS = {"airliner"}

MAMMAL_TEMPLATES = [
    "a {c} with enormous feathered eagle wings, flying high above the city",
    "a {c} with large black leathery bat wings, flying at dusk",
    "a {c} with huge colorful monarch butterfly wings, in a sunny garden",
    "a {c} with two pairs of large feathered wings, four wings total, flying above the rooftops",
    "a giant {c} looming over tiny cars and terrified people on a city street, immense scale",
]
GIANT_TEMPLATES = [
    "a giant {c} looming over tiny cars and terrified people on a city street, immense scale",
    "a colossal {c} as tall as the buildings, tiny people fleeing at its feet in a city",
    "an enormous {c} towering over a highway full of tiny cars, people running in fear",
    "a gigantic {c} in a city square, dwarfing the tiny buses and pedestrians around it",
    "a titanic {c} standing over a town, tiny houses and people below for scale",
]
OBJECT_TEMPLATES = [
    "a {c} with enormous brown feathered wings, flying high over the city",
    "a {c} with huge white feathered angel wings, flying above the clouds",
    "a {c} with large black feathered wings, flying through a stormy night sky",
    "a {c} with golden feathered wings, flying over snowy mountains at sunrise",
    "a {c} with many feathered wings along its sides, flying through a bright blue sky",
]
AIRLINER_TEMPLATES = [
    "a real passenger airliner shrunk to the size of a small car, parked on a city street between real cars, people standing beside it",
    "a tiny passenger airliner the size of a car among normal traffic on a busy city street",
    "a miniature real airliner the size of a van, parked in a city plaza with people walking around it",
    "a shrunken passenger jet the size of a car, sitting at a city bus stop next to a bus and pedestrians",
    "a real airliner shrunk down to the size of a car, driving on a highway among normal cars",
]


def _templates_for(cls):
    if cls in BIG_ANIMALS:     return GIANT_TEMPLATES
    if cls in AIRLINERS:       return AIRLINER_TEMPLATES
    if cls in WINGED_OBJECTS:  return OBJECT_TEMPLATES
    return MAMMAL_TEMPLATES

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
        templates = _templates_for(cls)
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
