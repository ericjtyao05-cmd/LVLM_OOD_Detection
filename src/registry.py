"""Tiny registry so models and detectors are swappable from experiment.yaml.

Add a model:   @register_model("my-vlm")   on a class implementing VLMBackbone.
Add a detector:@register_detector("my-score") on a fn(id_stats, feats, logits)->scores.

The pipeline only ever looks things up by string key, so switching model/detector
never touches orchestration code -- only configs/experiment.yaml.
"""

from __future__ import annotations

from typing import Callable

MODEL_REGISTRY: dict[str, type] = {}
DETECTOR_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    def deco(cls):
        if name in MODEL_REGISTRY:
            raise KeyError(f"model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


def register_detector(name: str):
    def deco(fn):
        if name in DETECTOR_REGISTRY:
            raise KeyError(f"detector '{name}' already registered")
        DETECTOR_REGISTRY[name] = fn
        return fn
    return deco


def get_model(name: str, **kw):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'. registered: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kw)


def get_detector(name: str) -> Callable:
    if name not in DETECTOR_REGISTRY:
        raise KeyError(f"unknown detector '{name}'. registered: {list(DETECTOR_REGISTRY)}")
    return DETECTOR_REGISTRY[name]


# ----------------------------------------------------------------------------
# Interface contracts (documentation; not enforced)
# ----------------------------------------------------------------------------
class VLMBackbone:
    """A model wrapper must expose:

        feat_dim: int
        def extract(self, images: list[PIL.Image]) -> dict:
            returns {"hidden": Tensor[N, D],      # last-token hidden state
                     "logits": Tensor[N, C]}      # class-restricted next-token logits
    Implementations live in src/models/ and self-register via @register_model.
    """
    feat_dim: int

    def extract(self, images):  # pragma: no cover - interface only
        raise NotImplementedError
