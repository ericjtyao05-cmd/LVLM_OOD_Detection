"""Post-hoc OOD detectors over an LVLM's internals.

Each detector is a function  fn(logits, hidden, stats) -> np.ndarray  where the
returned score is **higher for more-OOD** samples.

* logits : [N, C] class-restricted next-token logits (from the LVLM)
* hidden : [N, D] last-token hidden state (from the LVLM)
* stats  : dict produced by fit_stats() on ID-train hidden states (Mahalanobis
           needs it; logit-only scores ignore it)

Add a new score by writing a function and decorating it with
@register_detector("name") — nothing else in the pipeline changes.
"""

from __future__ import annotations

import numpy as np

from .registry import register_detector


# --------------------------------------------------------------------------- #
# Fit step (only Mahalanobis needs training statistics)
# --------------------------------------------------------------------------- #
def fit_stats(hidden_train: np.ndarray, labels_train: np.ndarray,
              n_classes: int, shrinkage: float = 1e-3) -> dict:
    """Class means + a single tied, shrinkage-regularized precision matrix.

    D (hidden dim, ~4096 for LLaVA) usually exceeds the sample count, so the raw
    covariance is singular. We use a tied covariance (pooled within-class) plus
    diagonal shrinkage toward the average variance — the standard robust fix.
    """
    hidden_train = np.asarray(hidden_train, dtype=np.float64)
    labels_train = np.asarray(labels_train)
    D = hidden_train.shape[1]
    means = np.zeros((n_classes, D))
    centered = np.empty_like(hidden_train)
    for c in range(n_classes):
        m = labels_train == c
        if m.sum() == 0:
            continue
        means[c] = hidden_train[m].mean(0)
        centered[m] = hidden_train[m] - means[c]
    cov = centered.T @ centered / max(len(hidden_train) - n_classes, 1)
    # shrinkage: (1-a)cov + a*mean(diag)*I  -> guarantees invertibility
    tr = np.trace(cov) / D
    cov = (1 - shrinkage) * cov + shrinkage * tr * np.eye(D)
    precision = np.linalg.inv(cov)
    return {"means": means, "precision": precision, "n_classes": n_classes}


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


@register_detector("msp")
def msp(logits, hidden, stats):
    """Maximum Softmax Probability (Hendrycks & Gimpel 2017). OOD = -max p."""
    return -_softmax(np.asarray(logits, float)).max(axis=1)


@register_detector("energy")
def energy(logits, hidden, stats, T: float = 1.0):
    """Energy score (Liu et al. 2020). OOD = -logsumexp(logits)."""
    logits = np.asarray(logits, float)
    m = logits.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp((logits - m) / T).sum(axis=1, keepdims=True) + 1e-12)).squeeze(1)
    return -T * lse


@register_detector("mahalanobis")
def mahalanobis(logits, hidden, stats):
    """Min class-conditional Mahalanobis distance on hidden states (Lee et al. 2018).

    OOD = min_c (h-mu_c)^T P (h-mu_c). Larger distance == more OOD.
    """
    if stats is None:
        raise ValueError("mahalanobis needs stats from fit_stats()")
    H = np.asarray(hidden, float)
    means, P = stats["means"], stats["precision"]
    dists = np.empty((H.shape[0], stats["n_classes"]))
    for c in range(stats["n_classes"]):
        d = H - means[c]
        dists[:, c] = np.einsum("nd,de,ne->n", d, P, d)
    return dists.min(axis=1)
