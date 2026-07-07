"""OOD metrics. Convention: higher score == more OOD; OOD is the positive class."""

from __future__ import annotations

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties averaged. Avoids a scipy dependency."""
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """P(score_ood > score_id), via Mann-Whitney U. Higher score == more OOD."""
    id_scores = np.asarray(id_scores, float)
    ood_scores = np.asarray(ood_scores, float)
    n1, n0 = ood_scores.size, id_scores.size
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([ood_scores, id_scores]))
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray,
               tpr_target: float = 0.95) -> float:
    """FPR (ID wrongly flagged OOD) at the threshold recalling tpr_target of OOD."""
    id_scores = np.asarray(id_scores, float)
    ood_scores = np.asarray(ood_scores, float)
    if id_scores.size == 0 or ood_scores.size == 0:
        return float("nan")
    thr = np.quantile(ood_scores, 1 - tpr_target)
    return float((id_scores >= thr).mean())


def summarize(id_scores, ood_scores) -> dict:
    return {"auroc": auroc(id_scores, ood_scores),
            "fpr95": fpr_at_tpr(id_scores, ood_scores)}
