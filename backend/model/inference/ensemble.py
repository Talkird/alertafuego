"""Combine several models' fire probabilities into one detection score.

Each model has its own thresholds, chosen on val scenes for the same false-alarm
budget, and they sit far apart (b-shift 0.005 by day, d-noleak 0.995), so a raw
probability means nothing across models: 0.01 is a confident b-shift detection
and a d-noleak non-detection. The score measures how far a pixel is above each
model's own threshold instead, and takes the model that is surest.
"""

import numpy as np

#: A pixel is a detection where its score reaches this, i.e. where some model's
#: probability reaches that model's threshold.
SCORE_THRESHOLD = 0.5

#: Probabilities are clipped this far from 0 and 1 before taking logits.
_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


def detection_score(probabilities: list[np.ndarray], thresholds: list[np.ndarray]) -> np.ndarray:
    """Per pixel, the sigmoid of the largest margin, in logits, of a model's
    probability over its own threshold (probabilities[i] against thresholds[i],
    arrays or numbers). Exactly SCORE_THRESHOLD at a model's threshold, and rising as
    any model is surer. With one model it ranks pixels as its probability does."""
    margin = np.max([_logit(p) - _logit(t) for p, t in zip(probabilities, thresholds)], axis=0)
    return (1 / (1 + np.exp(-margin))).astype(np.float32)
