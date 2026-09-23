# -*- coding: utf-8 -*-
"""Confidence- and distance-based OOD scorers for the main-line comparison.

All three attach to the SAME base model (the GNN4ID attack classifier), so the
comparison does not hand any method a stronger engine.

  MSP          Hendrycks & Gimpel 2017 : score = 1 - max_c P(c|x)
  ODIN         Liang et al. 2018      : temperature-scaled logits, then the MSP rule
  Mahalanobis  Lee et al. 2018        : distance to the nearest class-conditional
                                        Gaussian fitted on in-domain training data

`scaled_softmax` and `msp_scores` follow experiments/eval_msp_odin.py so the two
studies remain comparable.
"""
import numpy as np


def scaled_softmax(logits, temp=1.0):
    """Temperature-scaled softmax over classes (rows)."""
    if not np.isfinite(temp) or temp <= 0:
        raise ValueError(f'temp must be a positive finite number, got {temp!r}')
    z = np.asarray(logits, dtype=np.float64) / float(temp)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def msp_scores(p_class):
    """MSP OOD score: 1 - max_c P(c|x). Higher = more OOD."""
    p = np.asarray(p_class, dtype=np.float64)
    return 1.0 - p.max(axis=1)


def odin_scores(logits, temp=1000.0):
    """ODIN OOD score: MSP computed on temperature-scaled logits."""
    return msp_scores(scaled_softmax(logits, temp))


def fit_mahalanobis(X, y, n_classes):
    """Class-conditional Gaussians with a shared (pooled) covariance.

    Returns (mu, precision) where mu is [n_classes, D] and precision is the
    inverse of the pooled within-class covariance.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    D = X.shape[1]
    mu = np.zeros((n_classes, D))
    centered = np.zeros_like(X)
    for c in range(n_classes):
        m = y == c
        if not m.any():
            raise ValueError(f'class {c} has no training samples')
        mu[c] = X[m].mean(axis=0)
        centered[m] = X[m] - mu[c]
    if (y < 0).any() or (y >= n_classes).any():
        raise ValueError(
            f'labels must be contiguous integers in [0, {n_classes}); '
            f'got range [{y.min()}, {y.max()}]')
    cov = centered.T @ centered / max(len(X) - n_classes, 1)
    # ridge for numerical stability when D is large relative to n
    cov += np.eye(D) * 1e-6
    return mu, np.linalg.inv(cov)


def mahalanobis_scores(X, mu, precision):
    """Squared distance to the NEAREST class-conditional Gaussian. Higher = more OOD.

    Unlike scipy.spatial.distance.mahalanobis this is the squared distance
    (no square root is taken), so the two conventions differ by a square.
    """
    X = np.asarray(X, dtype=np.float64)
    # (N, C, D)
    diff = X[:, None, :] - np.asarray(mu)[None, :, :]
    # quadratic form per class
    d = np.einsum('ncd,de,nce->nc', diff, np.asarray(precision), diff)
    return d.min(axis=1)
