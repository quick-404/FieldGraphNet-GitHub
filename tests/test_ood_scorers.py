import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))

from ood_scorers import msp_scores, odin_scores, mahalanobis_scores, fit_mahalanobis


def test_msp_is_one_minus_max_softmax():
    p = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    out = msp_scores(p)
    assert np.allclose(out, [0.1, 0.5, 0.1])


def test_msp_higher_for_uncertain():
    confident = np.array([[0.99, 0.01]])
    unsure = np.array([[0.51, 0.49]])
    assert msp_scores(unsure)[0] > msp_scores(confident)[0]


def test_odin_matches_msp_when_temperature_is_one():
    logits = np.array([[2.0, 1.0], [0.1, 0.2]])
    assert np.allclose(odin_scores(logits, temp=1.0), msp_scores(
        np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)))


def test_odin_high_temperature_shrinks_toward_uniform():
    logits = np.array([[10.0, 0.0]])
    cold = odin_scores(logits, temp=1.0)[0]
    hot = odin_scores(logits, temp=1000.0)[0]
    assert hot > cold, 'higher temperature must look less confident, hence more OOD'


def test_mahalanobis_zero_at_class_mean():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4) * 0.5
    y = np.array([0] * 100 + [1] * 100)
    X[100:] += 3.0
    mu, prec = fit_mahalanobis(X, y, n_classes=2)
    d = mahalanobis_scores(X, mu, prec)
    assert d.shape == (200,)
    assert d[:100].mean() < d[100:].mean(), 'own-class points must score lower'
    # the score at an exact class mean must be bit-exact zero
    assert np.allclose(mahalanobis_scores(mu, mu, prec), 0.0, atol=1e-12)
    # and it must actually depend on the precision matrix, not just on mu
    d_other = mahalanobis_scores(X, mu, np.eye(4))
    assert not np.allclose(d, d_other), 'scores must depend on the precision matrix'


def test_mahalanobis_uses_nearest_class():
    mu = np.array([[0.0, 0.0], [10.0, 10.0]])
    prec = np.eye(2)
    X = np.array([[0.0, 0.0], [10.0, 10.0], [100.0, 100.0]])
    d = mahalanobis_scores(X, mu, prec)
    assert d[0] < 1e-6 and d[1] < 1e-6
    assert d[2] > d[0] and d[2] > d[1]
