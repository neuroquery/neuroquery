import numpy as np
from scipy import sparse

import pytest

from neuroquery import nmf


def test_smoothing_matrix_sqrt():
    rng = np.random.RandomState(0)
    X = rng.binomial(3, .3, size=(30, 7))
    V = nmf._smoothing_matrix_sqrt(X, n_components=5)
    assert V.shape == (7, 5)


def test_covariance_smoothing():
    rng = np.random.RandomState(0)
    X = rng.binomial(3, .3, size=(30, 7))
    op = nmf.CovarianceSmoothing(n_components=5).fit(X)
    smoothed = op.transform(X)
    assert smoothed.shape == X.shape
    assert np.allclose(smoothed, X, rtol=.2, atol=.2)
    assert np.allclose(
        1., np.linalg.norm(op.normalized_V_.dot(op.V_.T), axis=1))
    a = sparse.csr_matrix(((1.,), (0,), (0, 1)), shape=(1, 7))
    s = op.transform(a)
    assert np.allclose(
        s, a.A.ravel() * .9 + .1 * op.normalized_V_.dot(op.V_.T)[0])
