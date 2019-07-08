import tempfile

import numpy as np
from scipy import sparse

from neuroquery import nmf


def test_smoothing_matrix_sqrt():
    rng = np.random.RandomState(0)
    X = rng.binomial(3, 0.3, size=(30, 7))
    V = nmf._smoothing_matrix_sqrt(X, n_components=5)
    assert V.shape == (7, 5)


def test_covariance_smoothing():
    rng = np.random.RandomState(0)
    X = rng.binomial(3, 0.3, size=(30, 7))
    op = nmf.CovarianceSmoothing(n_components=5).fit(X)
    smoothed = op.transform(X)
    assert smoothed.shape == X.shape
    assert np.allclose(smoothed, X, rtol=0.2, atol=0.2)
    assert np.allclose(
        1.0, np.linalg.norm(op.normalized_V_.dot(op.V_.T), axis=1)
    )
    a = sparse.csr_matrix(((1.0,), (0,), (0, 1)), shape=(1, 7))
    s = op.transform(a)
    assert np.allclose(
        s, a.A.ravel() * 0.9 + 0.1 * op.normalized_V_.dot(op.V_.T)[0]
    )
    smoothed = nmf.CovarianceSmoothing(
        n_components=5, smoothing_weight=0.0
    ).fit_transform(X)
    assert np.allclose(smoothed, X)
    with tempfile.TemporaryDirectory() as tmp_dir:
        op.to_data_dir(tmp_dir)
        loaded = nmf.CovarianceSmoothing.from_data_dir(tmp_dir)
        assert np.allclose(loaded.transform(X), op.transform(X))
