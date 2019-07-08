import numpy as np

from neuroquery import nmf


def test_smoothing_matrix_sqrt():
    rng = np.random.RandomState(0)
    x = rng.binomial(3, .3, size=(30, 7))
    v = nmf.smoothing_matrix_sqrt(x, n_components=5)
    assert v.shape == (7, 5)
