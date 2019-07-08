import numpy as np

from sklearn import datasets
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

from neuroquery import smoothed_regression


def test_predictions():
    x, y = datasets.make_regression(
        n_samples=102,
        n_informative=5,
        n_features=91,
        n_targets=117,
        effective_rank=9,
        noise=0.5,
        shuffle=False,
        random_state=0,
    )
    x -= x.min() - 1
    reg = smoothed_regression.SmoothedRegression(
        n_components=5, smoothing_weight=1e-3)
    print(reg.fit(x, y).predict(x))
    sk_reg = RidgeCV()
    score = cross_val_score(reg, x, y, cv=5)
    sk_score = cross_val_score(sk_reg, x, y, cv=5)
    assert score.mean() > sk_score.mean()


def test_z_maps():
    rng = np.random.RandomState(0)
    X = rng.binomial(3, .3, size=(21, 9)).astype('float64')
    Y = rng.randn(21, 11)
    reg = smoothed_regression.SmoothedRegression(n_components=5).fit(X, Y)
    z = reg.transform_to_z_maps(X)
    assert z.shape == Y.shape
