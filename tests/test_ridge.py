import tempfile

import numpy as np
from scipy import sparse

from sklearn import datasets
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

import pytest

from neuroquery import ridge

ALPHAS = np.logspace(-6, 6, 13)


@pytest.mark.parametrize("noise", [0, 1, 1000])
@pytest.mark.parametrize("n_samples", [11, 101, 301])
def test_ridge(noise, n_samples):
    x, y = datasets.make_regression(
        n_features=100,
        n_targets=33,
        effective_rank=4,
        bias=100,
        n_informative=7,
        shuffle=False,
        noise=noise,
        n_samples=n_samples,
        random_state=1,
    )
    x += 71
    y -= 370
    xx, yy = x.copy(), y.copy()
    sk_ridge = RidgeCV(alphas=ALPHAS).fit(x, y)
    reg = ridge.RidgeGCV(alphas=ALPHAS).fit(x, y)
    assert sk_ridge.alpha_ == reg.alpha_
    assert np.allclose(reg.coef_, sk_ridge.coef_, atol=1e-7)
    assert np.allclose(reg.intercept_, sk_ridge.intercept_)
    assert (x == xx).all()
    assert (y == yy).all()


def test_ridge_dtype():
    x, y = datasets.make_regression(
        n_features=3, n_targets=7, n_samples=6, random_state=1
    )
    x = np.asarray(x, dtype="float32")
    y = np.asarray(y, dtype="float32")
    reg = ridge.RidgeGCV(alphas=ALPHAS).fit(x, y)
    assert reg.coef_.dtype == "float32"
    y = np.asarray(y, dtype="float64")
    reg = ridge.RidgeGCV(alphas=ALPHAS).fit(x, y)
    assert reg.coef_.dtype == "float64"


@pytest.mark.parametrize(
    "regressor", [ridge.RidgeGCV, ridge.AdaptiveRidge, ridge.SelectiveRidge]
)
def test_z_maps(regressor):
    x, y = datasets.make_regression(
        n_features=39,
        n_targets=13,
        effective_rank=7,
        bias=10,
        n_informative=5,
        shuffle=False,
        noise=5,
        n_samples=57,
        random_state=1,
    )
    test_feat = [1, 3, 7]
    reg = regressor(store_M=True).fit(x, y)
    for feat in test_feat:
        w = sparse.csr_matrix(([1], ([0], [feat])), shape=(1, 39))
        z1 = reg.transform_to_z_maps(w)[0]
        z2 = reg.z_maps()[feat]
        assert np.allclose(z1, z2)
    if hasattr(reg, "selected_features_"):
        z = reg.z_maps(full=False)
        assert z.shape == (len(reg.selected_features_), 13)
    with tempfile.TemporaryDirectory() as tmp_dir:
        fitted = ridge.FittedLinearModel.from_model(reg)
        fitted.to_data_dir(tmp_dir)
        loaded = ridge.FittedLinearModel.from_data_dir(tmp_dir)
        assert np.allclose(loaded.z_maps(), reg.z_maps())
        assert np.allclose(
            loaded.transform_to_z_maps(x), reg.transform_to_z_maps(x)
        )
        assert np.allclose(loaded.predict(x), reg.predict(x))
        if hasattr(reg, "selected_features_"):
            z = loaded.z_maps(full=False)
            assert z.shape == (len(reg.selected_features_), 13)


@pytest.mark.parametrize(
    "regressor", [ridge.RidgeGCV, ridge.AdaptiveRidge, ridge.SelectiveRidge]
)
def test_predictions(regressor):
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
    reg = regressor()
    sk_reg = RidgeCV()
    score = cross_val_score(reg, x, y, cv=5)
    sk_score = cross_val_score(sk_reg, x, y, cv=5)
    if regressor == ridge.RidgeGCV:
        assert np.allclose(score, sk_score)
        return
    assert score.mean() > sk_score.mean()
    reg1 = regressor(
        use_positive_part=False, regularize_var=False, energy_order=1
    )
    score = cross_val_score(reg1, x, y, cv=5)
    assert score.mean() > sk_score.mean()
    reg.fit(x, y)
    reg1.fit(x, y)
    assert (reg1.feat_penalty_ != reg.feat_penalty_).mean() > 0.5


def test_fitted_linear_model_no_var():
    reg = ridge.FittedLinearModel(np.ones((3, 7)), np.ones(3))
    with pytest.raises(ValueError):
        reg.z_maps()
    with pytest.raises(ValueError):
        reg.prediction_variance(np.ones((2, 7)))
    with pytest.raises(ValueError):
        reg.transform_to_z_maps(np.ones((2, 7)))
    pred = reg.predict(np.zeros((2, 7)))
    assert np.allclose(pred, np.ones((2, 3)))
