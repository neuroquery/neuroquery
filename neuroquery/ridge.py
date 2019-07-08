import pathlib

import numpy as np

from scipy import linalg

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model import LinearRegression


_DEFAULT_ALPHAS = np.logspace(-1, 3, 9)


def _preprocess(X, Y, feat_penalty=None):
    new_X = np.empty((X.shape[0], X.shape[1] + 1), dtype=X.dtype)
    new_X[:, -1] = 1
    new_X[:, :-1] = X
    X_mean = new_X.mean(axis=0)[:-1]
    np.subtract(new_X[:, :-1], X_mean, out=new_X[:, :-1])
    if feat_penalty is not None:
        np.divide(new_X[:, :-1], np.sqrt(feat_penalty), out=new_X[:, :-1])
    intercept = Y.mean(axis=0)
    Y = Y - intercept
    return new_X, X_mean, Y, intercept


def _gcv(U, s, Y, intercept, alphas):
    intercept_dim = U.var(axis=0) < 1e-12
    errors = np.zeros((len(alphas), Y.shape[0]), dtype=Y.dtype)
    for i, alpha in enumerate(alphas):
        t = s ** 2 / (s ** 2 + alpha)
        t[intercept_dim] = 1
        hat = np.einsum("ij,...j,kj->ik", U, t, U, optimize=True)
        scale = 1 - np.diag(hat)
        batch_size = np.maximum(1, Y.shape[1] // 10)
        for batch in range(0, Y.shape[1], batch_size):
            target = Y[:, batch : batch + batch_size]
            err = (target - hat.dot(target)) / scale[:, None]
            errors[i] += np.sum(err ** 2, axis=1)
    return errors


def ridge_gcv(X, Y, alphas=_DEFAULT_ALPHAS, feat_penalty=None):
    X, X_mean, Y, intercept = _preprocess(X, Y, feat_penalty)
    U, s, Vh = linalg.svd(X, full_matrices=False)
    V = Vh.T
    intercept_dim = U.var(axis=0) < 1e-12
    errors = _gcv(U, s, Y, intercept, alphas)
    mean_error = errors.mean(axis=1)
    error_std = np.std(errors, axis=1) / np.sqrt(errors.shape[1])
    best_alpha = alphas[np.argmin(mean_error)]
    t = s / (s ** 2 + best_alpha)
    t[intercept_dim] = 1
    M = np.einsum("ij,...j,kj->ik", V, t, U, optimize=True)
    coef = M.dot(Y)
    M = M[:-1]
    fitted_intercept = coef[-1]
    coef = coef[:-1]
    if feat_penalty is not None:
        coef = np.divide(coef, np.sqrt(feat_penalty)[:, None], out=coef)
        if M is not None:
            M = np.divide(M, np.sqrt(feat_penalty)[:, None], out=M)
    intercept -= X_mean.dot(coef)
    intercept -= fitted_intercept
    return coef, intercept, best_alpha, mean_error, error_std, M


def _get_variance(X, Y, coef, intercept, M):
    res_var = np.mean((X.dot(coef) + intercept - Y) ** 2, axis=0)
    var_filter = np.einsum("ij,ij->i", M, M)
    return res_var, var_filter


class RidgeGCV(LinearRegression):
    def __init__(
        self, alphas=_DEFAULT_ALPHAS, feat_penalty=None, store_M=False
    ):
        self.alphas = alphas
        self.feat_penalty = feat_penalty
        self.store_M = store_M

    def _compute_feat_penalty(self, X, Y):
        if not hasattr(self, "feat_penalty_"):
            self.feat_penalty_ = self.feat_penalty

    def fit(self, X, Y):
        self._compute_feat_penalty(X, Y)
        (
            coef,
            self.intercept_,
            self.alpha_,
            self.loo_errors_,
            self.error_std_,
            M,
        ) = ridge_gcv(
            X, Y, alphas=self.alphas, feat_penalty=self.feat_penalty_
        )
        self.coef_ = coef.T
        if M is not None:
            self._res_var, self._var_filter = _get_variance(
                X, Y, self.coef_.T, self.intercept_, M
            )
        if self.store_M:
            self.M_ = M
        return self

    def z_energy(self, use_positive_part=True):
        var = np.outer(self._var_filter, self._res_var)
        z = self.coef_.T / np.sqrt(var + var.mean())
        if use_positive_part:
            np.clip(z, 0, None, out=z)
        return (z ** 2).mean(axis=1)

    def z_maps(self):
        var = np.outer(self._var_filter, self._res_var)
        z = self.coef_.T / np.maximum(np.sqrt(var), 1e-24)
        return z

    def prediction_variance(self, X):
        XM = np.atleast_2d(safe_sparse_dot(X, self.M_, dense_output=True))
        XMMtXt = (XM ** 2).sum(axis=1, keepdims=True)
        return XMMtXt * self._res_var

    def transform_to_z_maps(self, X):
        return safe_sparse_dot(
            X, self.coef_.T, dense_output=True
        ) / np.maximum(np.sqrt(self.prediction_variance(X)), 1e-24)


def _linreg_energy(X, Y, alphas, use_positive_part=True):
    ridge = RidgeGCV(alphas=alphas).fit(X, Y)
    return ridge.z_energy(use_positive_part=use_positive_part)


class AdaptiveRidge(RidgeGCV):
    def __init__(
        self, alphas=_DEFAULT_ALPHAS, use_positive_part=True, store_M=False
    ):
        super().__init__(alphas=alphas, store_M=store_M)
        self.use_positive_part = use_positive_part

    def _compute_feat_penalty(self, X, Y):
        z_energy = _linreg_energy(
            X, Y, self.alphas, use_positive_part=self.use_positive_part
        )
        self.feat_penalty_ = 1 / np.maximum(
            z_energy - (z_energy.mean() + 2 * z_energy.std()), 0.001
        )
        self.feat_penalty_ /= self.feat_penalty_.min()


class SelectiveRidge(RidgeGCV):
    def __init__(
        self, alphas=_DEFAULT_ALPHAS, use_positive_part=True, store_M=False
    ):
        super().__init__(alphas=alphas, store_M=store_M)
        self.use_positive_part = use_positive_part

    def fit(self, X, Y):
        self.original_n_features_ = X.shape[1]
        adapt = AdaptiveRidge(
            alphas=self.alphas, use_positive_part=self.use_positive_part
        ).fit(X, Y)
        self.selected_features_ = np.arange(X.shape[1])[
            adapt.feat_penalty_ < adapt.feat_penalty_.max()
        ]
        self.feat_penalty_ = adapt.feat_penalty_[self.selected_features_]
        del adapt
        print("keeping {} features".format(len(self.selected_features_)))
        super().fit(X[:, self.selected_features_], Y)
        return self

    def predict(self, X):
        X = X[:, self.selected_features_]
        return (
            safe_sparse_dot(X, self.coef_.T, dense_output=True)
            + self.intercept_
        )

    def z_maps(self, full=True):
        var = np.outer(self._var_filter, self._res_var)
        z = self.coef_.T / np.maximum(np.sqrt(var), 1e-24)
        if not full:
            return z
        full_z = np.zeros((self.original_n_features_, self.coef_.shape[0]))
        full_z[self.selected_features_] = z
        return full_z

    def prediction_variance(self, X):
        X = X[:, self.selected_features_]
        XM = np.atleast_2d(safe_sparse_dot(X, self.M_, dense_output=True))
        XMMtXt = (XM ** 2).sum(axis=1, keepdims=True)
        return XMMtXt * self._res_var

    def transform_to_z_maps(self, X):
        pred_variance = self.prediction_variance(X)
        X = X[:, self.selected_features_]
        return safe_sparse_dot(
            X, self.coef_.T, dense_output=True
        ) / np.maximum(np.sqrt(pred_variance), 1e-24)


class FittedLinearModel(RidgeGCV):
    @classmethod
    def from_data_dir(cls, model_dir):
        model_dir = pathlib.Path(model_dir)
        kwargs = {}
        for name in [
            "coef",
            "intercept",
            "M",
            "residual_var",
            "selected_features",
            "original_n_features",
        ]:
            if (model_dir / "{}.npy".format(name)).is_file():
                kwargs[name] = np.load(str(model_dir / "{}.npy".format(name)))
        return cls(**kwargs)

    @classmethod
    def from_model(cls, model):
        return cls(
            model.coef_,
            model.intercept_,
            model.M_,
            model._res_var,
            getattr(model, "selected_features_", None),
            getattr(model, "original_n_features_", None),
        )

    def to_data_dir(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "coef",
            "intercept",
            "M",
            "residual_var",
            "selected_features",
            "original_n_features",
        ]:
            if getattr(self, name, None) is not None:
                np.save(
                    str(model_dir / "{}.npy".format(name)), getattr(self, name)
                )

    def __init__(
        self,
        coef,
        intercept,
        M,
        residual_var,
        selected_features=None,
        original_n_features=None,
    ):
        self.coef = coef
        self.intercept = intercept
        self.M = M
        self.residual_var = residual_var
        self.selected_features = selected_features
        self.original_n_features = original_n_features
        self.fit()

    def fit(self, X=None, y=None):
        self.coef_ = self.coef
        self.intercept_ = self.intercept
        self.M_ = self.M
        self._res_var = self.residual_var
        if self.selected_features is not None:
            self.selected_features_ = self.selected_features
        else:
            self.selected_features_ = np.arange(self.coef_.shape[1])
        if self.original_n_features is None:
            self.original_n_features_ = self.coef_.shape[1]
        else:
            self.original_n_features_ = self.original_n_features
        self._var_filter = np.einsum("ij,ij->i", self.M_, self.M_)
        return self

    def predict(self, X):
        X = X[:, self.selected_features_]
        return (
            safe_sparse_dot(X, self.coef_.T, dense_output=True)
            + self.intercept_
        )

    def z_maps(self, full=True):
        var = np.outer(self._var_filter, self._res_var)
        z = self.coef_.T / np.maximum(np.sqrt(var), 1e-24)
        if not full:
            return z
        full_z = np.zeros((self.original_n_features_, self.coef_.shape[0]))
        full_z[self.selected_features_] = z
        return full_z

    def prediction_variance(self, X):
        X = X[:, self.selected_features_]
        XM = np.atleast_2d(safe_sparse_dot(X, self.M_, dense_output=True))
        XMMtXt = (XM ** 2).sum(axis=1, keepdims=True)
        return XMMtXt * self._res_var

    def transform_to_z_maps(self, X):
        pred_variance = self.prediction_variance(X)
        X = X[:, self.selected_features_]
        return safe_sparse_dot(
            X, self.coef_.T, dense_output=True
        ) / np.maximum(np.sqrt(pred_variance), 1e-24)
