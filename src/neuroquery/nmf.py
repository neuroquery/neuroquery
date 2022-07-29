import pathlib

import numpy as np
from scipy import sparse

from sklearn.decomposition import NMF
from sklearn.base import BaseEstimator, TransformerMixin


def _smoothing_matrix_sqrt(X, n_components=300):
    nmf = NMF(
        init=None,
        n_components=n_components,
        max_iter=200,
        random_state=0,
        alpha=1e-1,
        l1_ratio=0.1,
        verbose=0,
    )
    u = nmf.fit_transform(X)
    v = nmf.components_.T
    z = np.linalg.norm(u, axis=0)
    v = z * v
    return v


class CovarianceSmoothing(BaseEstimator, TransformerMixin):
    @classmethod
    def from_data_dir(cls, model_dir):
        model_dir = pathlib.Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        V = np.load(str(model_dir / "V.npy"))
        sw = np.load(str(model_dir / "smoothing_weight.npy"))
        n_components = V.shape[1]
        model = cls(n_components, sw)
        model.V_ = V
        V_norm = np.linalg.norm(V.dot(V.T), axis=1)
        zero_norm = V_norm == 0
        V_norm[zero_norm] = 1
        model.normalized_V_ = model.V_ / V_norm[:, None]
        model.normalized_V_[zero_norm] = 0
        return model

    def to_data_dir(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        np.save(str(model_dir / "V.npy"), self.V_)
        np.save(str(model_dir / "smoothing_weight.npy"), self.smoothing_weight)

    def __init__(self, n_components=300, smoothing_weight=0.1):
        self.n_components = n_components
        self.smoothing_weight = smoothing_weight

    def fit(self, X):
        self.V_ = _smoothing_matrix_sqrt(X, n_components=self.n_components)
        V_norm = np.linalg.norm(self.V_.dot(self.V_.T), axis=1)
        zero_norm = V_norm == 0
        V_norm[zero_norm] = 1
        self.normalized_V_ = self.V_ / V_norm[:, None]
        self.normalized_V_[zero_norm] = 0
        return self

    def transform(self, X):
        if sparse.issparse(X):
            X = X.A
        s = np.einsum(
            "ij,jk,lk", X, self.normalized_V_, self.V_, optimize=True
        )
        return self.smoothing_weight * s + (1 - self.smoothing_weight) * X
