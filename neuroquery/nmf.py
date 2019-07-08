import numpy as np
from scipy import sparse

from sklearn.decomposition import NMF


def _smoothing_matrix_sqrt(X, n_components=300):
    nmf = NMF(
        n_components=n_components, max_iter=200,
        random_state=0, alpha=1e-1, l1_ratio=.1, verbose=1)
    u = nmf.fit_transform(X)
    v = nmf.components_.T
    z = np.linalg.norm(u, axis=0)
    v = z * v
    return v


class CovarianceSmoothing(object):

    def __init__(self, n_components=300, p=.1):
        self.n_components = n_components
        self.p = p

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
        s = np.einsum('ij,jk,lk', X, self.normalized_V_, self.V_, optimize=True)
        return self.p * s + (1 - self.p) * X
