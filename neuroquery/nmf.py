import numpy as np

from sklearn.decomposition import NMF


def smoothing_matrix_sqrt(X, n_components=300):
    nmf = NMF(
        n_components=n_components, max_iter=200,
        random_state=0, alpha=1e-1, l1_ratio=.1, verbose=1)
    u = nmf.fit_transform(X)
    v = nmf.components_.T
    z = np.linalg.norm(u, axis=0)
    v = z * v
    return v
