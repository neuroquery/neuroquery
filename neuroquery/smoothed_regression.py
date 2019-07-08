from sklearn.base import BaseEstimator, RegressorMixin

from neuroquery import ridge, nmf


class SmoothedRegression(BaseEstimator, RegressorMixin):
    def __init__(self,
                 alphas=ridge._DEFAULT_ALPHAS,
                 n_components=300,
                 smoothing_weight=.1):
        self.alphas = alphas
        self.n_components = n_components
        self.smoothing_weight = smoothing_weight

    def fit(self, X, Y):
        self.smoothing_ = nmf.CovarianceSmoothing(
            n_components=self.n_components,
            smoothing_weight=self.smoothing_weight).fit(X)
        self.regression_ = ridge.SelectiveRidge(alphas=self.alphas,
                                                store_M=True).fit(X, Y)
        return self

    def predict(self, X):
        X = self.smoothing_.transform(X)
        return self.regression_.predict(X)

    def transform_to_z_maps(self, X):
        X = self.smoothing_.transform(X)
        return self.regression_.transform_to_z_maps(X)
