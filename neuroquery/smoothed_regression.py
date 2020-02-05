import pathlib

from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin

from neuroquery import ridge, nmf


class SmoothedRegression(BaseEstimator, RegressorMixin):
    @classmethod
    def from_data_dir(cls, model_dir):
        model_dir = pathlib.Path(model_dir)
        smoothing_dir = model_dir / "smoothing"
        smoothing = nmf.CovarianceSmoothing.from_data_dir(smoothing_dir)
        regression_dir = model_dir / "regression"
        regression = ridge.FittedLinearModel.from_data_dir(regression_dir)
        model = cls(
            alphas=None,
            n_components=smoothing.n_components,
            smoothing_weight=smoothing.smoothing_weight,
        )
        model.smoothing_ = smoothing
        model.regression_ = regression
        return model

    def to_data_dir(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        smoothing_dir = model_dir / "smoothing"
        smoothing_dir.mkdir()
        self.smoothing_.to_data_dir(smoothing_dir)
        regression_dir = model_dir / "regression"
        regression_dir.mkdir()
        ridge.FittedLinearModel.from_model(self.regression_).to_data_dir(
            regression_dir
        )

    def __init__(
        self,
        alphas=ridge._DEFAULT_ALPHAS,
        n_components=300,
        smoothing_weight=0.1,
        transform_to_z=True
    ):
        self.alphas = alphas
        self.n_components = n_components
        self.smoothing_weight = smoothing_weight
        self.transform_to_z = transform_to_z

    def fit(self, X, Y):
        if sparse.issparse(X):
            X = X.A
        self.smoothing_ = nmf.CovarianceSmoothing(
            n_components=self.n_components,
            smoothing_weight=self.smoothing_weight,
        ).fit(X)
        self.regression_ = ridge.SelectiveRidge(
            alphas=self.alphas, store_M=True
        ).fit(X, Y)
        return self

    def predict(self, X):
        X = self.smoothing_.transform(X)
        return self.regression_.predict(X)

    def transform_to_z_maps(self, X):
        X = self.smoothing_.transform(X)
        return self.regression_.transform_to_z_maps(X)

    def transform_to_brain_maps(self, X):
        if not self.transform_to_z:
            return self.predict(X)
        try:
            return self.transform_to_z_maps(X)
        except Exception:
            return self.predict(X)
