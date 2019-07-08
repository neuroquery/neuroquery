import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import normalize

from neuroquery._img_utils import get_masker


class TextToBrain(object):
    def __init__(self, vectorizer, smoothed_regression, mask_img=None):
        self.vectorizer = vectorizer
        self.smoothed_regression = smoothed_regression
        self.mask_img = mask_img

    def full_vocabulary(self):
        return self.vectorizer.get_vocabulary()

    def _supervised_features(self):
        if not hasattr(
            self.smoothed_regression.regression_, "selected_features_"
        ):
            return np.arange(
                self.smoothed_regression.regression_.coef_.shape[1]
            )
        return self.smoothed_regression.regression_.selected_features_

    def supervised_vocabulary(self):
        return np.asarray(self.full_vocabulary())[self._supervised_features()]

    def _similar_words(self, tfidf, vocabulary=None):
        if vocabulary is None:
            vocabulary = self.full_vocabulary()
        if sparse.issparse(tfidf):
            tfidf = tfidf.A.squeeze()
        similar = pd.Series(tfidf, index=vocabulary).sort_values(
            ascending=False
        )
        return similar[similar > 0]

    def _beta_norms(self):
        return np.linalg.norm(
            self.smoothed_regression.regression_.coef_, axis=0
        )

    def _get_masker(self):
        if not hasattr(self, "masker_"):
            self.masker_ = get_masker(self.mask_img)
        return self.masker_

    def _supervised_vocabulary_set(self):
        if not hasattr(self, "supervised_vocabulary_set_"):
            self.supervised_vocabulary_set_ = set(self.supervised_vocabulary())
        return self.supervised_vocabulary_set_

    def transform(self, documents):
        raw_tfidf = self.vectorizer.transform(documents)
        raw_tfidf = normalize(raw_tfidf, copy=False)
        self.smoothed_regression.regression_.intercept_ = 0.0
        z_scores = self.smoothed_regression.transform_to_z_maps(raw_tfidf)
        masker = self._get_masker()
        z_maps_unmasked = list(map(masker.inverse_transform, z_scores))
        smoothed_tfidf = self.smoothed_regression.smoothing_.transform(
            raw_tfidf
        )
        smoothed_tfidf = normalize(smoothed_tfidf, copy=False)
        return {
            "z_map": z_maps_unmasked,
            "raw_tfidf": raw_tfidf,
            "smoothed_tfidf": smoothed_tfidf,
        }

    def __call__(self, document):
        self.vectorizer.tokenizer.keep_pos = True
        result = self.transform([document])
        result = {k: v[0] for k, v in result.items()}
        similar_words = pd.DataFrame(
            {
                "similarity": self._similar_words(result["smoothed_tfidf"]),
                "weight_in_query": self._similar_words(result["raw_tfidf"]),
                "weight_in_brain_map": self._similar_words(
                    result["smoothed_tfidf"][self._supervised_features()]
                    * self._beta_norms(),
                    self.supervised_vocabulary(),
                ),
            }
        )
        similar_words.fillna(0.0, inplace=True)
        result["similar_words"] = similar_words
        self._supervised_vocabulary_set()
        result[
            "highlighted_text"
        ] = self.vectorizer.tokenizer.highlighted_text(
            lambda w: {
                "in_model": (
                    "true" if w in self.supervised_vocabulary_set_ else "false"
                )
            }
        )
        return result
