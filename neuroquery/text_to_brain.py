import pathlib

import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import normalize
from nilearn import image

from neuroquery.img_utils import get_masker
from neuroquery import tokenization, smoothed_regression


class TextToBrain(object):
    """Text -> brain map encoder.

    It encodes text into statistical maps of the brain and also provides a list
    of related terms.

    It can be initialized with a fitted regression model
    (`neuroquery.smoothed_regression.SmoothedRegression`) or loaded using
    `from_data_dir`. Most users will probably load a pre-trained model with
    `from_data_dir`.

    Parameters
    ----------
    vectorizer : `neuroquery.tokenization.TextVectorizer`
        An object that transforms text into TFIDF features.

    smoothed_regression : `neuroquery.smoothed_regression.SmoothedRegression`
        A reduced-rank regression that combines feature smoothing, projection,
        and linear regression. The input features must correspond to the
        outputs of `vectorizer`.

    mask_img : Nifti1Image
        Mask of the regression targets. The non-zero voxels correspond to the
        dependent variables.

    """
    @classmethod
    def from_data_dir(cls, model_dir):
        """Load a pre-trained TextToBrain model.

        Parameters
        ----------
        model_dir : str
            path to a directory containing the serialized trained model.
            The directory must be organized as the one returned by
            `neuroquery.datasets.fetch_neuroquery_model`.
        """
        model_dir = pathlib.Path(model_dir)
        vectorizer = tokenization.TextVectorizer.from_vocabulary_file(
            str(model_dir / "vocabulary.csv"), voc_mapping="auto"
        )
        regression = smoothed_regression.SmoothedRegression.from_data_dir(
            str(model_dir)
        )
        mask_img = image.load_img(str(model_dir / "mask_img.nii.gz"))
        return cls(vectorizer, regression, mask_img)

    def to_data_dir(self, model_dir):
        """Save the model so it can later be loaded with `from_data_dir`."""
        model_dir = pathlib.Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer.to_vocabulary_file(str(model_dir / "vocabulary.csv"))
        self.smoothed_regression.to_data_dir(model_dir)
        self._get_masker().mask_img_.to_filename(
            str(model_dir / "mask_img.nii.gz")
        )

    def __init__(self, vectorizer, smoothed_regression, mask_img):
        self.vectorizer = vectorizer
        self.smoothed_regression = smoothed_regression
        self.mask_img = mask_img

    def full_vocabulary(self):
        """All the terms recognized by the model."""
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
        """Terms selected as features for the supervised regression."""
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
        """Transform a set of documents

        Parameters
        ----------
        documents : list or array of str
            the text snippets to transform

        Returns
        -------
        list of dict, each containing:
            - "z_map": a nifti image of the most relevant brain regions.
            - "raw_tfidf": the vectorized documents.
            - "smoothed_tfidf": the tfidf after semantic smoothing.

        """
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
        """Transform a document

        Parameters
        ----------
        document : str
            the text to transform

        Returns
        -------
        dict containing:
            - "z_map": a nifti image of the most relevant brain regions.
            - "raw_tfidf": the vectorized documents.
            - "similar_words": pandas DataFrame containing related terms.
                - "similarity" is how much the term is related.
                - "weight_in_brain_map" is the contribution of the term in the
                  predicted "z_map".
                - "weight_in_query" is the TFIDF of the term in `document`.
            - "highlighted_text": an XML document showing which terms were
              recognized in the provided text.
            - "smoothed_tfidf": the tfidf after semantic smoothing.
        """
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
            },
            columns=["similarity", "weight_in_brain_map", "weight_in_query"],
        )
        similar_words.fillna(0.0, inplace=True)
        similar_words.sort_values(
            by="weight_in_brain_map", ascending=False, inplace=True
        )
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
