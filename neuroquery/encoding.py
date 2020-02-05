import pathlib

import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot
from nilearn import image

from neuroquery.img_utils import get_masker
from neuroquery import tokenization, smoothed_regression

_MAX_SIMILAR_DOCS_RETURNED = 100


class NeuroQueryModel(object):
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

    corpus_info : dict, optional (default=None)
        Data required to report which studies are most relevant for a query.
        Must contain:
            - "metadata": pandas DataFrame, each row describing a study
            - "tfidf": scipy sparse matrix or numpy array, TFIDF features for
              the documents. Rows must correspond to the same studies as in
              "metadata", and columns to the terms in the vectorizer's
              vocabulary.
        If corpus_info is not available the model will not report most similar
        studies.
    """

    @classmethod
    def from_data_dir(cls, model_dir):
        """Load a pre-trained TextToBrain model.

        Parameters
        ----------
        model_dir : str
            path to a directory containing the serialized trained model.
            The directory must be organized as the one returned by
            `neuroquery.datasets.fetch_neuroquery_model`, except that
            `corpus_metadata.csv` and `corpus_tfidf.npz` are optional.
        """
        model_dir = pathlib.Path(model_dir)
        vectorizer = tokenization.TextVectorizer.from_vocabulary_file(
            str(model_dir / "vocabulary.csv"),
            voc_mapping="auto",
            add_unigrams=False,
        )
        regression = smoothed_regression.SmoothedRegression.from_data_dir(
            str(model_dir)
        )
        mask_img = image.load_img(str(model_dir / "mask_img.nii.gz"))
        corpus_tfidf = model_dir / "corpus_tfidf.npz"
        corpus_metadata = model_dir / "corpus_metadata.csv"
        if corpus_tfidf.is_file() and corpus_metadata.is_file():
            corpus_info = {}
            corpus_info["tfidf"] = sparse.load_npz(str(corpus_tfidf))
            corpus_info["metadata"] = pd.read_csv(
                str(corpus_metadata), encoding="utf-8"
            )
        else:
            corpus_info = None
        return cls(vectorizer, regression, mask_img, corpus_info=corpus_info)

    def to_data_dir(self, model_dir):
        """Save the model so it can later be loaded with `from_data_dir`."""
        model_dir = pathlib.Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer.to_vocabulary_file(str(model_dir / "vocabulary.csv"))
        self.smoothed_regression.to_data_dir(model_dir)
        self._get_masker().mask_img_.to_filename(
            str(model_dir / "mask_img.nii.gz")
        )
        if self.corpus_info is not None:
            sparse.save_npz(
                str(model_dir / "corpus_tfidf.npz"),
                sparse.csr_matrix(self.corpus_info["tfidf"]),
            )
            self.corpus_info["metadata"].to_csv(
                str(model_dir / "corpus_metadata.csv"), index=False
            )

    def __init__(
        self, vectorizer, smoothed_regression, mask_img, corpus_info=None
    ):
        self.vectorizer = vectorizer
        self.smoothed_regression = smoothed_regression
        self.mask_img = mask_img
        self.corpus_info = corpus_info

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

    def document_frequencies(self):
        if self.corpus_info is None:
            return None
        if not hasattr(self, "document_frequencies_"):
            document_frequencies = (self.corpus_info["tfidf"] > 0).sum(axis=0)
            document_frequencies = np.asarray(document_frequencies).ravel()
            self.document_frequencies_ = pd.Series(
                document_frequencies, index=self.full_vocabulary()
            )
        return self.document_frequencies_

    def _similar_words(self, tfidf, vocabulary=None):
        if vocabulary is None:
            vocabulary = self.full_vocabulary()
        if sparse.issparse(tfidf):
            tfidf = tfidf.A.squeeze()
        similar = pd.Series(tfidf, index=vocabulary).sort_values(
            ascending=False
        )
        return similar[similar > 0]

    def similar_documents(self, tfidf):
        if self.corpus_info is None:
            return None
        similarities = safe_sparse_dot(
            tfidf, self.corpus_info["tfidf"].T, dense_output=True
        ).ravel()
        order = np.argsort(similarities)[::-1]
        order = order[similarities[order] > 0][:_MAX_SIMILAR_DOCS_RETURNED]
        ordered_simil = similarities[order]
        similar_docs = (
            self.corpus_info["metadata"].iloc[order].reset_index(drop=True)
        )
        similar_docs["similarity"] = ordered_simil
        return similar_docs

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
        brain_maps = self.smoothed_regression.transform_to_brain_maps(
            raw_tfidf)
        masker = self._get_masker()
        brain_maps_unmasked = list(map(masker.inverse_transform, brain_maps))
        smoothed_tfidf = self.smoothed_regression.smoothing_.transform(
            raw_tfidf
        )
        smoothed_tfidf = normalize(smoothed_tfidf, copy=False)
        return {
            "brain_map": brain_maps_unmasked,
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
            - "similar_words": pandas DataFrame containing related terms.
                - "similarity" is how much the term is related.
                - "weight_in_brain_map" is the contribution of the term in the
                  predicted "z_map".
                - "weight_in_query" is the TFIDF of the term in `document`.
            - "similar_documents": if no corpus_info was provided, this is
              `None`. Otherwise it is a DataFrame containing information about
              the most relevant studies.
            - "highlighted_text": an XML document showing which terms were
              recognized in the provided text.
            - "smoothed_tfidf": the tfidf after semantic smoothing.
            - "raw_tfidf": the vectorized documents.
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
        doc_freq = self.document_frequencies()
        if doc_freq is not None:
            similar_words["n_documents"] = doc_freq.loc[similar_words.index]
            similar_words = similar_words.loc[
                :,
                [
                    "similarity",
                    "weight_in_brain_map",
                    "weight_in_query",
                    "n_documents",
                ],
            ]
        result["similar_words"] = similar_words
        result["similar_documents"] = self.similar_documents(
            result["smoothed_tfidf"]
        )
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
