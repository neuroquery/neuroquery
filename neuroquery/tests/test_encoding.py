import tempfile

import numpy as np
import pandas as pd
from sklearn import datasets
from nibabel.nifti1 import Nifti1Image

import pytest

from neuroquery import encoding, smoothed_regression, tokenization


def _dataset_and_voc():
    n_features = 91
    x, y = datasets.make_regression(
        n_samples=102,
        n_informative=5,
        n_features=n_features,
        n_targets=117,
        effective_rank=9,
        noise=0.5,
        shuffle=False,
        random_state=0,
    )
    x -= x.min() - 1
    voc = list(map("feature{}".format, range(n_features)))
    return x, y, voc


def _mask_img(n):
    img = np.ones((1, 1, n))
    affine = np.eye(4)
    return Nifti1Image(img, affine=affine)


def test_neuroquery_model():
    x, y, voc = _dataset_and_voc()
    vect = tokenization.TextVectorizer.from_vocabulary(voc)
    reg = smoothed_regression.SmoothedRegression(n_components=10).fit(x, y)
    encoder = encoding.NeuroQueryModel(
        vect, reg, mask_img=_mask_img(y.shape[1])
    )
    text = "feature0 and feature8 compared to feature73"
    res = encoder(text)
    simil = res["similar_words"]
    assert simil.loc["feature0"]["similarity"] != 0
    assert simil.loc["feature0"]["weight_in_brain_map"] != 0
    assert simil.loc["feature0"]["weight_in_query"] != 0
    assert simil.loc["feature8"]["weight_in_query"] != 0
    assert simil.loc["feature8"]["similarity"] != 0
    assert simil.loc["feature8"]["weight_in_brain_map"] == pytest.approx(0)
    assert simil.loc["feature18"]["weight_in_brain_map"] == pytest.approx(0)
    assert simil.loc["feature18"]["weight_in_query"] == pytest.approx(0)
    assert res["similar_documents"] is None
    with tempfile.TemporaryDirectory() as tmp_dir:
        encoder.to_data_dir(tmp_dir)
        loaded = encoding.NeuroQueryModel.from_data_dir(tmp_dir)
        assert not loaded.vectorizer.add_unigrams
    encoded = loaded(text)["z_map"].get_data()
    assert np.allclose(encoded, res["z_map"].get_data())

    n_docs = 4
    tfidf = np.zeros((n_docs, x.shape[1]))
    tfidf[:n_docs, :n_docs] = np.eye(n_docs)
    metadata = pd.DataFrame.from_dict({"id": np.arange(n_docs)})
    encoder = encoding.NeuroQueryModel(
        vect,
        reg,
        mask_img=_mask_img(y.shape[1]),
        corpus_info={"tfidf": tfidf, "metadata": metadata},
    )
    for i in range(n_docs):
        res = encoder(encoder.full_vocabulary()[i])
        assert res["similar_documents"]["id"][0] == i
        assert res["similar_words"]["n_documents"][0] == 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        encoder.to_data_dir(tmp_dir)
        loaded = encoding.NeuroQueryModel.from_data_dir(tmp_dir)
        assert not loaded.vectorizer.add_unigrams

    for i in range(n_docs):
        res = encoder(encoder.full_vocabulary()[i])
        assert res["similar_documents"].id[0] == i
