"""
Advanced example: training a NeuroQuery model from scratch
==========================================================

In this example, we train a NeuroQuery model from TFIDF descriptors and peak
activation coordinates for around 13K neuroimaging studies.

We transform the coordinates into brain maps, fit a regression model, show an
example prediction, and store the trained model.

This example is more computation-intensive than the others. It runs in around
40 minutes in total (around 25 mn to transform coordinates into brain maps, and
15 mn to fit the regression model) and uses up to 6 GB of memory.

Note: for this demo we use a very coarse resolution of brain maps (6mm voxels),
change `target_affine` to e.g. `(4, 4, 4)` to increase resolution. With 4mm
resolution, the example requires around 60 mn and 14 GB of memory.
"""

######################################################################
# Collect training data
# ---------------------
import pathlib

import numpy as np
from scipy import sparse
import pandas as pd
from joblib import Memory
from nilearn import plotting

from neuroquery import datasets
from neuroquery.img_utils import coordinates_to_maps
from neuroquery.smoothed_regression import SmoothedRegression
from neuroquery.tokenization import TextVectorizer
from neuroquery.encoding import NeuroQueryModel

# Choose where to store the cache and the model once it is trained
output_directory = "trained_text_to_brain_model"
cache_directory = "cache"

data_dir = pathlib.Path(datasets.fetch_neuroquery_model())

corpus_metadata = pd.read_csv(str(data_dir / "corpus_metadata.csv"))
vectorizer = TextVectorizer.from_vocabulary_file(
    str(data_dir / "vocabulary.csv")
)

# The TFIDF features stored with NeuroQuery data correspond to the terms in
# `vocabulary.csv` and the studies in `corpus_metadata.csv`;
# see `README.md` in the data directory for details
tfidf = sparse.load_npz(str(data_dir / "corpus_tfidf.npz"))

coordinates = pd.read_csv(datasets.fetch_peak_coordinates())

######################################################################
# Transform the coordinates into brain maps
# -----------------------------------------

# We cache the `coordinates_to_maps` function with joblib to avoid recomputing
# this if we train a new model.
coord_to_maps = Memory(cache_directory).cache(coordinates_to_maps)

# You can set target_affine to a different value to increase image resolution
# or reduce computation time. The model on neuroquery.org uses 4 mm
# resolution i.e. target_affine=(4, 4, 4)
# You can also adjust the smoothing by setting `fwhm` (Full Width at Half
# maximum)
brain_maps, masker = coord_to_maps(
    coordinates, target_affine=(6, 6, 6), fwhm=9.0
)
brain_maps = brain_maps[(brain_maps.values != 0).any(axis=1)]

######################################################################
# Make sure TFIDF and brain maps are aligned (correspond to the same studies)

pmids = brain_maps.index.intersection(corpus_metadata["pmid"])
rindex = pd.Series(
    np.arange(corpus_metadata.shape[0]), index=corpus_metadata["pmid"]
)
tfidf = tfidf.A[rindex.loc[pmids].values, :]
brain_maps = brain_maps.loc[pmids, :]

######################################################################
# Train the regression model
# --------------------------
regressor = SmoothedRegression(alphas=[1.0, 10.0, 100.0])

print(
    "Fitting smoothed regression model on {} samples...".format(tfidf.shape[0])
)
regressor.fit(tfidf, brain_maps.values)

######################################################################
# Build a NeuroQuery model and serialize it
# -----------------------------------------
# It is an interface to the regression model that tokenizes the text of
# queries, unmasks the predicted brain maps, and formats the outputs.
# This is the type of object that we will serialize and that is used in other
# examples.
corpus_metadata = corpus_metadata.set_index("pmid").loc[pmids, :].reset_index()
encoder = NeuroQueryModel(
    vectorizer,
    regressor,
    masker.mask_img_,
    corpus_info={
        "tfidf": sparse.csr_matrix(tfidf),
        "metadata": corpus_metadata,
    },
)
encoder.to_data_dir(output_directory)

######################################################################
# Show an example prediction from our freshly trained model
# ---------------------------------------------------------
query = "Reading words"
print('Encoding "{}"'.format(query))

result = encoder(query)

plotting.view_img(result["brain_map"], threshold=3.0).open_in_browser()

print("Similar words:")
print(result["similar_words"].head())
print("\nSimilar documents:")
print(result["similar_documents"].head())

print("\nmodel saved in {}".format(output_directory))

# Display in notebook
plotting.view_img(result["brain_map"], threshold=3.0)

######################################################################
# Now that the model is trained and saved, it can easily be loaded in a later
# session

encoder = NeuroQueryModel.from_data_dir(output_directory)

######################################################################
# Summary
# -------
# We have trained and used a NeuroQuery model based on coordinates and TFIDF
# features. As we have seen, in order to train a model, we need:
# - An array of TFIDF features
# - An array of masked brain maps (shape n samples x n voxels), which can be
#   easily obtained from coordinates with `coordinates_to_maps`, such that each
#   brain map corresponds to a row of the TFIDF matrix
#
# Then to construct a `NeuroQueryModel` that can answer queries, we also need
# to build a `TextVectorizer`, either by reading a vocabulary file as is done
# in this example, or simply by calling
# `TextVectorizer.from_vocabulary(feature_names)`, where `feature_names` is a
# list of strings giving the terms that correspond to each column of the TFIDF
# matrix.
# Finally, optionally, `NeuroQueryModel` can also be provided with data about
# the corpus (e.g. article titles), which will be used to describe documents
# related to queries if available.
#
# Therefore a model can be trained and used using a TFIDF matrix, a DataFrame
# of coordinates or an array of brain maps, and a vocabulary list only. It does
# not require a dataset on disk with any particular directory structure.
#
# Finally, the TFIDF features themselves can easily be obtained with
# `TextVectorizer`, or similar vectorizers from `scikit-learn`, if you have a
# corpus of text. Here we use the TFIDF distributed in `neuroquery_data`
# because we do not have access to the corpus of text they were derived from.
