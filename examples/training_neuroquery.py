import pathlib

import numpy as np
from scipy import sparse
import pandas as pd
from joblib import Memory

from neuroquery.img_utils import coordinates_to_maps
from neuroquery.smoothed_regression import SmoothedRegression
from neuroquery.tokenization import TextVectorizer
from neuroquery.encoding import NeuroQueryModel

coord_to_maps = Memory('/tmp/').cache(coordinates_to_maps)
data_dir = pathlib.Path('/home/jerome/workspace/neuroquery_data/training_data')

pmids = np.loadtxt(str(data_dir / 'pmids.txt'), delimiter=',', dtype=int)
voc = np.loadtxt(str(data_dir / 'feature_names.txt'), delimiter=',', dtype=str)
tfidf = sparse.load_npz(str(data_dir / 'corpus_tfidf.npz'))

coordinates = pd.read_csv(str(data_dir / 'coordinates.csv'))[:100]

brain_maps, masker = coord_to_maps(coordinates)
brain_maps = brain_maps[(brain_maps.values != 0).any(axis=1)]

kept_pmids = pd.Series(pmids).isin(brain_maps.index).values
tfidf = tfidf[kept_pmids, :]
pmids = pmids[kept_pmids]
brain_maps = brain_maps.loc[pmids, :]

regressor = SmoothedRegression()

print("Fitting smoothed regression model...")
regressor.fit(tfidf, brain_maps.values)
vectorizer = TextVectorizer.from_vocabulary(voc).fit()

encoder = NeuroQueryModel(vectorizer, regressor, masker.mask_img_)
encoder.to_data_dir('/tmp/trained_text_to_brain_model')

from nilearn import plotting
plotting.view_img(encoder('huntington')['z_map']).open_in_browser()
