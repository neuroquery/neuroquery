# # Meta-analysis: an example of use of the KDA & MKDA method.

# +
import pandas as pd
from nilearn import plotting
from scipy.sparse import load_npz

from neuroquery import datasets
from neuroquery.img_utils import coordinates_to_arrays
from neuroquery.meta_analysis import KDA, MKDA
from neuroquery.data_utils import which_pmids, select_pmids
# -

# Load the coordinates and corpus data from the NeuroQuery dataset.
coordinates = pd.read_csv(datasets.fetch_peak_coordinates())
tfidf = load_npz(datasets.fetch_tfidf())
pmids = datasets.fetch_pmids()
keywords = datasets.fetch_keywords()

# Select the studies related to a given keyword.
keyword = 'prosopagnosia'
pmids = which_pmids(tfidf, pmids, keywords, keyword)
coordinates = select_pmids(coordinates, pmids)

# Compute the MKDA maps from the coordinates.
kda_img = KDA(coordinates)

# Compute the MKDA maps from the coordinates.
mkda_img = MKDA(coordinates)

# Plot the result
plotting.view_img(kda_img, threshold=3.0).open_in_browser()
plotting.view_img(mkda_img, threshold=3.0).open_in_browser()
