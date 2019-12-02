# # Meta-analysis: an example of use of the KDA & MKDA method.

# +
import pandas as pd
from nilearn import plotting

from neuroquery import datasets
from neuroquery.img_utils import coordinates_to_arrays
from neuroquery.meta_analysis import KDA, MKDA
# -

# Load the coordinates from the NeuroQuery dataset.
coordinates = pd.read_csv(datasets.fetch_peak_coordinates())

# Compute the MKDA maps from the coordinates.
kda_img = KDA(coordinates)

# Compute the MKDA maps from the coordinates.
mkda_img = MKDA(coordinates)

# Plot the result
plotting.view_img(kda_img, threshold=3.0).open_in_browser()
plotting.view_img(mkda_img, threshold=3.0).open_in_browser()
