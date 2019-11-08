# # Meta-analysis: an example of use of the MKDA method.

# +
import pandas as pd
from nilearn import plotting

from neuroquery import datasets
from neuroquery.img_utils import coordinates_to_arrs
from neuroquery.meta_analysis import MKDA

# +
# Load the coordinates from the NeuroQuery dataset.
coordinates = pd.read_csv(datasets.fetch_peak_coordinates())

# Use the coordinates_to_arrs function to get a generator of the dirac maps' arrays
iter_arrs, masker = coordinates_to_arrs(coordinates)
affine = masker.mask_img_.affine
# -

# Compute the MKDA maps from the array generator. Since the iter_arrs generate arrays only
# we have to pass the affine separatly.
mkda_img = MKDA(iter_arrs, affine)

# Plot the result
plotting.view_img(mkda_img, threshold=3.0).open_in_browser()
