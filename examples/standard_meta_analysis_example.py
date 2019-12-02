# # Meta-analysis: an example of use of the KDA & MKDA method.

# +
import pandas as pd
from nilearn import plotting

from neuroquery import datasets
from neuroquery.img_utils import coordinates_to_arrays
from neuroquery.meta_analysis import KDA, MKDA

# +
# Load the coordinates from the NeuroQuery dataset.
coordinates = pd.read_csv(datasets.fetch_peak_coordinates())

# Use the coordinates_to_arrays function to get a generator of the
# dirac maps' arrays
iter_arrays, masker = coordinates_to_arrays(coordinates)
affine = masker.mask_img_.affine
# -

# Compute the MKDA maps from the array generator. Since the iter_arrays
# generate arrays only we have to pass the affine separatly.
kda_img = KDA(iter_arrays, affine)

# Since the previous iterator has been consulmed, we create a new one
iter_arrays, masker = coordinates_to_arrays(coordinates)
affine = masker.mask_img_.affine
# -

# Compute the MKDA maps from the array generator. Since the iter_arrays
# generate arrays only we have to pass the affine separatly.
mkda_img = MKDA(iter_arrays, affine)

# Plot the result
plotting.view_img(kda_img, threshold=3.0).open_in_browser()
plotting.view_img(mkda_img, threshold=3.0).open_in_browser()
