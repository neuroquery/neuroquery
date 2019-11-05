"""Show an example of use of MKDA method."""

import pandas as pd
from nilearn import plotting

from neuroquery import datasets
from neuroquery.img_utils import iter_coordinates_to_peaks_imgs
from neuroquery.meta_analysis import MKDA

coordinates = pd.read_csv(datasets.fetch_peak_coordinates())
mkda_img = MKDA(iter_coordinates_to_peaks_imgs(coordinates))
plotting.view_img(mkda_img, threshold=3.0).open_in_browser()
