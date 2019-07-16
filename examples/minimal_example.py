"""
Minimal example: encoding with NeuroQuery
=========================================

The model used here is the same as the one deployed on the neuroquery website
( https://neuroquery.saclay.inria.fr ).
"""
######################################################################
# Encode a query into a statistical map of the brain
# --------------------------------------------------
from neuroquery import datasets, text_to_brain
from nilearn.plotting import view_img

encoder = text_to_brain.TextToBrain.from_data_dir(
    datasets.fetch_neuroquery_model()
)

######################################################################
query = """Parkinson's disease"""

view_img(encoder(query)["z_map"], threshold=3.1)

######################################################################
# Note: if you are not using a jupyter notebook, use `.open_in_browser()` to
# open the plot above.
