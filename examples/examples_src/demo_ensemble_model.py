"""
Demo of an ensemble model
=========================

30 NeuroQuery models were trained with subsampled vocabularies and averaged.
This example downloads this ensemble model and shows a prediction.

The bagging seems to make the model more robust and improve results for some
queries. However the query smoothing and encoding cannot be separated anymore.
"""

######################################################################

from nilearn import plotting
from neuroquery import fetch_neuroquery_model
from neuroquery.encoding import SimpleEncoder

model_dir = fetch_neuroquery_model(model_name="ensemble_model_2020-02-12")
encoder = SimpleEncoder.from_data_dir(model_dir)

######################################################################

query = "theory of mind"
result = encoder(query)
plotting.view_img(
    result["brain_map"], title=query, threshold="97%", colorbar=False)

######################################################################

plotting.view_img(
    result["brain_map"], title=query, threshold="97%", colorbar=False
).open_in_browser()
