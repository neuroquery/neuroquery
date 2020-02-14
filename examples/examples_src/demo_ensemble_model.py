from nilearn import plotting
from neuroquery import fetch_neuroquery_model
from neuroquery.encoding import SimpleEncoder

model_dir = fetch_neuroquery_model(model_name="ensemble_model_2020-02-12")
encoder = SimpleEncoder.from_data_dir(model_dir)

query = "theory of mind"
result = encoder(query)
plotting.view_img(
    result["brain_map"], title=query, threshold="97%", colormap=False
).open_in_browser()
