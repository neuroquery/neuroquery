from nilearn import plotting
from neuroquery import fetch_neuroquery_model
from neuroquery.encoding import SimpleEncoder

encoder = SimpleEncoder.from_data_dir("/tmp/ensemble_model")

query = "theory of mind"
result = encoder(query)
plotting.view_img(
    result["brain_map"], title=query, threshold="99%").open_in_browser()
