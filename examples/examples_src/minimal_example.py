"""
Minimal example: encoding with NeuroQuery
=========================================

The model used here is the same as the one deployed on the neuroquery website
( https://neuroquery.saclay.inria.fr ).
You can run this example on binder: https://mybinder.org/v2/gh/neuroquery/neuroquery.git/master?filepath=examples
"""
######################################################################
# Encode a query into a statistical map of the brain
# --------------------------------------------------
from neuroquery import fetch_neuroquery_model, NeuroQueryModel
from nilearn.plotting import view_img

encoder = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

######################################################################
query = """Huntington's disease"""
result = encoder(query)
view_img(result["z_map"], threshold=3.1)

######################################################################
# (drag the mouse on this interactive plot to see other slices)
#
# Note: if you are not using a jupyter notebook, use `.open_in_browser()` to
# open the plot above:

view_img(result["z_map"], threshold=3.1).open_in_browser()


######################################################################
# Display some relevant terms:

print(result["similar_words"].head(15))

######################################################################
# Display some relevant studies:

print("\nsimilar studies:\n")
print(result["similar_documents"].head())
