"""
Basic neuroquery example: downloading a trained model and making a few queries
==============================================================================

This example shows basic neuroquery functionality:

- download a trained neuroquery model
- query it with some text
- have a look at the result: a brain map and a list of similar or related
    terms.

The model used here is the same as the one deployed on the neuroquery website
( https://neuroquery.org ), and this example shows how to easily
reproduce the website's functionality in a Python script.

You can run this example on binder: https://mybinder.org/v2/gh/neuroquery/neuroquery.git/main?filepath=examples
"""

######################################################################
# Download the vocabulary and learned coefficients of a NeuroQuery model
# ----------------------------------------------------------------------

from neuroquery import fetch_neuroquery_model

neuroquery_data = fetch_neuroquery_model()
print(neuroquery_data)


######################################################################
# Load the NeuroQuery model
# -------------------------

from neuroquery import NeuroQueryModel

encoder = NeuroQueryModel.from_data_dir(neuroquery_data)


######################################################################
# Query the model and display the results
# ---------------------------------------

# taken from Wikipedia
query = "Aphasia is an inability to comprehend or formulate language"

response = encoder(query)
print(response.keys())


######################################################################
# The "brain_map" entry of the results is a brain map showing the anatomical
# regions that are most strongly associated with the query in the neuroimaging
# literature. It is a `Nifti1Image` which can be saved, displayed, etc.

from nilearn import plotting

print(type(response["brain_map"]))
plotting.plot_stat_map(
    response["brain_map"], display_mode="z", title="aphasia", threshold=3.1
)


######################################################################
#

# Display the map on the cortical surface:
view = plotting.view_img_on_surf(response["brain_map"], threshold=3.1)
view.open_in_browser()
# (in a Jupyter notebook, we can display an inline view):
view

######################################################################
#

# Or open interactive viewer:
view = plotting.view_img(response["brain_map"], threshold=3.1)
view.open_in_browser()
# (in a Jupyter notebook, we can display an inline view):
view

######################################################################
# "similar_words" is a DataFrame containing terms that are related to the
# query. For each related terms, three numbers are provided.
# "similarity" is the strength of the association between the term and the
# query, according to co-occurrence statistics in the literature.

print("Most similar terms:\n")
print(
    response["similar_words"]
    .sort_values(by="similarity", ascending=False)
    .head()
)

######################################################################
# "weight_in_brain_map" is the importance of the term in the brain map. It
# depends both on the similarity with the query and on the strength of the link
# between this term and brain activity. Terms that are similar to the query and
# have a strong association with brain activity in the literature get a higher
# weight.

print("\nMost important terms for brain map prediction:\n")
print(
    response["similar_words"]
    .sort_values(by="weight_in_brain_map", ascending=False)
    .head()
)

######################################################################
# "weight_in_query" is the importance of the term in the query itself. It
# reflects the number of times each term appears in the query (reweighted so
# that very common, uninformative words get a lower weight). For example, terms
# that do not appear in the query get a "weight_in_query" of 0.

print("\nTerms recognized in the query:\n")
print(
    response["similar_words"]
    .query("weight_in_query != 0")
    .sort_values(by="weight_in_query", ascending=False)
    .head()
)

######################################################################
# "similar_documents" contains a list of related studies.
# for each study it provides its PubMed ID, title, similarity to the query, and
# a link to its PubMed page.

print("\nsimilar studies:\n")
print(response["similar_documents"].head())

######################################################################
# Finally, "highlighted_text" contains the text of the query itself, with
# markups that indicates which terms were recognized and used by the model
# (usually terms related to neuroimaging):

print("\ntokenized query:\n")
print(response["highlighted_text"])

# `print_highlighted_text` can help display it in a terminal for debugging.

from neuroquery.tokenization import print_highlighted_text

print("\n")
print_highlighted_text(response["highlighted_text"])

######################################################################

plotting.show()
