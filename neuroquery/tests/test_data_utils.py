"""Test the meta_analysis.py file."""
import tempfile
import pandas as pd
import numpy as np
from hypothesis import given
from hypothesis import strategies as strats
from numpy.random import randint, uniform

from neuroquery import data_utils


affine = np.array([[-2.,    0.,    0.,   90.],
                   [0.,    2.,    0., -126.],
                   [0.,    0.,    2.,  -72.],
                   [0.,    0.,    0.,    1.]])


@strats.composite
def random_coordinates(draw):
    """Generate a random coordinates dataframe."""
    NS = draw(strats.integers(min_value=1, max_value=10),)
    N_max = draw(strats.integers(min_value=2, max_value=100))
    L = []
    for s in range(NS):
        n_peaks = randint(1, N_max)
        peaks = uniform(low=(0, 0, 0, 1), high=(91, 109, 91, 2), size=(n_peaks, 4))
        x, y, z, _ = affine @ peaks.T
        L.append(np.array((x, y, z, s*np.ones(n_peaks))))

    x, y, z, pmids = np.hstack(L)

    return pd.DataFrame(data={
        'x': x,
        'y': y,
        'z': z,
        'pmid': pmids
    }), NS, N_max


def test_build_index():
    L = ['kw1', 'kw2', 'kw3']
    encode, decode = data_utils.build_index(L)

    assert encode['kw1'] == 0
    assert encode['kw2'] == 1
    assert encode['kw3'] == 2
    assert decode[0] == 'kw1'
    assert decode[1] == 'kw2'
    assert decode[2] == 'kw3'

    # Test with a file
    file = tempfile.NamedTemporaryFile(mode='w+t')
    for s in L:
        file.write(f'{s}\n')
    file.seek(0)  # Go back to the beginning of the file

    encode, decode = data_utils.build_index(file.name)

    assert encode['kw1'] == 0
    assert encode['kw2'] == 1
    assert encode['kw3'] == 2
    assert decode[0] == 'kw1'
    assert decode[1] == 'kw2'
    assert decode[2] == 'kw3'

    file.close()


@given(
    data=random_coordinates()
)
def test_select_pmids(data):
    coordinates, NS, N_max = data

    ids = range(NS)
    coords = data_utils.select_pmids(coordinates, ids)
    assert len(coords.index) == len(coordinates.index)

    ids = [0, NS-1]
    coords = data_utils.select_pmids(coordinates, ids)
    selected_ids = coords.pmid.unique()
    assert 0 in selected_ids
    assert NS-1 in selected_ids


@given(
    data=random_coordinates()
)
def test_which_pmids(data):
    tfidf = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ])
    pmids = ['100', '101', '102']
    keywords = ['kw1', 'kw2', 'kw3']

    selected_pmids = data_utils.which_pmids(tfidf, pmids, keywords, 'kw1')

    assert 101 in selected_pmids
    assert 102 in selected_pmids

