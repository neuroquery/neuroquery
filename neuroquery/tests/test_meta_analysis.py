"""Test the meta_analysis.py file."""
import scipy
import pandas as pd
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as strats
from numpy.random import randint, uniform

from neuroquery import meta_analysis


affine = np.array([[-2.,    0.,    0.,   90.],
                [0.,    2.,    0., -126.],
                [0.,    0.,    2.,  -72.],
                [0.,    0.,    0.,    1.]])


def random_coordinates(NS, N_max):
    """Generate a random coordinates dataframe."""
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
    })


def test_uniform_kernel():
    """Test the creation of an uniform kernel."""
    kern = meta_analysis._uniform_kernel(1, 1, 1, 1)

    expected_layer1 = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    expected_layer2 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])
    expected_layer3 = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    expected = np.array([
        expected_layer1,
        expected_layer2,
        expected_layer3
    ])

    assert np.array_equal(kern, expected)

    kern2 = meta_analysis._uniform_kernel(5, 1, 1, 1)

    assert kern2[5, 0, 5] == 1
    assert kern2[5, 5, 0] == 1
    assert kern2[0, 5, 5] == 1
    assert kern2[5, 5, 10] == 1
    assert kern2[5, 10, 5] == 1
    assert kern2[10, 5, 5] == 1
    assert kern2[10, 5, 6] == 0
    assert kern2[10, 6, 5] == 0


@given(
    NS=strats.integers(min_value=1, max_value=10),
    N_max=strats.integers(min_value=2, max_value=100)
)
def test_KDA(NS, N_max):
    """Test the KDA meta_analysis."""
    coordinates = random_coordinates(NS, N_max)

    KDA = meta_analysis.KDA(coordinates, r=5).get_fdata()

    assert KDA.all() >= 0
    assert KDA.all() <= NS*N_max

    with pytest.raises(ValueError):
        meta_analysis.MKDA(pd.DataFrame({'A': []}), r=5)

@given(
    NS=strats.integers(min_value=1, max_value=10),
    N_max=strats.integers(min_value=2, max_value=100)
)
def test_MKDA(NS, N_max):
    """Test the MKDA meta_analysis."""
    coordinates = random_coordinates(NS, N_max)

    MKDA = meta_analysis.MKDA(coordinates, r=5).get_fdata()

    assert MKDA.all() >= 0
    assert MKDA.all() <= NS

    with pytest.raises(ValueError):
        meta_analysis.MKDA(pd.DataFrame({'A': []}), r=5)
