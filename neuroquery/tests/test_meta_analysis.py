"""Test the meta_analysis.py file."""
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as strats

from neuroquery import meta_analysis


def rand_peaks_arrs(shape, NS, N_max):
    """Generate a random list of peaks arrays."""
    if N_max == 0:
        return [np.zeros(shape)]*NS

    return [np.random.randint(N_max, size=shape)]*NS


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
    N_max=strats.integers(min_value=0, max_value=100),
    shape=strats.tuples(
        strats.integers(min_value=1, max_value=10),
        strats.integers(min_value=1, max_value=10),
        strats.integers(min_value=1, max_value=10)
        )
)
def test_KDA(shape, NS, N_max):
    """Test the KDA meta_analysis."""
    peaks_arrs = rand_peaks_arrs(shape, NS, N_max)

    affine = np.eye(4)
    KDA = meta_analysis.KDA(peaks_arrs, affine, r=1).get_fdata()

    assert KDA.shape == shape
    assert KDA.all() >= 0
    assert KDA.all() <= NS*N_max

    with pytest.raises(ValueError):
        meta_analysis.KDA([], affine, r=1)


@given(
    NS=strats.integers(min_value=1, max_value=10),
    N_max=strats.integers(min_value=0, max_value=100),
    shape=strats.tuples(
        strats.integers(min_value=1, max_value=10),
        strats.integers(min_value=1, max_value=10),
        strats.integers(min_value=1, max_value=10)
        )
)
def test_MKDA(shape, NS, N_max):
    """Test the MKDA meta_analysis."""
    peaks_arrs = rand_peaks_arrs(shape, NS, N_max)

    affine = np.eye(4)
    MKDA = meta_analysis.MKDA(peaks_arrs, affine, r=1).get_fdata()

    assert MKDA.shape == shape
    assert MKDA.all() >= 0
    assert MKDA.all() <= NS

    with pytest.raises(ValueError):
        meta_analysis.MKDA([], affine, r=1)
