import numpy as np

from nilearn.input_data import NiftiSpheresMasker

import pytest

from neuroquery import img_utils


def test_get_masker():
    masker = img_utils.get_masker()
    assert np.allclose(np.diag(masker.mask_img_.affine)[:3], [-2.0, 2.0, 2.0])


def test_coords_to_voxels():
    voxels = [[2, 10, 20], [50, 1, 53], [1, 1, 1]]
    img = img_utils.load_mni152_brain_mask()
    coords = img.affine.dot(
        (np.hstack([voxels, np.ones((len(voxels), 1))]).T)
    ).T[:, :-1]
    computed_vox = img_utils.coords_to_voxels(
        coords.tolist() + [coords[-1] - 3, [340, 20, 20]]
    )
    assert computed_vox.shape == np.asarray(voxels).shape
    assert np.allclose(computed_vox, voxels, atol=1)


def test_gaussian_coord_smoothing():
    coords = [(0.0, 0.0, 0.0), (10.0, -10.0, 30.0)]
    computed_img = img_utils.gaussian_coord_smoothing(coords)
    masker = NiftiSpheresMasker(coords + [(-10.0, 10.0, -30)]).fit()
    values = masker.transform(computed_img)[0]
    assert (values[:2] > computed_img.get_data().max() / 2.0).all()
    assert values[-1] == pytest.approx(0.0)
