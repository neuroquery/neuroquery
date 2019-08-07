import numpy as np
import pandas as pd

from nilearn.input_data import NiftiSpheresMasker

import pytest

from neuroquery import img_utils


def test_get_masker():
    masker = img_utils.get_masker()
    assert np.allclose(np.diag(masker.mask_img_.affine)[:3], [-2.0, 2.0, 2.0])
    new_masker = img_utils.get_masker(mask_img=masker)
    assert new_masker is masker
    new_masker = img_utils.get_masker(masker.mask_img_)
    assert new_masker.mask_img_ is masker.mask_img_
    masker = img_utils.get_masker(masker.mask_img_, target_affine=(5, 5, 5))
    assert np.allclose(np.diag(masker.mask_img_.affine)[:3], [5.0, 5.0, 5.0])
    masker = img_utils.get_masker(masker.mask_img_, target_affine=(3., 3., 3.))
    assert np.allclose(np.diag(masker.mask_img_.affine)[:3], [3.0, 3.0, 3.0])
    masker = img_utils.get_masker(masker.mask_img_, target_affine=7.)
    assert np.allclose(np.diag(masker.mask_img_.affine)[:3], [7.0, 7.0, 7.0])
    masker = img_utils.get_masker(
        masker.mask_img_, target_affine=4.3 * np.eye(3))
    assert np.allclose(np.diag(masker.mask_img_.affine)[:3], [4.3, 4.3, 4.3])


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


def test_coordinates_to_maps():
    coords = pd.DataFrame.from_dict({
        'pmid': [3, 17, 17, 2, 2],
        'x': [0., 0., 10., 5., 3.],
        'y': [0., 0., -10., 15., -9.],
        'z': [27., 0., 30., 17., 177.],
    })
    maps, masker = img_utils.coordinates_to_maps(coords)
    assert maps.shape == (3, 28542)
    coords_17 = [(0.0, 0.0, 0.0), (10.0, -10.0, 30.0)]
    img_17 = img_utils.gaussian_coord_smoothing(coords_17, target_affine=4.)
    assert np.allclose(masker.transform(img_17), maps.loc[17, :].values,
                       atol=1e-10)
