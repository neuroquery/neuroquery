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
    masker = img_utils.get_masker(
        masker.mask_img_, target_affine=(3.0, 3.0, 3.0)
    )
    assert np.allclose(np.diag(masker.mask_img_.affine)[:3], [3.0, 3.0, 3.0])
    masker = img_utils.get_masker(masker.mask_img_, target_affine=7.0)
    assert np.allclose(np.diag(masker.mask_img_.affine)[:3], [7.0, 7.0, 7.0])
    masker = img_utils.get_masker(
        masker.mask_img_, target_affine=4.3 * np.eye(3)
    )
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
    coords = pd.DataFrame.from_dict(
        {
            "pmid": [3, 17, 17, 2, 2],
            "x": [0.0, 0.0, 10.0, 5.0, 3.0],
            "y": [0.0, 0.0, -10.0, 15.0, -9.0],
            "z": [27.0, 0.0, 30.0, 17.0, 177.0],
        }
    )
    maps, masker = img_utils.coordinates_to_maps(coords)
    assert maps.shape == (3, 28542)
    coords_17 = [(0.0, 0.0, 0.0), (10.0, -10.0, 30.0)]
    img_17 = img_utils.gaussian_coord_smoothing(coords_17, target_affine=4.0)
    assert np.allclose(
        masker.transform(img_17), maps.loc[17, :].values, atol=1e-10
    )


def test_coordinates_to_arrays():
    coords = pd.DataFrame.from_dict(
        {
            "pmid": [3, 17, 17, 2, 2],
            "x": [0.0, 0.0, 10.0, 5.0, 3.0],
            "y": [0.0, 0.0, -10.0, 15.0, -9.0],
            "z": [27.0, 0.0, 30.0, 17.0, 77.0],
        }
    )
    iter_arrs, masker = img_utils.coordinates_to_arrays(coords)
    affine = masker.mask_img.affine

    ijk = []
    xyz = [(5, 15, 17), (3, -9, 77), (0, 0, 27), (0, 0, 0), (10, -10, 30)]

    for x, y, z in xyz:
        i, j, k, _ = np.floor((np.linalg.pinv(affine) @ np.array([[x, y, z, 1]]).T)).astype(int)
        ijk.append((i, j, k))

    arr2 = next(iter_arrs)
    arr3 = next(iter_arrs)
    arr17 = next(iter_arrs)

    assert arr2[ijk[0]] == 1
    assert arr2[ijk[1]] == 1
    assert arr3[ijk[2]] == 1
    assert arr17[ijk[3]] == 1
    assert arr17[ijk[4]] == 1
