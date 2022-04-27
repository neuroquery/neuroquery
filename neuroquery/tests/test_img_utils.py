import numpy as np
import pandas as pd

from nilearn.input_data import NiftiSpheresMasker
from nilearn.image import get_data

import pytest

from neuroquery import img_utils


def test_get_masker():
    masker = img_utils.get_masker()
    assert np.allclose(
        np.abs(np.diag(masker.mask_img_.affine)[:3]), [2.0, 2.0, 2.0]
    )
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
    assert (values[:2] > get_data(computed_img).max() / 2.0).all()
    assert values[-1] == pytest.approx(0.0)


@pytest.mark.parametrize("persist_memmap", [True, False])
def test_coordinates_to_maps(persist_memmap, tmp_path):
    coords = pd.DataFrame.from_dict(
        {
            "pmid": [3, 17, 17, 2, 2],
            "x": [0.0, 0.0, 10.0, 5.0, 3.0],
            "y": [0.0, 0.0, -10.0, 15.0, -9.0],
            "z": [27.0, 0.0, 30.0, 17.0, 177.0],
        }
    )
    if persist_memmap:
        memmap = tmp_path.joinpath("maps.dat")
    else:
        memmap = None
    maps, masker = img_utils.coordinates_to_maps(
        coords, output_memmap_file=memmap
    )
    # nilearn mni mask changed
    assert maps.shape == (3, 28542) or maps.shape == (3, 29398)
    coords_17 = [(0.0, 0.0, 0.0), (10.0, -10.0, 30.0)]
    img_17 = img_utils.gaussian_coord_smoothing(coords_17, target_affine=4.0)
    assert np.allclose(
        masker.transform(img_17), maps.loc[17, :].values, atol=1e-10
    )


# Original version of the coordinates_to_maps method
def _coordinates_to_maps(
    coordinates, mask_img=None, target_affine=(4, 4, 4), fwhm=9.0
):
    print(
        "Transforming {} coordinates for {} articles".format(
            coordinates.shape[0], len(set(coordinates["pmid"]))
        )
    )
    masker = img_utils.get_masker(
        mask_img=mask_img, target_affine=target_affine
    )
    images, img_pmids = [], []
    for pmid, img in img_utils.iter_coordinates_to_maps(
        coordinates, mask_img=masker, fwhm=fwhm
    ):
        images.append(masker.transform(img).ravel().astype("float32"))
        img_pmids.append(pmid)
    return pd.DataFrame(images, index=img_pmids), masker


@pytest.mark.parametrize("persist_memmap", [True, False])
def test_parallel_coordinates_to_maps_should_match_original_implementation(
    persist_memmap, tmp_path
):
    coords = pd.DataFrame.from_dict(
        {
            "pmid": [3, 17, 17, 2, 2],
            "x": [0.0, 0.0, 10.0, 5.0, 3.0],
            "y": [0.0, 0.0, -10.0, 15.0, -9.0],
            "z": [27.0, 0.0, 30.0, 17.0, 177.0],
        }
    )
    if persist_memmap:
        memmap = tmp_path.joinpath("maps.dat")
    else:
        memmap = None
    maps, masker = img_utils.coordinates_to_maps(
        coords, n_jobs=2, output_memmap_file=memmap
    )
    original_maps, original_masker = _coordinates_to_maps(coords)

    pd.testing.assert_frame_equal(maps, original_maps)
    np.testing.assert_equal(masker.affine_, original_masker.affine_)
