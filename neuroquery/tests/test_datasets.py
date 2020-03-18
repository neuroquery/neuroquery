import tempfile
import pathlib
from unittest import mock
import pytest

from neuroquery import encoding
from neuroquery import datasets


class _FileResponse(object):
    def __init__(self, data_file):
        self.data_file = str(
            pathlib.Path(__file__).parent / "data" / data_file
        )
        self.status_code = 200
        with open(self.data_file, "rb") as f:
            self.content = f.read()

    def raise_for_status(self):
        pass


def test_fetch_neuroquery_model():
    resp = _FileResponse("mock-neuroquery_model.zip")
    with mock.patch("requests.get", return_value=resp) as mock_get:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = datasets.fetch_neuroquery_model(tmp_dir)
            model = encoding.NeuroQueryModel.from_data_dir(data_dir)
            res = model("reading words")
            assert "brain_map" in res
            data_dir = datasets.fetch_neuroquery_model(tmp_dir)
            mock_get.assert_called_once()
    with pytest.raises(ValueError):
        datasets.fetch_neuroquery_model(model_name="bad_model")


def test_get_neuroquery_data_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = str(pathlib.Path(tmp_dir) / "neuroquery_data_1")
        nq_dir = datasets.get_neuroquery_data_dir(target_dir)
        assert nq_dir == target_dir
        assert pathlib.Path(nq_dir).is_dir()
        env = {"HOME": tmp_dir}
        with mock.patch("os.environ", env):
            nq_dir = datasets.get_neuroquery_data_dir()
            assert nq_dir == str(pathlib.Path(tmp_dir) / "neuroquery_data")
            assert pathlib.Path(nq_dir).is_dir()
        env_data_dir = str(pathlib.Path(tmp_dir) / "env_data_dir")
        env = {"HOME": tmp_dir, "NEUROQUERY_DATA_DIR": env_data_dir}
        with mock.patch("os.environ", env):
            nq_dir = datasets.get_neuroquery_data_dir()
            assert nq_dir == env_data_dir
            assert pathlib.Path(nq_dir).is_dir()


def test_fetch_peak_coordinates():
    resp = mock.Mock()
    resp.content = b"peak_coordinates_test"
    with mock.patch("requests.get", return_value=resp) as mock_get:
        with tempfile.TemporaryDirectory() as tmp_dir:
            coord_file = datasets.fetch_peak_coordinates(tmp_dir)
            with open(coord_file) as f:
                assert f.read() == "peak_coordinates_test"
            coord_file = datasets.fetch_peak_coordinates(tmp_dir)
            mock_get.assert_called_once()


def test_get_available_model_names():
    names = datasets.get_available_model_names()
    assert names == list(datasets._MODEL_URLS.keys())
