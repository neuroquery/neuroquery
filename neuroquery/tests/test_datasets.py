import tempfile
import pathlib
from unittest import mock

import pytest

from neuroquery import text_to_brain
from neuroquery import datasets


class _FileResponse(object):
    def __init__(self, data_file):
        self.data_file = str(
            pathlib.Path(__file__).parent / "data" / data_file
        )
        self.headers = {"content-length": 1000}
        self.status_code = 200

    def iter_content(self, *args, **kwargs):
        with open(self.data_file, "rb") as f:
            data = f.read()
        chunksize = len(data) // 10
        chunks = range(0, len(data), chunksize)
        for chunk in chunks:
            yield data[chunk : chunk + chunksize]

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


class _BadResponse(object):
    def __init__(self, *args):
        self.status_code = 400

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


class _FileGetter(object):
    def __init__(self, data_file, fail_n_times=1):
        self.data_file = data_file
        self.fail_n_times = fail_n_times
        self.n_calls = 0

    def __call__(self, *args, **kwargs):
        self.n_calls += 1
        if self.n_calls > self.fail_n_times:
            return _FileResponse(self.data_file)
        return _BadResponse()


def test_fetch_neuroquery_model():
    getter = _FileGetter("mock-neuroquery_data-master.zip")
    with mock.patch("requests.get", getter):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = datasets.fetch_neuroquery_model(tmp_dir)
            model = text_to_brain.TextToBrain.from_data_dir(data_dir)
            res = model("reading words")
            data_dir = datasets.fetch_neuroquery_model(tmp_dir)
        assert getter.n_calls == 2
        assert "z_map" in res

    getter = _FileGetter("mock-neuroquery_data-master.zip", 7)
    with mock.patch("requests.get", getter):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(RuntimeError):
                data_dir = datasets.fetch_neuroquery_model(tmp_dir)
        assert getter.n_calls == 5
