import tempfile
import pathlib
from unittest import mock

import requests

from neuroquery import text_to_brain
from neuroquery import datasets


class _FileResponse(object):

    def __init__(self, data_file):
        self.data_file = str(
            pathlib.Path(__file__).parent / 'data' / data_file)
        self.headers = {'content-length': 1000}

    def iter_content(self, *args, **kwargs):
        with open(self.data_file, 'rb') as f:
            data = f.read()
        chunksize = len(data) // 10
        chunks = range(0, len(data), chunksize)
        for chunk in chunks:
            yield data[chunk: chunk + chunksize]

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


def test_fetch_neuroquery_model():
    resp = _FileResponse('mock-neuroquery_data-master.zip')
    with mock.patch('requests.get', mock.Mock(return_value=resp)):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = datasets.fetch_neuroquery_model(tmp_dir)
            model = text_to_brain.TextToBrain.from_data_dir(data_dir)
            res = model('reading words')
            data_dir = datasets.fetch_neuroquery_model(tmp_dir)
        requests.get.assert_called_once()
        assert 'z_map' in res
