import os
import pathlib
import tempfile
import zipfile
import shutil

import requests

from neuroquery._utils import try_n_times


def get_neuroquery_data_dir(data_dir=None):
    if data_dir is None:
        data_dir = (
            pathlib.Path(os.environ.get("HOME", ".")) / "neuroquery_data"
        )
    else:
        data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    return data_dir


@try_n_times()
def _download_file(url, out):
    with requests.get(url, stream=True) as resp:
        if resp.status_code != 200:
            raise RuntimeError(
                "Model failed to be downloaded. Try again later?"
            )
        length = int(resp.headers.get("content-length"))
        downloaded = 0
        with open(out, "wb") as out_f:
            for chunk in resp.iter_content(chunk_size=4096):
                downloaded += len(chunk)
                print(
                    "downloading: {:.1%}".format(downloaded / length),
                    end="\r",
                    flush=True,
                )
                out_f.write(chunk)
        return out


def _download_neuroquery_data(out_dir):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)
        repo_url = (
            "https://github.com/neuroquery/neuroquery_data/archive/master.zip"
        )
        f = str(tmp_dir / "neuroquery_data_master.zip")
        _download_file(repo_url, f)
        zf = zipfile.ZipFile(f)
        repo = tmp_dir / "neuroquery_data"
        zf.extractall(str(repo))
        data = repo / "neuroquery_data-master"
        shutil.copytree(str(data), out_dir)
    return out_dir


def fetch_neuroquery_model(data_dir=None):
    data_dir = get_neuroquery_data_dir(data_dir)
    out_dir = data_dir / "neuroquery_data"
    if out_dir.is_dir():
        return str(out_dir / "neuroquery_model")
    _download_neuroquery_data(str(out_dir))
    return str(out_dir / "neuroquery_model")
