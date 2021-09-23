import os
import pathlib
import tempfile
import zipfile
import shutil
import zlib

import requests


_MODEL_URLS = {
    "ensemble_model_2020-02-12": "https://osf.io/gj7ek/download",
    "neuroquery_model": "https://osf.io/598tj/download",
}


def get_available_model_names():
    return list(_MODEL_URLS.keys())


def get_neuroquery_data_dir(data_dir=None):
    if data_dir is not None:
        chosen_data_dir = pathlib.Path(data_dir)
    else:
        env_data_dir = os.environ.get("NEUROQUERY_DATA_DIR", None)
        if env_data_dir is not None:
            chosen_data_dir = pathlib.Path(env_data_dir)
        else:
            chosen_data_dir = (
                pathlib.Path(os.environ.get("HOME", ".")) / "neuroquery_data"
            )
    chosen_data_dir.mkdir(exist_ok=True)
    return str(chosen_data_dir)


def _download_neuroquery_model(data_dir, model_name):
    if model_name not in _MODEL_URLS:
        raise ValueError(
            "You asked to download the model: '{}' but it does not exist,"
            " available models are: {}".format(model_name, _MODEL_URLS.keys())
        )
    print("Downloading NeuroQuery model: {}".format(model_name))
    resp = requests.get(_MODEL_URLS[model_name])
    resp.raise_for_status()
    print("Downloaded.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)
        zip_path = str(tmp_dir / "neuroquery_data_master.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        with zipfile.ZipFile(zip_path) as zip_f:
            extract_dir = tmp_dir / "extract_dir"
            zip_f.extractall(str(extract_dir))
        model_path = extract_dir / model_name
        out_dir = pathlib.Path(data_dir) / model_name
        shutil.copytree(str(model_path), str(out_dir))
    return out_dir


def fetch_neuroquery_model(data_dir=None, model_name="neuroquery_model"):
    data_dir = pathlib.Path(get_neuroquery_data_dir(data_dir))
    out_dir = data_dir / model_name
    if out_dir.is_dir():
        return str(out_dir)
    _download_neuroquery_model(str(data_dir), model_name)
    return str(out_dir)


def fetch_peak_coordinates(data_dir=None):
    data_dir = pathlib.Path(get_neuroquery_data_dir(data_dir))
    out_file = data_dir / "coordinates.csv"
    if out_file.is_file():
        return str(out_file)
    print("Downloading coordinates")
    resp = requests.get(
        "https://raw.githubusercontent.com/neuroquery/neuroquery_data/master/data/data-neuroquery_version-1_coordinates.tsv.gz"
    )
    resp.raise_for_status()
    content = zlib.decompress(resp.content, 15 + 32)
    content_lines = content.split(b"\n")
    content_lines_tabs = [line.split(b"\t") for line in content_lines]

    out_file = str(out_file)

    with open(out_file, "wb") as f:
        for line in content_lines_tabs:
            for item in line:
                f.write(item + b",")
            f.write(b"\n")
    print("Downloaded.")
    return out_file
