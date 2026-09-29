"""Dataset construction and loading.

Build each dataset from its downloaded sources (see data/<dataset>/README.md):
    python -m src.data.cocobias
    python -m src.data.dci
    python -m src.data.visogender --download_images
    python -m src.data.visualcot

Inference loads the built splits through load_train_data and load_valid_data. Caption runs
format one query with its shots through CAPTION_FORMAT_FUNCS[data_name]; pronoun and VQA runs
call format_visogender and format_visualcot directly.
"""

from collections.abc import Callable

from src.data import cocobias, dci, visogender, visualcot
from src.utils import load_json

CAPTION_FORMAT_FUNCS: dict[str, Callable] = {
    "cocobias": cocobias.format_coco2014,
    "dci":      dci.format_dci,
}

TRAIN_PATHS: dict[str, str] = {
    "cocobias":      cocobias.BIAS_CAPTION_TRAIN_PATH,
    "dci":           dci.BIAS_CAPTION_TRAIN_PATH,
    "visualcot":     visualcot.BIAS_COT_TRAIN_PATH,
    "visogender-OO": visogender.OO_TRAIN_JSON_PATH,
    "visogender-OP": visogender.OP_TRAIN_JSON_PATH,
}

VALID_PATHS: dict[str, str] = {
    "cocobias":      cocobias.BIAS_CAPTION_VALID_PATH,
    "dci":           dci.BIAS_CAPTION_VALID_PATH,
    "visualcot":     visualcot.BIAS_COT_VALID_PATH,
    "visogender-OO": visogender.OO_VALID_JSON_PATH,
    "visogender-OP": visogender.OP_VALID_JSON_PATH,
}

DATASET_CAPTION_KEYS: dict[str, str] = {
    "cocobias": "caption",
    "dci":      "short_caption",
}

DATASET_IMAGE_BASE: dict[str, str] = {
    "cocobias": cocobias.COCO_IMAGES_DIR,
    "dci":      dci.DCI_DIR,
}

__all__ = [
    "DATASET_CAPTION_KEYS",
    "DATASET_IMAGE_BASE",
    "CAPTION_FORMAT_FUNCS",
    "TRAIN_PATHS",
    "VALID_PATHS",
    "load_train_data",
    "load_valid_data",
]


def _split_key(data_name: str, data_type: str | None) -> str:
    if data_name == "visogender":
        return f"visogender-{data_type}"
    return data_name


def load_train_data(
    data_name: str,
    data_type: str | None=None,
    image_type: str | None=None,
) -> dict[str, dict]:
    if data_name == "cocobias" and image_type == "coco-balanced":
        path = cocobias.BIAS_CAPTION_TRAIN_BALANCED_PATH
    else:
        path = TRAIN_PATHS[_split_key(data_name, data_type)]
    data = load_json(path)
    print(f"Shots loaded from {path}.")
    return data


def load_valid_data(data_name: str, data_type: str | None=None) -> dict[str, dict]:
    path = VALID_PATHS[_split_key(data_name, data_type)]
    data = load_json(path)
    print(f"Queries loaded from {path}.")
    return data
