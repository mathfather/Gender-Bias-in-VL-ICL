import argparse
import json
import os
from types import SimpleNamespace

import nltk

from src.config import (
    EXP_FOLDERS,
    FEMININE,
    GENDERED_EXP_TYPES,
    MASCULINE,
    NEUTRAL,
    RANDOM_EXP_TYPES,
    RESULT,
    RETRIEVAL_EXP_TYPES,
    SYNTHETIC_IMAGE_TYPES,
)

MASCULINE_SET = set(MASCULINE)
FEMININE_SET = set(FEMININE)
NEUTRAL_SET = set(NEUTRAL)


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> list | dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_json(data: list | dict, path: str, indent: int=4) -> None:
    with open(path, "w") as file:
        json.dump(data, file, indent=indent)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def present_args(args: argparse.Namespace | SimpleNamespace) -> None:
    items = vars(args)
    width = max((len(key) for key in items), default=0) + 6
    print("=" * 30)
    print("Settings:")
    for key, value in items.items():
        print(f"{key + ':':<{width}}{value}")
    print("=" * 30)


def select_exp_path(exp_type: str | int) -> str:
    key = str(exp_type)
    if key not in EXP_FOLDERS:
        raise ValueError(f"Invalid exp_type '{key}'; expected one of {sorted(EXP_FOLDERS)}.")
    return os.path.join(RESULT, EXP_FOLDERS[key])


def build_result_folder(
    exp_type: str | int,
    data_name: str,
    data_type: str,
    shot_seed: int,
    model_name: str,
    num_shot: int,
    instruction: str | None=None,
) -> str:
    """results/<setting>/<data_name>-<data_type>-<shot_seed>/<model_name>-<num_shot>shot[-<instruction>]."""
    model_folder = f"{model_name}-{num_shot}shot"
    if instruction is not None:
        model_folder = f"{model_folder}-{instruction}"
    dataset_folder = f"{data_name}-{data_type}-{shot_seed}"
    return os.path.join(select_exp_path(exp_type), dataset_folder, model_folder)


def check_caption_run(exp_type: str, data_name: str, image_type: str) -> None:
    """Raise ValueError for a setting, dataset and image source combination the paper never runs."""
    if data_name == "dci":
        if image_type != "dci":
            raise ValueError(f"data_name 'dci' needs image_type 'dci', got '{image_type}'.")
        if exp_type not in RANDOM_EXP_TYPES:
            raise ValueError(
                f"data_name 'dci' needs exp_type in {RANDOM_EXP_TYPES}, got '{exp_type}'."
            )
    elif image_type == "dci":
        raise ValueError(f"image_type 'dci' needs data_name 'dci', got '{data_name}'.")
    if image_type == "coco-balanced" and exp_type not in RETRIEVAL_EXP_TYPES:
        raise ValueError(
            f"image_type 'coco-balanced' needs exp_type in {RETRIEVAL_EXP_TYPES}, got '{exp_type}'."
        )
    if image_type in SYNTHETIC_IMAGE_TYPES and exp_type not in GENDERED_EXP_TYPES:
        raise ValueError(
            f"image_type '{image_type}' needs exp_type in {GENDERED_EXP_TYPES}, got '{exp_type}'."
        )


def select_device(gpu_id: int) -> str:
    import torch

    num_devices = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= num_devices:
        raise ValueError(f"gpu_id {gpu_id} is not in [0, {num_devices}).")
    return f"cuda:{gpu_id}"


def detect_gender_from_text(text: str) -> str:
    """'Male', 'Female', 'Invalid' (both genders), 'Neutral' (neutral words only) or 'Unknown'."""
    tokens = set(nltk.word_tokenize(text.lower().strip()))
    has_male = bool(tokens & MASCULINE_SET)
    has_female = bool(tokens & FEMININE_SET)
    if has_male and has_female:
        return "Invalid"
    if has_male:
        return "Male"
    if has_female:
        return "Female"
    if tokens & NEUTRAL_SET:
        return "Neutral"
    return "Unknown"
