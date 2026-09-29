import os
import pickle
from collections.abc import Callable
from functools import partial

import numpy as np

from src.config import DATASETS, EXP_FOLDERS
from src.data.cocobias import COCO_DIR
from src.sampling.random_sampling import (
    random_sampling,
    random_sampling_balanced,
    random_sampling_same_gender,
)
from src.sampling.similarity_sampling import similarity_sampling

RANDOM_SAMPLERS: dict[str, Callable[..., list[str]]] = {
    "1": random_sampling,
    "2": partial(random_sampling_same_gender, gender="Male"),
    "3": partial(random_sampling_same_gender, gender="Female"),
    "4": random_sampling_balanced,
}
RETRIEVAL_SIM_TYPES: dict[str, str] = {
    "5": "II",
    "6": "IT",
}


def _load_ranking(image_type: str, sim_type: str) -> np.ndarray:
    """[num_query, num_train] train ids, most similar first (written by sim_retrieval)."""
    with open(os.path.join(COCO_DIR, f"{image_type}-{sim_type}-cos.pkl"), "rb") as f:
        return pickle.load(f)["sorted_train_id"]


def select_shot_ids(
    exp_type: str,
    num_shot: int,
    shot_data: dict[str, dict],
    shot_seed: int,
    seed: int,
    data_name: str,
    image_type: str | None=None,
) -> list[str] | np.ndarray:
    """Ids shared by every query (settings 1-4), or a [num_query, num_shot] id array (5-6)."""
    assert data_name in DATASETS, f"Invalid data_name '{data_name}'."
    if exp_type in RANDOM_SAMPLERS:
        return RANDOM_SAMPLERS[exp_type](
            num_shot=num_shot, shot_data=shot_data, shot_seed=shot_seed, seed=seed
        )
    if exp_type not in RETRIEVAL_SIM_TYPES:
        raise ValueError(f"Invalid exp_type '{exp_type}'; expected one of {sorted(EXP_FOLDERS)}.")
    if data_name != "cocobias":
        raise ValueError(f"Retrieval settings need data_name 'cocobias', got '{data_name}'.")
    # shot_seed is unused: the retrieval order is fixed by the precomputed similarities.
    return similarity_sampling(num_shot, _load_ranking(image_type, RETRIEVAL_SIM_TYPES[exp_type]))


def present_shot_ids(shot_ids: list[str] | np.ndarray, shot_data: dict[str, dict]) -> None:
    print("=" * 20)
    print("Shot ids:")
    if isinstance(shot_ids, list):
        for i, image_id in enumerate(shot_ids):
            print(f"{i + 1}. {image_id} - {shot_data[image_id]['detected_gender']}")
    else:
        print(shot_ids)
    print("=" * 20)
