import random

from src.config import SAMPLE_POOL_SIZE


def _draw_pool(image_ids: list[str], num_shot: int) -> list[str]:
    """First num_shot ids of a random pool of up to SAMPLE_POOL_SIZE ids, so k shots prefix k+2."""
    return random.sample(image_ids, min(SAMPLE_POOL_SIZE, len(image_ids)))[:num_shot]


def random_sampling(
    num_shot: int,
    shot_data: dict[str, dict],
    shot_seed: int,
    seed: int,
) -> list[str]:
    random.seed(shot_seed)
    shot_image_ids = _draw_pool(list(shot_data), num_shot) if num_shot > 0 else []
    random.seed(seed)
    return shot_image_ids


def random_sampling_same_gender(
    num_shot: int,
    shot_data: dict[str, dict],
    shot_seed: int,
    seed: int,
    gender: str,
) -> list[str]:
    assert gender in ("Male", "Female"), f"Invalid gender '{gender}'."
    random.seed(shot_seed)
    shot_image_ids = []
    if num_shot > 0:
        image_ids = [i for i in shot_data if shot_data[i]["detected_gender"] == gender]
        shot_image_ids = _draw_pool(image_ids, num_shot)
    random.seed(seed)
    return shot_image_ids


def random_sampling_balanced(
    num_shot: int,
    shot_data: dict[str, dict],
    shot_seed: int,
    seed: int,
) -> list[str]:
    random.seed(shot_seed)
    shot_image_ids = []
    if num_shot > 0:
        assert num_shot % 2 == 0, f"Balanced sampling needs an even num_shot, got {num_shot}."
        male_ids = [i for i in shot_data if shot_data[i]["detected_gender"] == "Male"]
        female_ids = [i for i in shot_data if shot_data[i]["detected_gender"] == "Female"]
        # The male pool is drawn first; swapping the draws changes every sampled shot.
        male_shot_ids = _draw_pool(male_ids, num_shot // 2)
        female_shot_ids = _draw_pool(female_ids, num_shot // 2)
        for male_id, female_id in zip(male_shot_ids, female_shot_ids, strict=False):
            shot_image_ids += [male_id, female_id]
    random.seed(seed)
    return shot_image_ids
