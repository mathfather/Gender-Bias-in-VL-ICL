import argparse
import os
import random

from src.config import RANDOM_SEED
from src.data.cocobias import BIAS_CAPTION_TRAIN_BALANCED_PATH, BIAS_CAPTION_TRAIN_PATH
from src.utils import load_json, save_json


def gender_ids(data: dict[str, dict], gender: str) -> list[str]:
    ids = [iid for iid, entry in data.items() if entry["detected_gender"] == gender]
    return sorted(ids, key=int)


def downsample_ids(ids: list[str], target: int) -> list[str]:
    # Only the majority side draws from the RNG; sampling both sides would change the draw order.
    return sorted(random.sample(ids, target), key=int) if len(ids) > target else ids


def main() -> None:
    if os.path.exists(BIAS_CAPTION_TRAIN_BALANCED_PATH):
        print(f"Balanced pool already exists: {BIAS_CAPTION_TRAIN_BALANCED_PATH}")
        return
    random.seed(RANDOM_SEED)
    data = load_json(BIAS_CAPTION_TRAIN_PATH)
    male_ids = gender_ids(data, "Male")
    female_ids = gender_ids(data, "Female")
    print(
        f"Loaded {len(data)} entries from {BIAS_CAPTION_TRAIN_PATH}: "
        f"{len(male_ids)} Male, {len(female_ids)} Female."
    )

    target = min(len(male_ids), len(female_ids))
    kept_male_ids = downsample_ids(male_ids, target)
    kept_female_ids = downsample_ids(female_ids, target)
    balanced = {iid: data[iid] for iid in sorted(kept_male_ids + kept_female_ids, key=int)}
    assert len(kept_male_ids) == len(kept_female_ids) == target, (
        f"Balanced pool mismatch: {len(kept_male_ids)} M / {len(kept_female_ids)} F "
        f"(target {target})."
    )
    save_json(balanced, BIAS_CAPTION_TRAIN_BALANCED_PATH)
    print(
        f"Saved {len(balanced)} entries to {BIAS_CAPTION_TRAIN_BALANCED_PATH}: "
        f"{target} Male, {target} Female."
    )


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
