import argparse
import os
import pickle
from collections import OrderedDict

import numpy as np

from src.config import SIM_TYPES
from src.data.cocobias import BIAS_CAPTION_TRAIN_BALANCED_PATH, COCO_DIR
from src.utils import load_json


def filter_pkl(src_path: str, dst_path: str, kept_ids: np.ndarray) -> None:
    """Keeps only kept_ids in every row of a sim_retrieval ranking, preserving the order."""
    with open(src_path, "rb") as file:
        info = pickle.load(file)
    sorted_train_id: np.ndarray = info["sorted_train_id"]  # (valid, train), descending similarity
    cossim: np.ndarray = info["cossim"]

    keep_mask = np.isin(sorted_train_id, kept_ids)
    counts = keep_mask.sum(axis=1)
    assert (counts == kept_ids.size).all(), (
        f"Non-uniform per-row retention in {src_path}: unique counts {np.unique(counts)}."
    )
    new_train = np.empty((sorted_train_id.shape[0], kept_ids.size), dtype=sorted_train_id.dtype)
    new_cos = np.empty((cossim.shape[0], kept_ids.size), dtype=cossim.dtype)
    for i in range(sorted_train_id.shape[0]):
        new_train[i] = sorted_train_id[i][keep_mask[i]]
        new_cos[i] = cossim[i][keep_mask[i]]

    out = OrderedDict()
    out["sorted_train_id"] = new_train
    out["cossim"] = new_cos
    out["valid_id"] = info["valid_id"]
    with open(dst_path, "wb") as file:
        pickle.dump(out, file)
    print(f"{src_path} {sorted_train_id.shape} -> {dst_path} {new_train.shape}")


def main() -> None:
    dst_paths = {
        sim_type: os.path.join(COCO_DIR, f"coco-balanced-{sim_type}-cos.pkl")
        for sim_type in SIM_TYPES
    }
    if all(os.path.exists(path) for path in dst_paths.values()):
        print(f"Balanced rankings already exist: {', '.join(dst_paths.values())}")
        return
    balanced = load_json(BIAS_CAPTION_TRAIN_BALANCED_PATH)
    kept_ids = np.array(sorted(int(iid) for iid in balanced), dtype=np.int32)
    print(f"Balanced pool: {kept_ids.size} ids from {BIAS_CAPTION_TRAIN_BALANCED_PATH}")
    for sim_type, dst_path in dst_paths.items():
        filter_pkl(os.path.join(COCO_DIR, f"coco-{sim_type}-cos.pkl"), dst_path, kept_ids)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
