import argparse
import os
import pickle
from collections import OrderedDict

import clip
import numpy as np

from src.config import FOLDER_MAP, SIM_TYPES
from src.data import load_train_data, load_valid_data
from src.data.cocobias import COCO_DIR, COCO_IMAGES_DIR, select_gendered_caption
from src.metric.clip import extract_caps, extract_imgs
from src.utils import present_args, select_device

CLIP_BACKBONE = "ViT-B/32"


def main(args: argparse.Namespace) -> None:
    sorted_info_path = os.path.join(COCO_DIR, f"coco-{args.sim_type}-cos.pkl")
    if os.path.exists(sorted_info_path):
        print(f"{sorted_info_path} exists, skipping.")
        return

    device = select_device(args.gpu_id)
    valid_data = load_valid_data("cocobias")
    train_data = load_train_data("cocobias")
    valid_ids = list(valid_data)
    train_ids = list(train_data)
    image_folder = os.path.join(COCO_IMAGES_DIR, FOLDER_MAP["coco"])
    valid_filepaths = [os.path.join(image_folder, valid_data[i]["file_name"]) for i in valid_ids]
    train_filepaths = [os.path.join(image_folder, train_data[i]["file_name"]) for i in train_ids]
    model, _ = clip.load(CLIP_BACKBONE, device=device, jit=False)
    model.eval()

    if args.sim_type == "II":
        train_features = extract_imgs(train_filepaths, model, device)
    else:
        train_captions = [
            select_gendered_caption(train_data[i]["caption"], train_data[i]["detected_gender"])
            for i in train_ids
        ]
        train_features = extract_caps(train_captions, model, device)
    valid_features = extract_imgs(valid_filepaths, model, device)

    valid_features = valid_features / np.sqrt(np.sum(valid_features**2, axis=1, keepdims=True))
    train_features = train_features / np.sqrt(np.sum(train_features**2, axis=1, keepdims=True))
    valid_features = valid_features.astype(np.float32)
    train_features = train_features.astype(np.float32)
    cos_sim = valid_features @ train_features.T
    cos_sim = np.clip(cos_sim, a_min=0, a_max=None)

    sorted_indices = np.argsort(cos_sim, axis=1).astype(np.int32)
    sorted_values  = np.take_along_axis(cos_sim, sorted_indices, axis=1)
    sorted_indices = np.flip(sorted_indices, axis=1)
    sorted_values  = np.flip(sorted_values, axis=1)
    train_ids_arr = np.array([int(i) for i in train_ids])

    sorted_info = OrderedDict()
    sorted_info["sorted_train_id"] = train_ids_arr[sorted_indices]
    sorted_info["cossim"] = sorted_values
    sorted_info["valid_id"] = np.array([int(i) for i in valid_ids])
    with open(sorted_info_path, "wb") as f:
        pickle.dump(sorted_info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sim_type", type=str, default="IT", choices=SIM_TYPES)
    args = parser.parse_args()
    present_args(args)
    main(args)
