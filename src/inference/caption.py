import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from nltk import sent_tokenize
from tqdm import tqdm

from src.config import (
    CAPTION_DATASETS,
    CAPTION_IMAGE_TYPES,
    CAPTION_MAX_NEW_TOKEN,
    EXP_FOLDERS,
    NUM_SHOTS,
    RANDOM_EXP_TYPES,
    RANDOM_SEED,
    SHOT_SEEDS,
    TQDM_STYLE,
)
from src.data import CAPTION_FORMAT_FUNCS, load_train_data, load_valid_data
from src.model import MODEL_MAP, ModelHelper, load_model
from src.prompt import CAPTION_INSTRUCTIONS
from src.sampling import present_shot_ids, select_shot_ids
from src.utils import (
    build_result_folder,
    check_caption_run,
    mkdir,
    present_args,
    save_json,
    select_device,
)


def run_caption_inference(
    args: argparse.Namespace | SimpleNamespace,
    model_helper: ModelHelper,
    valid_data: dict[str, dict],
    train_data: dict[str, dict],
    shot_image_ids: list[str] | np.ndarray,
) -> dict[str, list]:
    format_func = CAPTION_FORMAT_FUNCS[args.data_name]
    instruction = CAPTION_INSTRUCTIONS[args.instruction]
    tqdm_valid_data = tqdm(
        valid_data, desc="Generating captions", bar_format=TQDM_STYLE, dynamic_ncols=True
    )
    image_ids, generated_captions, reference_captions, file_names = [], [], [], []
    for i, image_id in enumerate(tqdm_valid_data):
        if args.exp_type in RANDOM_EXP_TYPES:
            shot_ids = shot_image_ids
        else:
            shot_ids = [str(j) for j in shot_image_ids[i].astype(int)]
        assert len(shot_ids) == args.num_shot, f"Got {len(shot_ids)} shots for {args.num_shot}."

        formatted_captions, images = format_func(
            all_valid_data=valid_data, all_train_data=train_data, num_shot=args.num_shot,
            query_image_id=image_id, shot_image_ids=shot_ids, image_type=args.image_type
        )
        model_input = model_helper.prepare_caption_input(formatted_captions, images, instruction)
        sentences = sent_tokenize(model_helper.generate(model_input, CAPTION_MAX_NEW_TOKEN))
        caption = sentences[0] if sentences else ""

        tqdm_valid_data.set_postfix({"Generated": caption})
        generated_captions.append(caption)
        reference_captions.append(formatted_captions[-1])
        file_names.append(valid_data[image_id]["file_name"])
        image_ids.append(image_id)

    return {
        "image_id": image_ids,
        "file_name": file_names,
        "reference_caption": reference_captions,
        "generated_caption": generated_captions,
    }


def caption_exp(
    args: argparse.Namespace | SimpleNamespace,
    model_helper: ModelHelper,
    valid_data: dict[str, dict],
    train_data: dict[str, dict],
) -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    result_folder_path = build_result_folder(
        args.exp_type, args.data_name, args.image_type, args.shot_seed, args.model_name,
        args.num_shot, args.instruction
    )
    result_path = os.path.join(result_folder_path, "result.tsv")
    if os.path.exists(result_path):
        return
    mkdir(result_folder_path)

    shot_ids = select_shot_ids(
        args.exp_type, args.num_shot, train_data, args.shot_seed, RANDOM_SEED, args.data_name,
        args.image_type
    )
    present_shot_ids(shot_ids, train_data)
    with torch.no_grad():
        results = run_caption_inference(args, model_helper, valid_data, train_data, shot_ids)
    pd.DataFrame.from_dict(results).to_csv(result_path, sep="\t", header=True, index=False)

    args_dict = vars(args) | {"seed": RANDOM_SEED, "max_new_token": CAPTION_MAX_NEW_TOKEN}
    args_dict["shot_id"] = shot_ids if args.exp_type in RANDOM_EXP_TYPES else ""
    save_json(args_dict, os.path.join(result_folder_path, "args.json"))


def main(args: argparse.Namespace) -> None:
    check_caption_run(args.exp_type, args.data_name, args.image_type)
    valid_data = load_valid_data(args.data_name)
    train_data = load_train_data(args.data_name, image_type=args.image_type)
    model_helper = load_model(args.model_name, device=select_device(args.gpu_id))
    caption_exp(args, model_helper, valid_data, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot_seed", type=int, default=SHOT_SEEDS[0], choices=SHOT_SEEDS)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--exp_type", type=str, default="1", choices=sorted(EXP_FOLDERS))
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAP))
    parser.add_argument("--data_name", type=str, default="cocobias", choices=CAPTION_DATASETS)
    parser.add_argument("--image_type", type=str, default="coco", choices=CAPTION_IMAGE_TYPES)
    parser.add_argument("--instruction", type=str, default="I1", choices=list(CAPTION_INSTRUCTIONS))
    parser.add_argument("--num_shot", type=int, required=True, choices=NUM_SHOTS)
    args = parser.parse_args()
    present_args(args)
    main(args)
