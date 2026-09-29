import argparse
import os
from types import SimpleNamespace

import clip
import pandas as pd
from clip.model import CLIP
from string2string.metrics import sacreBLEU

from src.config import (
    CAPTION_DATASETS,
    CAPTION_IMAGE_TYPES,
    EXP_FOLDERS,
    FOLDER_MAP,
    NUM_SHOTS,
    SHOT_SEEDS,
    SYNTHETIC_IMAGE_TYPES,
)
from src.data import DATASET_CAPTION_KEYS, DATASET_IMAGE_BASE, load_valid_data
from src.metric import (
    caption_bleu,
    caption_clipscore,
    caption_error_rates,
    caption_n_words,
    caption_reveal_rates,
)
from src.model import MODEL_MAP
from src.prompt import CAPTION_INSTRUCTIONS
from src.utils import (
    build_result_folder,
    check_caption_run,
    detect_gender_from_text,
    present_args,
    save_json,
    select_device,
)

CLIP_BACKBONE = "ViT-B/32"


def load_clip(device: str) -> CLIP:
    clip_model, _ = clip.load(CLIP_BACKBONE, device=device, jit=False)
    clip_model.eval()
    return clip_model


def eval_caption(
    args: argparse.Namespace | SimpleNamespace,
    sbleu: sacreBLEU,
    clip_model: CLIP,
    device: str,
) -> None:
    """Adds per-sample scores to the run's result.tsv and writes its evaluation.json."""
    result_folder = build_result_folder(
        args.exp_type, args.data_name, args.image_type, args.shot_seed, args.model_name,
        args.num_shot, args.instruction
    )
    result_path = os.path.join(result_folder, "result.tsv")
    eval_path = os.path.join(result_folder, "evaluation.json")
    if not os.path.exists(result_path):
        print(f"[SKIP] {result_path} not found")
        return
    if os.path.exists(eval_path):
        print(f"[SKIP] {eval_path} already exists")
        return

    result = pd.read_csv(result_path, sep="\t", header=0)
    result["generated_caption"] = result["generated_caption"].fillna("")
    evaluation = {}

    valid_data = load_valid_data(args.data_name)
    image_ids = result["image_id"].tolist()
    result["detected_gender"] = [valid_data[str(i)]["detected_gender"] for i in image_ids]
    detected_g = result["detected_gender"]
    captions = result["generated_caption"].tolist()
    caption_key = DATASET_CAPTION_KEYS[args.data_name]
    reference_captions = [valid_data[str(i)][caption_key] for i in image_ids]
    if reference_captions and isinstance(reference_captions[0], str):
        reference_captions = [[ref] for ref in reference_captions]

    bleu_results, bleu_scores = caption_bleu(captions, reference_captions, detected_g, sbleu)
    result["BLEU-4"] = bleu_scores
    evaluation.update(bleu_results)

    nword_results, n_words = caption_n_words(captions, detected_g)
    result["n_word"] = n_words
    evaluation.update(nword_results)

    # Synthetic images replace only the shots; the captioned query images are the real COCO ones.
    clip_image_type = "coco" if args.image_type in SYNTHETIC_IMAGE_TYPES else args.image_type
    image_folder = os.path.join(DATASET_IMAGE_BASE[args.data_name], FOLDER_MAP[clip_image_type])
    image_paths = [os.path.join(image_folder, valid_data[str(i)]["file_name"]) for i in image_ids]
    clip_results, clip_scores = caption_clipscore(
        captions, image_paths, detected_g, clip_model, device, batch_size=args.batch_size
    )
    result["CLIPScore"] = clip_scores
    evaluation.update(clip_results)

    result["predicted_gender"] = result["generated_caption"].apply(detect_gender_from_text)
    pred_g = result["predicted_gender"]
    evaluation.update(caption_error_rates(detected_g, pred_g))
    evaluation.update(caption_reveal_rates(detected_g, pred_g))

    result.to_csv(result_path, sep="\t", header=True, index=False)
    save_json(evaluation, eval_path)


def main(args: argparse.Namespace) -> None:
    check_caption_run(args.exp_type, args.data_name, args.image_type)
    device = select_device(args.gpu_id)
    eval_caption(args, sacreBLEU(), load_clip(device), device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="1", choices=sorted(EXP_FOLDERS))
    parser.add_argument("--shot_seed", type=int, default=SHOT_SEEDS[0], choices=SHOT_SEEDS)
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAP))
    parser.add_argument("--data_name", type=str, default="cocobias", choices=CAPTION_DATASETS)
    parser.add_argument("--image_type", type=str, default="coco", choices=CAPTION_IMAGE_TYPES)
    parser.add_argument("--instruction", type=str, default="I1", choices=list(CAPTION_INSTRUCTIONS))
    parser.add_argument("--num_shot", type=int, required=True, choices=NUM_SHOTS)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    present_args(args)
    main(args)
