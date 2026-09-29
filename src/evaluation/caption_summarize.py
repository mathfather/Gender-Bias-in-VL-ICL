import argparse

from src.config import CAPTION_DATASETS, CAPTION_IMAGE_TYPES, EXP_FOLDERS
from src.evaluation.summarize import summarize_runs
from src.model import MODEL_MAP
from src.prompt import CAPTION_INSTRUCTIONS
from src.utils import check_caption_run, present_args


def main(args: argparse.Namespace) -> None:
    check_caption_run(args.exp_type, args.data_name, args.image_type)
    # A 0-shot run shows no shot images, so COCOBias reuses its "coco" run for every image type.
    zero_shot_data_type = "coco" if args.data_name == "cocobias" else None
    summarize_runs(
        args.exp_type, args.data_name, args.image_type, args.model_name, args.model_name,
        args.instruction, zero_shot_data_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="1", choices=sorted(EXP_FOLDERS))
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAP))
    parser.add_argument("--data_name", type=str, default="cocobias", choices=CAPTION_DATASETS)
    parser.add_argument("--image_type", type=str, default="coco", choices=CAPTION_IMAGE_TYPES)
    parser.add_argument("--instruction", type=str, default="I1", choices=list(CAPTION_INSTRUCTIONS))
    args = parser.parse_args()
    present_args(args)
    main(args)
