import argparse
from types import SimpleNamespace

from src.config import (
    CAPTION_DATASETS,
    CAPTION_IMAGE_TYPES,
    EXP_FOLDERS,
    NUM_SHOTS,
    select_shot_seeds,
)
from src.data import load_train_data, load_valid_data
from src.inference.caption import caption_exp
from src.model import MODEL_MAP, load_model
from src.prompt import CAPTION_INSTRUCTIONS
from src.utils import check_caption_run, present_args, select_device


def main(args: argparse.Namespace) -> None:
    check_caption_run(args.exp_type, args.data_name, args.image_type)
    model_helper = load_model(args.model_name, device=select_device(args.gpu_id))
    valid_data = load_valid_data(args.data_name)
    train_data = load_train_data(args.data_name, image_type=args.image_type)
    for num_shot in args.num_shot:
        for shot_seed in select_shot_seeds(args.exp_type, num_shot):
            run_args = SimpleNamespace(
                shot_seed=shot_seed,
                gpu_id=args.gpu_id,
                exp_type=args.exp_type,
                model_name=args.model_name,
                data_name=args.data_name,
                image_type=args.image_type,
                num_shot=num_shot,
                instruction=args.instruction,
            )
            present_args(run_args)
            caption_exp(run_args, model_helper, valid_data, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--exp_type", type=str, default="1", choices=sorted(EXP_FOLDERS))
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAP))
    parser.add_argument("--image_type", type=str, default="coco", choices=CAPTION_IMAGE_TYPES)
    parser.add_argument("--data_name", type=str, default="cocobias", choices=CAPTION_DATASETS)
    parser.add_argument("--instruction", type=str, default="I1", choices=list(CAPTION_INSTRUCTIONS))
    parser.add_argument("--num_shot", type=int, nargs="+", default=NUM_SHOTS, choices=NUM_SHOTS)
    args = parser.parse_args()
    present_args(args)
    main(args)
