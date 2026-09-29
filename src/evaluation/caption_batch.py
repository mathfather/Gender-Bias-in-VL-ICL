import argparse
from types import SimpleNamespace

from string2string.metrics import sacreBLEU

from src.config import (
    CAPTION_DATASETS,
    CAPTION_IMAGE_TYPES,
    EXP_FOLDERS,
    NUM_SHOTS,
    select_shot_seeds,
)
from src.evaluation.caption import eval_caption, load_clip
from src.model import MODEL_MAP
from src.prompt import CAPTION_INSTRUCTIONS
from src.utils import check_caption_run, present_args, select_device


def main(args: argparse.Namespace) -> None:
    check_caption_run(args.exp_type, args.data_name, args.image_type)
    sbleu = sacreBLEU()
    device = select_device(args.gpu_id)
    clip_model = load_clip(device)

    for model_name in args.model_name:
        for num_shot in args.num_shot:
            for shot_seed in select_shot_seeds(args.exp_type, num_shot):
                run_args = SimpleNamespace(
                    shot_seed=shot_seed, exp_type=args.exp_type, model_name=model_name,
                    data_name=args.data_name, image_type=args.image_type, num_shot=num_shot,
                    instruction=args.instruction, batch_size=args.batch_size
                )
                present_args(run_args)
                eval_caption(run_args, sbleu, clip_model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="1", choices=sorted(EXP_FOLDERS))
    parser.add_argument("--model_name", type=str, nargs="+", required=True, choices=list(MODEL_MAP))
    parser.add_argument("--num_shot", type=int, nargs="+", default=NUM_SHOTS, choices=NUM_SHOTS)
    parser.add_argument("--image_type", type=str, default="coco", choices=CAPTION_IMAGE_TYPES)
    parser.add_argument("--data_name", type=str, default="cocobias", choices=CAPTION_DATASETS)
    parser.add_argument("--instruction", type=str, default="I1", choices=list(CAPTION_INSTRUCTIONS))
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    present_args(args)
    main(args)
