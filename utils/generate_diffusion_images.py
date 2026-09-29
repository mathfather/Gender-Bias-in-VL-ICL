import argparse
import os
from collections import namedtuple

import torch
from tqdm import tqdm

from src.config import MODEL, RANDOM_SEED, TQDM_STYLE
from src.data import load_train_data
from src.data.cocobias import COCO_IMAGES_DIR, select_gendered_caption
from src.utils import mkdir, present_args, save_json, select_device

SDMPipeline = namedtuple("SDMPipeline", ["class_name", "call_kwargs", "per_image_seed"])

SDM_PIPELINES = {
    "FLUX.1-dev":
        SDMPipeline(
            "FluxPipeline",
            {"guidance_scale": 3.5, "num_inference_steps": 50, "max_sequence_length": 512},
            True
        ),
    "stable-diffusion-3.5-large":
        SDMPipeline(
            "StableDiffusion3Pipeline",
            {"num_inference_steps": 28, "guidance_scale": 3.5},
            False
        ),
}
RESOLUTION = 1024


def main(args: argparse.Namespace) -> None:
    import diffusers

    torch.manual_seed(RANDOM_SEED)
    device = select_device(args.gpu_id)
    sdm = SDM_PIPELINES[args.model_name]
    train_data = load_train_data("cocobias")
    image_folder = os.path.join(COCO_IMAGES_DIR, args.model_name)
    mkdir(image_folder)

    pipeline_cls = getattr(diffusers, sdm.class_name)
    pipe = pipeline_cls.from_pretrained(
        os.path.join(MODEL, args.model_name), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    pipe = pipe.to(device)

    for image_id in tqdm(train_data, desc=f"Generating {args.model_name}", bar_format=TQDM_STYLE):
        entry = train_data[image_id]
        image_name = entry["file_name"]
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            continue
        prompt = select_gendered_caption(entry["caption"], entry["detected_gender"])
        print(f"Image: {image_name} | Gender: {entry['detected_gender']} | Caption: {prompt}")
        call_kwargs = dict(sdm.call_kwargs)
        if sdm.per_image_seed:
            # FLUX restarts a CPU generator for every image. SD3.5L draws from the global RNG
            # seeded once above, so its images depend on how many were generated before them.
            call_kwargs["generator"] = torch.Generator("cpu").manual_seed(RANDOM_SEED)
        image = pipe(prompt=prompt, height=RESOLUTION, width=RESOLUTION, **call_kwargs).images[0]
        image.save(image_path)

    save_json({"seed": RANDOM_SEED, **vars(args)}, os.path.join(image_folder, "args.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=sorted(SDM_PIPELINES))
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    present_args(args)
    main(args)
