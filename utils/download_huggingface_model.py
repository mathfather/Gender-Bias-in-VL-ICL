import argparse
import os

from huggingface_hub import snapshot_download

from src.config import MODEL
from src.utils import present_args

# The two loaders that expect a folder other than the repo name; must match MODEL_MAP in
# src/model/utils.py.
FOLDER_OVERRIDES = {
    "Qwen/Qwen-VL":          "QwenVL",
    "openbmb/MiniCPM-o-2_6": "MiniCPM",
}


def main(args: argparse.Namespace) -> None:
    folder = FOLDER_OVERRIDES.get(args.repo_id, os.path.split(args.repo_id)[1])
    local_dir = os.path.join(MODEL, folder)
    snapshot_download(repo_id=args.repo_id, local_dir=local_dir)
    print(f"Files have been downloaded to {local_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", type=str, required=True,
        help="Hugging Face repo id, e.g. OpenGVLab/InternVL3_5-8B"
    )
    args = parser.parse_args()
    present_args(args)
    main(args)
