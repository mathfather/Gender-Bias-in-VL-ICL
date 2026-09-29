import importlib
import os

from PIL import Image

from src.config import MODEL
from src.model.base import ModelHelper

MODEL_MAP: dict[str, str] = {
    "QwenVL":     "QwenVL",
    "MiniCPM":    "MiniCPM",
    "Phi35V":     "Phi-3.5-vision-instruct",
    "idefics3":   "Idefics3-8B-Llama3",
    "InternVL35": "InternVL3_5-8B",
    "Qwen3VL":    "Qwen3-VL-8B-Instruct",
}

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "QwenVL":     ("src.model.qwenvl",     "QwenVLHelper"),
    "MiniCPM":    ("src.model.minicpm",    "MiniCPMHelper"),
    "Phi35V":     ("src.model.phi35v",     "Phi35VHelper"),
    "idefics3":   ("src.model.idefics3",   "Idefics3Helper"),
    "InternVL35": ("src.model.internvl35", "InternVL35Helper"),
    "Qwen3VL":    ("src.model.qwen3vl",    "Qwen3VLHelper"),
}


def load_image(image_file: str) -> Image.Image:
    return Image.open(image_file).convert("RGB")


def load_images(image_files: list[str]) -> list[Image.Image]:
    return [load_image(image_file) for image_file in image_files]


def load_model(model_name: str, device: str="cuda") -> ModelHelper:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'; expected one of {sorted(MODEL_REGISTRY)}.")
    module_path, class_name = MODEL_REGISTRY[model_name]
    helper_cls = getattr(importlib.import_module(module_path), class_name)
    return helper_cls.load(os.path.join(MODEL, MODEL_MAP[model_name]), device)
