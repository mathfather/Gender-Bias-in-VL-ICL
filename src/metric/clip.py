import clip
import numpy as np
import torch
from clip.model import CLIP
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
from tqdm import tqdm

from src.config import TQDM_STYLE

# Adapted from https://github.com/jmhessel/clipscore.

NUM_WORKERS = 8
# Fixed 224px preprocessing matches the ViT-B/32 CLIP the callers load.
NUM_PIXEL = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


class CLIPCapDataset(Dataset):
    def __init__(self, data: list[str]) -> None:
        self.data = data

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"caption": clip.tokenize(self.data[index], truncate=True).squeeze()}

    def __len__(self) -> int:
        return len(self.data)


class CLIPImageDataset(Dataset):
    def __init__(self, image_paths: list[str]) -> None:
        self.image_paths = image_paths
        self.preprocess = Compose([
            Resize(NUM_PIXEL, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(NUM_PIXEL),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = Image.open(self.image_paths[index])
        return {"image": self.preprocess(image)}

    def __len__(self) -> int:
        return len(self.image_paths)


def extract_caps(
    captions: list[str],
    model: CLIP,
    device: str="cuda",
    batch_size: int=256,
) -> np.ndarray:
    data = DataLoader(
        CLIPCapDataset(captions), batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False
    )
    all_text_features = []
    with torch.no_grad():
        for batch in tqdm(data, desc="Extract cap feat.", bar_format=TQDM_STYLE):
            text_features = model.encode_text(batch["caption"].to(device))
            all_text_features.append(text_features.cpu().numpy())
    return np.vstack(all_text_features)


def extract_imgs(
    image_paths: list[str],
    model: CLIP,
    device: str="cuda",
    batch_size: int=64,
) -> np.ndarray:
    data = DataLoader(
        CLIPImageDataset(image_paths), batch_size=batch_size, num_workers=NUM_WORKERS,
        shuffle=False
    )
    all_image_features = []
    with torch.no_grad():
        for batch in tqdm(data, desc="Extract img feat.", bar_format=TQDM_STYLE):
            batch_images = batch["image"].to(device)
            if torch.device(device).type == "cuda":
                batch_images = batch_images.to(torch.float16)
            image_features = model.encode_image(batch_images)
            all_image_features.append(image_features.cpu().numpy())
    return np.vstack(all_image_features)


def get_clip_score(cap_features: np.ndarray, img_features: np.ndarray) -> np.ndarray:
    """Per-row max(cos, 0) of the paired features: CLIPScore with weight 1.0, before the x100."""
    img_features = img_features / np.sqrt(np.sum(img_features**2, axis=1, keepdims=True) + 1e-8)
    cap_features = cap_features / np.sqrt(np.sum(cap_features**2, axis=1, keepdims=True) + 1e-8)
    img_features = img_features.astype(np.float32)
    cap_features = cap_features.astype(np.float32)
    cos_sim = np.sum(img_features * cap_features, axis=1)
    return np.clip(cos_sim, a_min=0, a_max=None)
