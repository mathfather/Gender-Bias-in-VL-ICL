# https://huggingface.co/OpenGVLab/InternVL3_5-8B

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.model.base import ModelHelper
from src.prompt import PRONOUN_MAP, PRONOUN_QUERY_TEMPLATE, PRONOUN_TEMPLATE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
# Id of "<|im_end|>"; the checkpoint's generation_config.json sets no eos_token_id.
IM_END_TOKEN_ID = 151645


def _build_transform(input_size: int=448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int=1,
    max_num: int=12,
    image_size: int=448,
    use_thumbnail: bool=False,
) -> list[Image.Image]:
    """Split image into image_size tiles on the grid closest to its aspect ratio."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _load_internvl_image(image_file: str, input_size: int=448, max_num: int=12) -> torch.Tensor:
    """Pixel values of the image tiles, shape [num_tiles, 3, input_size, input_size]."""
    image = Image.open(image_file).convert("RGB")
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


class InternVL35Helper(ModelHelper):
    HIS_TOKEN_ID = 25235
    HER_TOKEN_ID = 1923

    @classmethod
    def load(cls, model_path: str, device: str="cuda"):
        model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        return cls(model, tokenizer)

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def _load_pixel_values(self, image_list: list[str]) -> tuple[torch.Tensor, list[int]]:
        all_pixel_values = []
        num_patches_list = []
        for image_file in image_list:
            tiles = _load_internvl_image(image_file)
            tiles = tiles.to(dtype=torch.bfloat16, device=self.model.device)
            all_pixel_values.append(tiles)
            num_patches_list.append(tiles.size(0))
        pixel_values = torch.cat(all_pixel_values, dim=0)
        return pixel_values, num_patches_list

    def _build_prompt(self, model_input: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(input_ids, attention_mask) on the model device; sets model.img_context_token_id."""
        history = model_input["history"]
        question = model_input["question"]
        num_patches_list = model_input["num_patches_list"]
        query_prefix = model_input.get("query_prefix")

        template = self.model.conv_template.copy()
        template.system_message = self.model.system_message

        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt()

        if query_prefix:
            # The space before query_prefix is part of the published prompt.
            prompt += " " + query_prefix

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            prompt = prompt.replace("<image>", image_tokens, 1)

        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.model.device)
        attention_mask = model_inputs["attention_mask"].to(self.model.device)
        return input_ids, attention_mask

    def prepare_caption_input(
        self,
        caption_list: list[str],
        image_list: list[str],
        instruction: str,
    ) -> dict:
        pixel_values, num_patches_list = self._load_pixel_values(image_list)
        history = [(f"<image>\n{instruction}", shot_caption) for shot_caption in caption_list[:-1]]
        return {
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
            "history": history,
            "question": f"<image>\n{instruction}",
        }

    def prepare_pronoun_input(
        self,
        occupations: list[str],
        genders: list[str],
        objects: list[str],
        image_list: list[str],
    ) -> dict:
        pixel_values, num_patches_list = self._load_pixel_values(image_list)
        history = []
        for occupation, gender, obj in zip(
            occupations[:-1], genders[:-1], objects[:-1], strict=True
        ):
            pronoun = PRONOUN_MAP[gender]
            history.append(("<image>", PRONOUN_TEMPLATE.format(occupation, pronoun, obj)))
        return {
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
            "history": history,
            "question": "<image>",
            "query_prefix": PRONOUN_QUERY_TEMPLATE.format(occupations[-1]),
        }

    def generate(self, model_input: dict, max_new_tokens: int) -> str:
        input_ids, attention_mask = self._build_prompt(model_input)
        output = self.model.generate(
            pixel_values=model_input["pixel_values"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            min_new_tokens=1,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            length_penalty=1.0,
            num_return_sequences=1,
            eos_token_id=IM_END_TOKEN_ID,
            pad_token_id=IM_END_TOKEN_ID,
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

    def sample_next_token(self, model_input: dict) -> tuple[str, float, float]:
        input_ids, attention_mask = self._build_prompt(model_input)
        with torch.no_grad():
            vit_embeds = self.model.extract_feature(model_input["pixel_values"])
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            batch, seq_len, hidden = input_embeds.shape
            flat_embeds = input_embeds.reshape(batch * seq_len, hidden)
            selected = input_ids.reshape(batch * seq_len) == self.model.img_context_token_id
            flat_embeds[selected] = vit_embeds.reshape(-1, hidden).to(flat_embeds.device)
            input_embeds = flat_embeds.reshape(batch, seq_len, hidden)
            out = self.model.language_model(
                inputs_embeds=input_embeds, attention_mask=attention_mask
            )
            next_logits = out.logits[:, -1, :]
            return self._classify_gender(next_logits)
