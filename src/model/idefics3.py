# https://huggingface.co/docs/transformers/model_doc/idefics3

import torch
from transformers import Idefics3ForConditionalGeneration, Idefics3Processor
from transformers.image_utils import load_image

from src.model.base import ModelHelper
from src.prompt import PRONOUN_MAP, PRONOUN_QUERY_TEMPLATE, PRONOUN_TEMPLATE


class Idefics3Helper(ModelHelper):
    HIS_TOKEN_ID = 26301
    HER_TOKEN_ID = 1964

    @classmethod
    def load(cls, model_path: str, device: str="cuda"):
        processor = Idefics3Processor.from_pretrained(model_path)
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            _attn_implementation="flash_attention_2",
        ).eval()
        return cls(model, processor)

    def __init__(
        self,
        model: Idefics3ForConditionalGeneration,
        processor: Idefics3Processor,
    ) -> None:
        self.model = model
        self.processor = processor

    def prepare_caption_input(
        self,
        caption_list: list[str],
        image_list: list[str],
        instruction: str,
    ) -> dict:
        messages = []
        images = []
        for shot_caption, image in zip(caption_list[:-1], image_list[:-1], strict=True):
            messages.append({
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": instruction}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": shot_caption}]
            })
            images.append(load_image(image))
        messages.append({
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}]
        })
        images.append(load_image(image_list[-1]))
        return {"text": messages, "image": images}

    def prepare_pronoun_input(
        self,
        occupations: list[str],
        genders: list[str],
        objects: list[str],
        image_list: list[str],
    ) -> dict:
        messages = []
        images = []
        for image, occupation, gender, obj in zip(
            image_list[:-1], occupations[:-1], genders[:-1], objects[:-1], strict=True
        ):
            pronoun = PRONOUN_MAP[gender]
            messages.append({"role": "user", "content": [{"type": "image"}]})
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": PRONOUN_TEMPLATE.format(occupation, pronoun, obj)}
                ]
            })
            images.append(load_image(image))
        messages.append({"role": "user", "content": [{"type": "image"}]})
        images.append(load_image(image_list[-1]))
        return {
            "text": messages,
            "image": images,
            "query_prefix": PRONOUN_QUERY_TEMPLATE.format(occupations[-1]),
        }

    def generate(self, model_input: dict, max_new_tokens: int) -> str:
        prompt = self.processor.apply_chat_template(model_input["text"], add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=model_input["image"], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            min_new_tokens=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            length_penalty=1,
            num_return_sequences=1,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text[0].split("\nAssistant: ")[-1]

    def sample_next_token(self, model_input: dict) -> tuple[str, float, float]:
        prompt = self.processor.apply_chat_template(model_input["text"], add_generation_prompt=True)
        # The space before query_prefix is part of the published prompt.
        prompt += " " + model_input["query_prefix"]
        inputs = self.processor(text=prompt, images=model_input["image"], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, logits_to_keep=1)
            next_logits = out.logits[:, -1, :]
            return self._classify_gender(next_logits)
