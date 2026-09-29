# https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

import os

import torch
from transformers import (
    AutoProcessor,
    BatchFeature,
    ProcessorMixin,
    Qwen3VLForConditionalGeneration,
)

from src.model.base import ModelHelper, format_qa_turns
from src.prompt import PRONOUN_MAP, PRONOUN_QUERY_TEMPLATE, PRONOUN_TEMPLATE


class Qwen3VLHelper(ModelHelper):
    HIS_TOKEN_ID = 25235
    HER_TOKEN_ID = 1923

    @classmethod
    def load(cls, model_path: str, device: str="cuda"):
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).eval()
        return cls(model, processor)

    def __init__(self, model: Qwen3VLForConditionalGeneration, processor: ProcessorMixin) -> None:
        self.model = model
        self.processor = processor

    @staticmethod
    def _image_content(image_path: str) -> dict:
        return {"type": "image", "image": os.path.abspath(image_path)}

    def _apply_chat_template(self, messages: list[dict]) -> BatchFeature:
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def prepare_caption_input(
        self,
        caption_list: list[str],
        image_list: list[str],
        instruction: str,
    ) -> dict:
        messages = []
        for shot_caption, image in zip(caption_list[:-1], image_list[:-1], strict=True):
            messages.append({
                "role": "user",
                "content": [self._image_content(image), {"type": "text", "text": instruction}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": shot_caption}]
            })
        messages.append({
            "role": "user",
            "content": [self._image_content(image_list[-1]), {"type": "text", "text": instruction}]
        })
        return {"text": messages}

    def prepare_pronoun_input(
        self,
        occupations: list[str],
        genders: list[str],
        objects: list[str],
        image_list: list[str],
    ) -> dict:
        messages = []
        for image, occupation, gender, obj in zip(
            image_list[:-1], occupations[:-1], genders[:-1], objects[:-1], strict=True
        ):
            pronoun = PRONOUN_MAP[gender]
            messages.append({"role": "user", "content": [self._image_content(image)]})
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": PRONOUN_TEMPLATE.format(occupation, pronoun, obj)}
                ]
            })
        messages.append({"role": "user", "content": [self._image_content(image_list[-1])]})
        return {
            "text": messages,
            "query_prefix": PRONOUN_QUERY_TEMPLATE.format(occupations[-1]),
        }

    def prepare_qa_input(
        self,
        samples: list[dict],
        image_list: list[str],
        prompt_mode: str,
    ) -> dict:
        shots, query_text = format_qa_turns(samples, prompt_mode)
        messages = []
        for (user_text, response), image in zip(shots, image_list[:-1], strict=True):
            messages.append({
                "role": "user",
                "content": [self._image_content(image), {"type": "text", "text": user_text}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })
        messages.append({
            "role": "user",
            "content": [self._image_content(image_list[-1]), {"type": "text", "text": query_text}]
        })
        return {"text": messages}

    def generate(self, model_input: dict, max_new_tokens: int) -> str:
        inputs = self._apply_chat_template(model_input["text"]).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            min_new_tokens=1,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            length_penalty=1.0,
            num_return_sequences=1,
        )
        return self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    def sample_next_token(self, model_input: dict) -> tuple[str, float, float]:
        inputs = self._apply_chat_template(model_input["text"])
        prefix_ids = self.processor.tokenizer(
            model_input["query_prefix"], add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(prefix_ids)], dim=1
        )
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, logits_to_keep=1)
            next_logits = out.logits[:, -1, :]
            return self._classify_gender(next_logits)
