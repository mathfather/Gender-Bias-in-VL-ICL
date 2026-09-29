from abc import ABC, abstractmethod

import torch

from src.config import QA_MODELS
from src.prompt import QA_INSTRUCTIONS, QA_QUERY_TEMPLATE, QA_RESPONSE_TEMPLATES


def format_qa_turns(samples: list[dict], prompt_mode: str) -> tuple[list[tuple[str, str]], str]:
    """Shot (user text, response) pairs and the query user text; samples[-1] is the query."""
    instruction = QA_INSTRUCTIONS[prompt_mode]
    template = QA_RESPONSE_TEMPLATES[prompt_mode]
    shots = []
    for sample in samples[:-1]:
        user_text = QA_QUERY_TEMPLATE.format(question=sample["question"], instruction=instruction)
        response = template.format(answer=sample["answer"], thought=sample["thought"])
        shots.append((user_text, response))
    query_text = QA_QUERY_TEMPLATE.format(question=samples[-1]["question"], instruction=instruction)
    return shots, query_text


class ModelHelper(ABC):
    """Contract shared by the LVLM wrappers; each subclass sets the token ids of "his" and "her"."""

    # From tokenizer(word, add_special_tokens=False) of each model's tokenizer.
    HIS_TOKEN_ID: int
    HER_TOKEN_ID: int

    @classmethod
    @abstractmethod
    def load(cls, model_path: str, device: str="cuda"):
        """Load the checkpoint at model_path onto device and return a helper.

        Args:
            model_path: Local snapshot folder under models/.
            device: Device map string passed to from_pretrained.

        Returns:
            A helper holding the loaded model and its tokenizer or processor.
        """

    @abstractmethod
    def prepare_caption_input(
        self,
        caption_list: list[str],
        image_list: list[str],
        instruction: str,
    ) -> object:
        """Build the captioning input.

        Args:
            caption_list: Shot captions followed by the query's reference caption (unused).
            image_list: Image paths aligned with caption_list.
            instruction: Caption instruction text attached to every image.

        Returns:
            Model-specific input for ``generate``.
        """

    @abstractmethod
    def prepare_pronoun_input(
        self,
        occupations: list[str],
        genders: list[str],
        objects: list[str],
        image_list: list[str],
    ) -> object:
        """Build the pronoun prediction input.

        Args:
            occupations: Occupation strings, shots followed by the query.
            genders: Perceived genders of the shots (last entry unused).
            objects: Object or participant strings of the shots (last entry unused).
            image_list: Image paths aligned with occupations.

        Returns:
            Model-specific input for ``sample_next_token``.
        """

    def prepare_qa_input(
        self,
        samples: list[dict],
        image_list: list[str],
        prompt_mode: str,
    ) -> object:
        """Build the VQA input; only the helpers of QA_MODELS override this.

        Args:
            samples: Dicts with question, answer and thought, shots followed by the query.
            image_list: Image paths aligned with samples.
            prompt_mode: "advanced" (CoT) or "base" (direct).

        Returns:
            Model-specific input for ``generate``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is not evaluated on VQA; use one of {QA_MODELS}."
        )

    @abstractmethod
    def generate(self, model_input: object, max_new_tokens: int) -> str:
        """Greedy-decode up to max_new_tokens and return the generated text."""

    @abstractmethod
    def sample_next_token(self, model_input: object) -> tuple[str, float, float]:
        """Return (pred_gender, prob_his, prob_her) for the next token after the query prefix."""

    def _classify_gender(self, next_logits: torch.Tensor) -> tuple[str, float, float]:
        probs = torch.softmax(next_logits, dim=-1)
        prob_his = probs[0, self.HIS_TOKEN_ID].item()
        prob_her = probs[0, self.HER_TOKEN_ID].item()
        if prob_his > prob_her:
            pred_gender = "masculine"
        elif prob_his < prob_her:
            pred_gender = "feminine"
        else:
            pred_gender = "neutral"
        return pred_gender, prob_his, prob_her
