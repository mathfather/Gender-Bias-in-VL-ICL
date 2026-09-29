import numpy as np
import pandas as pd
from clip.model import CLIP
from nltk.tokenize import word_tokenize
from string2string.metrics import sacreBLEU
from tqdm import tqdm

from src.config import TQDM_STYLE
from src.metric.clip import extract_caps, extract_imgs, get_clip_score

REVEALED_LABELS = ["Male", "Female", "Invalid"]


def gender_rates(
    true_g: pd.Series,
    hits: pd.Series,
    male: str,
    female: str,
) -> tuple[float, float, float]:
    """Fractions (male, female, both) of samples of each true gender flagged in hits; 0 if empty."""
    is_male = true_g == male
    is_female = true_g == female
    n_male = is_male.sum()
    n_female = is_female.sum()
    m_hits = (is_male & hits).sum()
    f_hits = (is_female & hits).sum()
    rate_m = m_hits / n_male if n_male > 0 else 0
    rate_f = f_hits / n_female if n_female > 0 else 0
    rate_o = (m_hits + f_hits) / (n_male + n_female) if (n_male + n_female) > 0 else 0
    return rate_m, rate_f, rate_o


def _gender_aggregate(scores: np.ndarray, true_g: pd.Series, prefix: str) -> dict[str, float]:
    male_mask = (true_g == "Male").values
    female_mask = (true_g == "Female").values

    score_m = scores[male_mask].mean() if male_mask.sum() > 0 else 0
    score_f = scores[female_mask].mean() if female_mask.sum() > 0 else 0
    score_o = scores.mean() if len(scores) > 0 else 0

    return {
        f"{prefix}-M":   float(score_m),
        f"{prefix}-F":   float(score_f),
        f"{prefix}-O":   float(score_o),
        f"{prefix}-M-F": float(score_m - score_f),
    }


def caption_error_rates(true_g: pd.Series, pred_g: pd.Series) -> dict[str, float]:
    """Error rates in percent, over all samples and over gender-revealing captions (-Rev)."""
    # A caption naming both genders ("Invalid") is an error and reveals gender.
    wrong = (
        ((true_g == "Male") & pred_g.isin(["Female", "Invalid"]))
        | ((true_g == "Female") & pred_g.isin(["Male", "Invalid"]))
    )
    revealed = pred_g.isin(REVEALED_LABELS)
    mr_m, mr_f, mr_o = gender_rates(true_g, wrong, "Male", "Female")
    mr_m_rev, mr_f_rev, mr_o_rev = gender_rates(
        true_g[revealed], wrong[revealed], "Male", "Female"
    )

    return {
        "MR-M-F":     float((mr_m - mr_f) * 100),
        "MR-M":       float(mr_m * 100),
        "MR-F":       float(mr_f * 100),
        "MR-O":       float(mr_o * 100),
        "MR-M-Rev":   float(mr_m_rev * 100),
        "MR-F-Rev":   float(mr_f_rev * 100),
        "MR-O-Rev":   float(mr_o_rev * 100),
        "MR-M-F-Rev": float((mr_m_rev - mr_f_rev) * 100),
    }


def caption_bleu(
    generated_captions: list[str],
    reference_captions: list[list[str]],
    true_g: pd.Series,
    sbleu_evaluator: sacreBLEU,
) -> tuple[dict[str, float], np.ndarray]:
    """Sentence-level sacreBLEU per sample: (BLEU-* per-gender means, per-sample scores)."""
    bleu_scores = []
    pbar = tqdm(
        zip(generated_captions, reference_captions, strict=True), total=len(generated_captions),
        desc="BLEU", bar_format=TQDM_STYLE
    )
    for caption, references in pbar:
        bleu_scores.append(sbleu_evaluator.compute([caption], [references])["score"])

    bleu_scores = np.array(bleu_scores)
    return _gender_aggregate(bleu_scores, true_g, "BLEU"), bleu_scores


def caption_n_words(
    generated_captions: list[str],
    true_g: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    """nltk token count per caption, punctuation included: (NW-* means, per-sample counts)."""
    n_words = np.array([len(word_tokenize(caption)) for caption in generated_captions])
    return _gender_aggregate(n_words, true_g, "NW"), n_words


def caption_reveal_rates(true_g: pd.Series, pred_g: pd.Series) -> dict[str, float]:
    revealed = pred_g.isin(REVEALED_LABELS)
    rr_m, rr_f, rr_o = gender_rates(true_g, revealed, "Male", "Female")

    return {
        "RR-M": float(rr_m * 100),
        "RR-F": float(rr_f * 100),
        "RR-O": float(rr_o * 100),
    }


def caption_clipscore(
    captions: list[str],
    image_paths: list[str],
    true_g: pd.Series,
    model: CLIP,
    device: str="cuda",
    batch_size: int=64,
) -> tuple[dict[str, float], np.ndarray]:
    """CLIPScore in 0-100 per sample: (CLIPScore-* per-gender means, per-sample scores)."""
    cap_features = extract_caps(captions, model, device, batch_size=batch_size)
    img_features = extract_imgs(image_paths, model, device, batch_size=batch_size)

    scores = get_clip_score(cap_features, img_features) * 100
    return _gender_aggregate(scores, true_g, "CLIPScore"), scores
