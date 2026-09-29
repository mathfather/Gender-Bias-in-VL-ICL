import numpy as np
import pandas as pd


def qa_error_rates(scores: np.ndarray, true_g: pd.Series) -> dict[str, float]:
    """Error rates in percent with per-sample error 1 - judge score, rounded to 2 decimals."""
    male_mask = (true_g == "Male").values
    female_mask = (true_g == "Female").values

    mr_m = (1 - scores[male_mask].mean()) * 100 if male_mask.sum() > 0 else 0.0
    mr_f = (1 - scores[female_mask].mean()) * 100 if female_mask.sum() > 0 else 0.0
    mr_o = (1 - scores.mean()) * 100 if len(scores) > 0 else 0.0

    return {
        "MR-M":   round(float(mr_m), 2),
        "MR-F":   round(float(mr_f), 2),
        "MR-O":   round(float(mr_o), 2),
        "MR-M-F": round(float(mr_m - mr_f), 2),
    }
