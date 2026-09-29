import pandas as pd

from src.metric.caption import gender_rates

REVEALED_LABELS = ["masculine", "feminine"]


def pronoun_error_rates(true_g: pd.Series, pred_g: pd.Series) -> dict[str, float]:
    # Only cross-gender predictions are errors; a neutral pronoun is not penalized.
    wrong = (
        ((true_g == "masculine") & (pred_g == "feminine"))
        | ((true_g == "feminine") & (pred_g == "masculine"))
    )
    mr_m, mr_f, mr_o = gender_rates(true_g, wrong, "masculine", "feminine")

    return {
        "MR-M-F": float((mr_m - mr_f) * 100),
        "MR-M":   float(mr_m * 100),
        "MR-F":   float(mr_f * 100),
        "MR-O":   float(mr_o * 100),
    }


def pronoun_reveal_rates(true_g: pd.Series, pred_g: pd.Series) -> dict[str, float]:
    revealed = pred_g.isin(REVEALED_LABELS)
    rr_m, rr_f, rr_o = gender_rates(true_g, revealed, "masculine", "feminine")

    return {
        "RR-M": float(rr_m * 100),
        "RR-F": float(rr_f * 100),
        "RR-O": float(rr_o * 100),
    }
