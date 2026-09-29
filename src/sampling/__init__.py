"""In-context shot selection for the six ICL settings.

select_shot_ids returns the shots of one run: fixed ids for settings 1-4, per-query ids for
settings 5 (SIIR) and 6 (SITR). The retrieval settings read CLIP similarity rankings over the
COCOBias training pool, computed once with:
    python -m src.sampling.sim_retrieval --sim_type II
    python -m src.sampling.sim_retrieval --sim_type IT
"""

from src.sampling.shot_selector import present_shot_ids, select_shot_ids

__all__ = [
    "present_shot_ids",
    "select_shot_ids",
]
