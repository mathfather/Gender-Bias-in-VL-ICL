import numpy as np


def similarity_sampling(num_shot: int, sorted_train_ids: np.ndarray) -> np.ndarray:
    """Top num_shot ids per query row, reversed so the most similar sits next to the query."""
    return sorted_train_ids[:, :num_shot][:, ::-1]
