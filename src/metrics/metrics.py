import numpy as np


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> np.ndarray:
    topk = np.argpartition(-y_pred, kth=k, axis=-1)[:, :k]
    hits = np.take_along_axis(y_true, topk, axis=-1).sum()
    maxhit = np.minimum(y_true.astype(bool).sum(axis=-1), k).sum()  # make sure it's boolean
    return hits/maxhit


# def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> np.ndarray:
#
#     penalties = 1. / np.log2(np.arange(2, k + 2))
#
#     topk_pred = np.argpartition(-y_pred, kth=k, axis=-1)[:, :k]
#     topk_true = np.argpartition(-y_true, kth=k, axis=-1)[:, :k]
#
#     selected = np.take_along_axis(y_true, topk_pred, axis=-1)
#     dcg = (np.array(selected) * penalties).sum(axis=-1)
#
#     selected = np.take_along_axis(y_true, topk_true, axis=-1)
#     idcg = (np.array(selected) * penalties).sum(axis=-1)
#
#     return np.mean(dcg.sum(axis=-1) / idcg.sum(axis=-1))
