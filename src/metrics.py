from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Set

import numpy as np
import torch


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Root-mean-square error between y_true and y_pred.
    Accepts lists, numpy arrays, or tensors (will be converted to numpy).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shapes do not match for rmse: {y_true_arr.shape} vs {y_pred_arr.shape}"
        )
    mse = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(math.sqrt(mse))


def precision_at_k_single(
    recommended: Sequence[int], true_set: Set[int], k: int
) -> float:
    """
    Precision@k for a single user: fraction of top-k recommended items that are relevant.
    `recommended` expected to be ordered list/array of item ids.
    """
    if k <= 0:
        return 0.0
    recs = list(recommended)[:k]
    if len(recs) == 0:
        return 0.0
    hits = sum(1 for item in recs if int(item) in true_set)
    return hits / float(len(recs))


def recall_at_k_single(
    recommended: Sequence[int], true_set: Set[int], k: int
) -> float:
    """
    Recall@k for a single user: fraction of relevant items that are present in top-k recommendations.
    If true_set is empty, returns 0.0.
    """
    if len(true_set) == 0:
        return 0.0
    recs = list(recommended)[:k]
    hits = sum(1 for item in recs if int(item) in true_set)
    return hits / float(len(true_set))


def ndcg_at_k_single(
    recommended: Sequence[int], true_set: Set[int], k: int
) -> float:
    """
    NDCG@k for a single user.
    DCG = sum_{i=1..k} rel_i / log2(i+1) where rel_i is 1 if recommended[i-1] in true_set else 0
    IDCG is sum of ideal gains (i.e., min(len(true_set), k) ones at top).
    """
    if k <= 0:
        return 0.0
    recs = list(recommended)[:k]
    gains = [1.0 if int(item) in true_set else 0.0 for item in recs]
    dcg = 0.0
    for idx, g in enumerate(gains):
        if g:
            dcg += g / math.log2(idx + 2)  # idx starts at 0, position = idx+1
    ideal_hits = min(len(true_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def precision_recall_ndcg_for_aligned(
    users: Sequence[int],
    recommended: Sequence[Sequence[int]],
    ground_truth_sets: Dict[int, Set[int]],
    k: int,
) -> Dict[str, float]:
    """
    Compute averaged precision@k, recall@k, ndcg@k for an aligned list of users.

    - users: sequence of user indices (length N)
    - recommended: sequence (length N) where each element is sequence/list/array of top-K item ids in order
    - ground_truth_sets: dict mapping user_idx -> set(item_idx) (true relevant items for evaluation)
    - k: top-k cutoff

    Returns dictionary: {"precision": float, "recall": float, "ndcg": float}
    """
    if len(users) != len(recommended):
        raise ValueError("Length of users and recommended must match")

    precisions = []
    recalls = []
    ndcgs = []
    for idx, u in enumerate(users):
        recs = recommended[idx]
        true_set = ground_truth_sets.get(int(u), set())
        if len(true_set) == 0:
            # skip users with no ground truth to avoid inflating metrics
            continue
        precisions.append(precision_at_k_single(recs, true_set, k))
        recalls.append(recall_at_k_single(recs, true_set, k))
        ndcgs.append(ndcg_at_k_single(recs, true_set, k))

    if len(precisions) == 0:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "ndcg": float(np.mean(ndcgs)),
    }


def evaluate_ranking_full(
    model,
    users: Sequence[int],
    ground_truth_sets: Dict[int, Set[int]],
    n_items: int,
    k: int = 10,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute full-ranking evaluation for given users by scoring all items.

    - model: PyTorch model with signature model(user_tensor, item_tensor) -> scores (batch,)
    - users: sequence of user indices to evaluate (length U)
    - ground_truth_sets: dict user_idx -> set(true_item_idx)
    - n_items: total number of items (indexed 0..n_items-1)
    - k: top-k to compute
    - batch_size: how many items to score at once per user (affects memory)
    - device: torch.device or device string; if None, use model device.

    Returns: dict with keys {"precision@k", "recall@k", "ndcg@k"} (averaged across users with non-empty ground truth).
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    users_arr = np.array(list(map(int, users)), dtype=np.int64)
    all_items = np.arange(n_items, dtype=np.int64)

    recommended_list = []
    # Score per user in loops; for medium-size datasets this is fine (MovieLens-100k n_items ~1.6k)
    model_device = device
    model.to(model_device)
    model.eval()
    with torch.no_grad():
        for u in users_arr:
            # accumulate scores for all items in chunks
            scores_chunks = []
            for start in range(0, n_items, batch_size):
                end = min(n_items, start + batch_size)
                items_chunk = all_items[start:end]
                users_chunk = np.full(
                    shape=(len(items_chunk),),
                    fill_value=int(u),
                    dtype=np.int64,
                )
                users_t = torch.from_numpy(users_chunk).long().to(model_device)
                items_t = torch.from_numpy(items_chunk).long().to(model_device)
                chunk_scores = model(
                    users_t, items_t
                )  # expects (batch,) tensor
                scores_chunks.append(chunk_scores.cpu().numpy())
            scores = np.concatenate(scores_chunks, axis=0)  # shape (n_items,)
            # get top-k item indices
            if k >= n_items:
                topk_idx = np.argsort(-scores)[:k]
            else:
                # partial sort for performance
                topk_idx = np.argpartition(-scores, k - 1)[:k]
                # sort topk for ranking order
                topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
            recommended_items = all_items[topk_idx]
            recommended_list.append(recommended_items)

    recommended_arr = np.stack(recommended_list, axis=0)  # (U, k)
    users_for_eval = users_arr
    metrics = precision_recall_ndcg_for_aligned(
        users_for_eval, recommended_arr, ground_truth_sets, k
    )
    return {
        "precision@k": metrics["precision"],
        "recall@k": metrics["recall"],
        "ndcg@k": metrics["ndcg"],
    }
