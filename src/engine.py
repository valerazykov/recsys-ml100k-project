from __future__ import annotations

import logging
import math
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src import metrics as metrics_mod

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    save_dir: str,
    config: Optional[dict] = None,
) -> None:
    """
    Save optimizer state and epoch, plus model weights via model.save_pretrained if available.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    state = {"epoch": epoch}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    try:
        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint metadata to {ckpt_path}")
    except Exception:
        logger.exception("Failed to save checkpoint metadata.")

    # save model weights using HF-style helper if available
    try:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(save_dir, config=config)
            logger.info(
                f"Saved model weights/config via save_pretrained to {save_dir}"
            )
        else:
            model_path = os.path.join(save_dir, "pytorch_model.bin")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved model.state_dict() to {model_path}")
    except Exception:
        logger.exception(
            "Failed to save model weights via save_pretrained; attempted state_dict fallback."
        )


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    model_type: str = "mf",
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = False,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """
    Run one training epoch.

    model_type: "mf"/"ncf"/"rating" => pointwise regression (users, items, ratings)
                "bpr"/"pairwise" => pairwise (users, pos_items, neg_items)

    Returns dict with aggregated metrics: {"loss": float, "examples": int}
    """
    model.train()
    device = torch.device(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_loss = 0.0
    n_examples = 0

    pbar = tqdm(dataloader, desc=f"Train epoch {epoch}", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        if model_type.lower() in ("mf", "ncf", "rating", "pointwise"):
            users, items, ratings = batch
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(users, items)
                if loss_fn is None:
                    loss = nn.MSELoss()(preds, ratings)
                else:
                    loss = loss_fn(preds, ratings)
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
            batch_loss = float(loss.item())
            batch_size = ratings.shape[0]
            total_loss += batch_loss * batch_size
            n_examples += batch_size
            pbar.set_postfix_str(f"loss={batch_loss:.4f}")
        elif model_type.lower() in ("bpr", "pairwise"):
            users, pos, neg = batch
            users = users.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pos_scores = model(users, pos)  # (B,)
                neg_scores = model(users, neg)  # (B,)
                diff = pos_scores - neg_scores
                loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
            batch_loss = float(loss.item())
            batch_size = users.shape[0]
            total_loss += batch_loss * batch_size
            n_examples += batch_size
            pbar.set_postfix_str(f"bpr_loss={batch_loss:.4f}")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    avg_loss = total_loss / max(1, n_examples)
    return {"loss": float(avg_loss), "examples": int(n_examples)}


def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    model_type: str = "mf",
    compute_full_ranking: bool = False,
    topk: int = 10,
    n_items: Optional[int] = None,
    eval_full_batch_size: int = 1024,
) -> Dict[str, float]:
    """
    Evaluate model on dataloader.

    If compute_full_ranking=True, will compute top-K metrics (precision@k, recall@k, ndcg@k)
    by scoring all items per user (exact evaluation). Requires n_items to be set.

    Returns dict with keys:
      - loss (for rating) or avg_bpr_loss (for pairwise)
      - rmse (for rating if applicable)
      - precision@k, recall@k, ndcg@k (if ranking computed)
    """
    model.eval()
    device = torch.device(device)
    use_bpr = model_type.lower() in ("bpr", "pairwise")
    total_loss = 0.0
    n_examples = 0

    y_trues = []
    y_preds = []

    # Build ground truth mapping for ranking evaluation (from dataloader.dataset.df if present)
    ground_truth = {}
    try:
        ds = getattr(dataloader, "dataset", None)
        if ds is not None and hasattr(ds, "df"):
            df = ds.df
            for row in df.itertuples(index=False):
                u = int(getattr(row, "user_idx"))
                i = int(getattr(row, "item_idx"))
                ground_truth.setdefault(u, set()).add(i)
    except Exception:
        logger.debug(
            "Could not extract ground truth from dataloader.dataset.df; full ranking may be unavailable."
        )

    # First pass: compute pointwise loss / rmse if possible
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Eval", leave=False)
        for batch in pbar:
            if not use_bpr:
                users, items, ratings = batch
                users = users.to(device)
                items = items.to(device)
                ratings = ratings.to(device)
                preds = model(users, items)
                loss = nn.MSELoss()(preds, ratings)
                total_loss += float(loss.item()) * users.shape[0]
                n_examples += users.shape[0]
                y_trues.append(ratings.cpu().numpy())
                y_preds.append(preds.cpu().numpy())
                pbar.set_postfix_str(
                    f"batch_rmse={float(math.sqrt(float(loss.item()))):.4f}"
                )
            else:
                users, pos, neg = batch
                users = users.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                pos_scores = model(users, pos)
                neg_scores = model(users, neg)
                diff = pos_scores - neg_scores
                loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))
                total_loss += float(loss.item()) * users.shape[0]
                n_examples += users.shape[0]
                pbar.set_postfix_str(f"bpr_loss={float(loss.item()):.4f}")

    metrics: Dict[str, float] = {}
    metrics["loss"] = float(total_loss / n_examples) if n_examples > 0 else 0.0

    # RMSE
    if len(y_trues) > 0:
        y_true_arr = np.concatenate(y_trues, axis=0)
        y_pred_arr = np.concatenate(y_preds, axis=0)
        metrics["rmse"] = metrics_mod.rmse(y_true_arr, y_pred_arr)

    # Full-ranking evaluation using src.metrics.evaluate_ranking_full
    if compute_full_ranking:
        if n_items is None:
            raise ValueError(
                "n_items must be provided for full ranking evaluation."
            )
        if len(ground_truth) == 0:
            raise ValueError(
                "Could not extract ground truth for ranking evaluation from dataloader.dataset.df"
            )

        users_for_eval = sorted(list(ground_truth.keys()))
        # call evaluate_ranking_full from metrics module
        ranking_res = metrics_mod.evaluate_ranking_full(
            model=model,
            users=users_for_eval,
            ground_truth_sets=ground_truth,
            n_items=n_items,
            k=topk,
            batch_size=eval_full_batch_size,
            device=device,
        )
        # rename keys to match previous naming
        metrics["precision@k"] = ranking_res.get(
            "precision@k", ranking_res.get("precision", 0.0)
        )
        metrics["recall@k"] = ranking_res.get(
            "recall@k", ranking_res.get("recall", 0.0)
        )
        metrics["ndcg@k"] = ranking_res.get(
            "ndcg@k", ranking_res.get("ndcg", 0.0)
        )

    return metrics
