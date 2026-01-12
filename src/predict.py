from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch

import src.model as model_mod


logger = logging.getLogger(__name__)


def _read_config_from_model_dir(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        logger.warning("config.json not found in model_dir %s", model_dir)
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_init_args_from_state(state: dict) -> dict:
    init = {}
    # common MF/NCF names
    if "user_emb.weight" in state and "item_emb.weight" in state:
        u_w = state["user_emb.weight"]
        i_w = state["item_emb.weight"]
        try:
            init["n_users"] = int(u_w.shape[0])
            init["n_items"] = int(i_w.shape[0])
            # try to infer embedding_dim
            init["embedding_dim"] = int(u_w.shape[1]) if len(u_w.shape) > 1 else init.get("embedding_dim")
        except Exception:
            pass
    return init


def load_model_from_dir(
    model_dir: str, map_location: str = "cpu"
) -> Tuple[model_mod.BaseRecModel, Dict[str, Any]]:
    """
    Robust loader: reads config.json if present, tries to get model type and model_init_args.
    If n_users/n_items missing, loads state_dict and infers them from embedding shapes.
    Then builds model via model_mod.build_model_from_cfg and loads weights.
    Returns: model (on CPU by default), cfg_loaded (dict)
    """
    logger.info("Loading model from %s", model_dir)
    cfg_loaded = _read_config_from_model_dir(model_dir)

    # try to find model type and model_init_args in several common layouts
    # layout A: cfg_loaded contains top-level "model": { "type": "...", "model_init_args": {...} }
    model_section = {}
    if isinstance(cfg_loaded, dict) and "model" in cfg_loaded and isinstance(cfg_loaded["model"], dict):
        model_section = cfg_loaded["model"]
    else:
        # maybe config.json already was the model section itself
        model_section = cfg_loaded if isinstance(cfg_loaded, dict) else {}

    mtype = (model_section.get("type") or model_section.get("model_type") or "mf")
    model_init_args = dict(model_section.get("model_init_args") or {})

    # If n_users / n_items missing -> try to infer from saved state dict
    state_path = os.path.join(model_dir, "pytorch_model.bin")
    state = None
    if (model_init_args.get("n_users") is None) or (model_init_args.get("n_items") is None):
        if os.path.exists(state_path):
            try:
                # load to CPU to inspect shapes
                state = torch.load(state_path, map_location="cpu")
                inferred = _infer_init_args_from_state(state)
                # don't overwrite existing values
                for k, v in inferred.items():
                    model_init_args.setdefault(k, v)
                if len(inferred) > 0:
                    logger.info("Inferred model_init_args from state_dict: %s", inferred)
            except Exception as e:
                logger.warning("Failed to load state_dict for inference: %s", e)
        else:
            logger.warning("state file %s not found; cannot infer n_users/n_items", state_path)

    # assemble a model_cfg acceptable for build_model_from_cfg
    model_cfg = {"type": mtype, "model_init_args": model_init_args}
    # copy a few top-level shortcuts if present in model_section to keep behavior consistent
    for k in ("embedding_dim", "ncf_hidden", "hidden_dims", "dropout", "clamp_preds", "min_rating", "max_rating"):
        if k in model_section and k not in model_init_args:
            model_cfg[k] = model_section[k]

    # build model
    try:
        model = model_mod.build_model_from_cfg(model_cfg)
    except Exception as e:
        # last resort: try MFModel/NCFModel defaults directly
        logger.exception("build_model_from_cfg failed: %s; attempting direct class instantiation", e)
        if mtype.lower().startswith("ncf"):
            model = model_mod.NCFModel(**model_init_args)
        else:
            model = model_mod.MFModel(**model_init_args)

    # load weights (if not loaded above)
    if state is None:
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
        else:
            logger.warning("No state_dict found at %s; returning uninitialized model", state_path)
            model.to(map_location)
            return model, cfg_loaded

    # If the saved state is actually for a subclass (keys like user_emb.*) and model is compatible, load
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # try to be a bit more permissive: drop unexpected keys or missing keys handling
        logger.warning("Strict load_state_dict failed: %s. Trying non-strict load.", e)
        model.load_state_dict(state, strict=False)

    model.to(map_location)
    model.eval()
    logger.info("Model loaded and moved to %s", map_location)
    return model, cfg_loaded


def recommend_topk_for_users(
    model: model_mod.BaseRecModel,
    users: List[int],
    n_items: int,
    top_k: int = 10,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
):
    """
    For each user in users, score all n_items and produce top_k item indices.
    Returns list of arrays (each length top_k).
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    all_items = np.arange(n_items, dtype=np.int64)
    recommended_list = []
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for u in users:
            # score in chunks to limit memory
            scores_chunks = []
            for start in range(0, n_items, batch_size):
                end = min(n_items, start + batch_size)
                items_chunk = all_items[start:end]
                users_chunk = np.full(
                    shape=(len(items_chunk),), fill_value=int(u), dtype=np.int64
                )
                users_t = torch.from_numpy(users_chunk).long().to(device)
                items_t = torch.from_numpy(items_chunk).long().to(device)
                out = model(users_t, items_t)  # expected (batch,)
                scores_chunks.append(out.cpu().numpy())
            scores = np.concatenate(scores_chunks, axis=0)  # (n_items,)
            # get top-k indices
            if top_k >= n_items:
                topk_idx = np.argsort(-scores)[:top_k]
            else:
                topk_idx = np.argpartition(-scores, top_k - 1)[:top_k]
                topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
            recommended_list.append(topk_idx.astype(np.int64))
    return recommended_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default="artifacts/model",
        help="Path to model dir with pytorch_model.bin and config.json",
    )
    parser.add_argument(
        "--input-path", type=str, required=True, help="CSV input with column 'user_idx'"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="CSV output path"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K items to return")
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Total number of items; if not provided, read from model config",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Scoring batch size for items"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger.info(f"Loading model from {args.model_dir}")
    model, _ = load_model_from_dir(args.model_dir, map_location="cpu")

    # try to get n_items from model attributes
    if args.n_items is None:
        try:
            n_items = int(getattr(model, "n_items"))
        except Exception:
            # try config
            cfg_path = os.path.join(args.model_dir, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            n_items = int(cfg.get("model_init_args", {}).get("n_items"))
    else:
        n_items = args.n_items

    if n_items is None:
        raise ValueError(
            "n_items must be provided either via --n-items or saved in model config."
        )

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(args.input_path)
    df = pd.read_csv(args.input_path)
    if "user_idx" not in df.columns:
        raise ValueError(
            "Input CSV must contain column 'user_idx' with integer user indices (0-based)."
        )

    users = df["user_idx"].astype(int).tolist()
    logger.info(
        f"Computing top-{args.top_k} for {len(users)} users over {n_items} items (batch_size={args.batch_size})"
    )
    recs = recommend_topk_for_users(
        model,
        users,
        n_items=n_items,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=torch.device("cpu"),
    )

    # write output
    out_rows = []
    for u, arr in zip(users, recs):
        out_rows.append(
            {"user_idx": int(u), "recommended_items": ",".join(map(str, arr.tolist()))}
        )
    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    logger.info(f"Wrote recommendations to {args.output_path}")


if __name__ == "__main__":
    main()
