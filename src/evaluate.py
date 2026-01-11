from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess

import torch

from src import dataset as dataset_mod
from src import engine as engine_mod
from src import model as model_mod
from src import utils

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained recsys model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="artifacts/model",
        help="Directory where model was saved (contains pytorch_model.bin and config.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for evaluation (e.g. cpu or cuda). If not set, uses config.train.device or cpu.",
    )
    parser.add_argument(
        "--no-full-ranking",
        action="store_true",
        help="Skip expensive full-ranking evaluation (precision@k/ndcg).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save metrics json. Default: <model-dir>/metrics_eval.json",
    )
    return parser.parse_args()


def load_model_from_dir(model_dir: str, device: torch.device):
    """
    Load model by reading config.json from model_dir (saved during save_pretrained).
    Build model from config via build_model_from_cfg() and load state_dict.
    """
    cfg_path = os.path.join(model_dir, "config.json")
    model_bin = os.path.join(model_dir, "pytorch_model.bin")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Model config not found at {cfg_path}")
    if not os.path.exists(model_bin):
        raise FileNotFoundError(f"Model weights not found at {model_bin}")

    # config.json may contain full training config (we saved whole cfg), so try to get 'model' section
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_disk = json.load(f)

    model_cfg = cfg_disk.get("model", {}) if isinstance(cfg_disk, dict) else {}
    # If config was saved as nested under "model", allow both possibilities
    # Ensure model_init_args exists (build_model_from_cfg expects model_cfg with model_init_args etc.)
    if "model_init_args" not in model_cfg:
        # maybe the whole config was saved; try to fill minimal model_cfg
        model_cfg.setdefault("model_init_args", {})
    # Build model instance (weights not loaded yet)
    logger.info(f"Building model from saved config keys: {list(model_cfg.keys())}")
    model = model_mod.build_model_from_cfg(model_cfg)
    # load weights
    map_loc = None if device.type == "cpu" else {"cuda:0": f"cuda:{device.index}" if device.index is not None else "cuda:0"}
    # safer: use map_location=device
    try:
        state = torch.load(model_bin, map_location=str(device))
    except Exception:
        # fallback to default load
        state = torch.load(model_bin, map_location="cpu")
    try:
        model.load_state_dict(state)
    except Exception:
        # if saved was entire model.state_dict under some nested key, attempt common fallbacks
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            raise

    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_dir} and moved to device {device}")
    return model, model_cfg


def main():
    args = parse_args()

    # load config and setup logging
    cfg = utils.load_yaml(args.config) if os.path.exists(args.config) else {}
    log_file = cfg.get("logging", {}).get("log_file", None)
    log_level = cfg.get("logging", {}).get("level", "INFO")
    utils.setup_basic_logger(level=log_level, log_file=log_file)
    logger.info(f"Using config: {args.config}")
    logger.info(f"Model dir: {args.model_dir}")

    # seed
    seed = int(cfg.get("seed", 42))
    utils.set_seed(seed)

    # device selection
    device_str = args.device or cfg.get("train", {}).get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # data paths
    data_cfg = cfg.get("data", {}) or {}
    ml100k_dir = data_cfg.get("ml100k_dir", "data/ml-100k")
    processed_dir = data_cfg.get("processed_dir") or os.path.join(ml100k_dir, "processed")
    train_path = data_cfg.get("train_path") or os.path.join(processed_dir, "train.csv")
    val_path = data_cfg.get("val_path") or os.path.join(processed_dir, "val.csv")
    test_path = data_cfg.get("test_path") or os.path.join(processed_dir, "test.csv")

    # prepare dataloaders (let dataset.get_dataloaders infer n_users/n_items if needed)
    train_cfg = cfg.get("train", {}) or {}
    batch_size = int(train_cfg.get("batch_size", 1024))
    num_workers = int(train_cfg.get("num_workers", 0))
    model_type = str(train_cfg.get("model_type", cfg.get("model", {}).get("type", "mf")))
    logger.info(f"Preparing dataloaders with model_type={model_type}, batch_size={batch_size}, num_workers={num_workers}")

    # pass None for n_users/n_items to let dataset infer from files
    dataloaders = dataset_mod.get_dataloaders(
        train_path=train_path,
        val_path=val_path if os.path.exists(val_path) else None,
        test_path=test_path if os.path.exists(test_path) else None,
        model_type=model_type,
        n_users=None,
        n_items=None,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=False,
        seed=seed,
    )

    # choose dataloader to evaluate: prefer test, else val
    eval_loader = dataloaders.test or dataloaders.val
    if eval_loader is None:
        raise RuntimeError("No eval dataloader found (neither test nor val). Run preprocessing to create splits.")

    # load model from model_dir
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model, model_cfg = load_model_from_dir(model_dir, device=device)

    # evaluation parameters
    compute_full = (not args.no_full_ranking) and bool(cfg.get("metrics", {}).get("compute_full_ranking", False))
    topk = int(cfg.get("metrics", {}).get("eval_topk", 10))
    eval_full_batch = int(cfg.get("evaluation", {}).get("full_ranking_batch", 4096))
    n_items = dataloaders.n_items

    logger.info(f"Starting evaluation. compute_full_ranking={compute_full}, topk={topk}, n_items={n_items}")

    metrics = engine_mod.evaluate(
        model=model,
        dataloader=eval_loader,
        device=device,
        model_type=model_type,
        compute_full_ranking=compute_full,
        topk=topk,
        n_items=n_items,
        eval_full_batch_size=eval_full_batch,
    )

    # where to save metrics
    out_path = args.output or os.path.join("artifacts/metrics", "metrics_eval.json")
    utils.ensure_dir(os.path.dirname(out_path) or ".")
    utils.save_json(metrics, out_path)
    logger.info(f"Saved evaluation metrics to {out_path}")
    logger.info(f"Evaluation results: {json.dumps(metrics, indent=2)}")

    # Optionally log git SHA and dvc.lock presence
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        logger.info(f"Git SHA: {git_sha}")
    except Exception:
        pass

    dvc_lock = "dvc.lock"
    if os.path.exists(dvc_lock):
        logger.info("Found dvc.lock in repo root (pipeline hash information available).")
    else:
        logger.info("dvc.lock not found in repo root.")

    logger.info("Evaluation finished.")


if __name__ == "__main__":
    main()
