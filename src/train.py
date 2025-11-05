from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

import torch
from torch import nn

from src import engine
from src import model as model_mod
from src import dataset as dataset_mod
from src import preprocess as preprocess_mod
from src import utils

logger = utils.logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train script for recsys project")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to YAML config")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing before training")
    parser.add_argument("--write-config", action="store_true", help="If set, write detected n_users/n_items back into config after preprocessing")
    parser.add_argument("--resume", type=str, default=None, help="Path to model dir to resume from (load pretrained weights)")
    parser.add_argument("--no-save", action="store_true", help="Do not save final model (useful for quick tests)")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="Override config values, format key=value (dot notation for nested). Example: train.epochs=5 train.lr=0.0005")
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[list]):
    if not overrides:
        return cfg
    # Very simple parser for key=value pairs; supports dot notation for nested keys.
    for kv in overrides:
        if "=" not in kv:
            logger.warning(f"Ignoring invalid override '{kv}', expected key=value.")
            continue
        k, v = kv.split("=", 1)
        # Try to interpret v as int/float/bool/json, fallback to string
        parsed_v: Any = None
        for cast in (int, float):
            try:
                parsed_v = cast(v)
                break
            except Exception:
                parsed_v = None
        if parsed_v is None:
            if v.lower() in ("true", "false"):
                parsed_v = v.lower() == "true"
            else:
                # try JSON parse for lists/dicts
                try:
                    parsed_v = json.loads(v)
                except Exception:
                    parsed_v = v
        # set nested
        parts = k.split(".")
        cur = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = parsed_v
        logger.info(f"Applied override: {k} = {parsed_v}")
    return cfg


def _to_float_or_none(x):
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    # handle strings like "1e-8" or "0.001"
    try:
        return float(x)
    except Exception:
        return None


def _to_betas_tuple(x):
    # Accept list/tuple or string like "[0.9,0.999]" or "0.9,0.999"
    if x is None:
        return (0.9, 0.999)
    if isinstance(x, (list, tuple)):
        try:
            return tuple(float(v) for v in x)
        except Exception:
            return (0.9, 0.999)
    if isinstance(x, str):
        try:
            # try JSON-like
            import json
            parsed = json.loads(x)
            if isinstance(parsed, (list, tuple)):
                return tuple(float(v) for v in parsed)
        except Exception:
            pass
        try:
            parts = [p.strip() for p in x.split(",")]
            return tuple(float(p) for p in parts if p != "")
        except Exception:
            return (0.9, 0.999)
    # fallback
    return (0.9, 0.999)


def build_optimizer(model: torch.nn.Module, optim_cfg: Dict[str, Any]):
    name = optim_cfg.get("name", "adam").lower()
    params = dict(optim_cfg.get("params", {}) or {})
    # pull lr either from params or top-level train.lr
    lr = params.pop("lr", None)
    if lr is None:
        lr = optim_cfg.get("lr", None)
    lr_val = _to_float_or_none(lr) or 1e-3

    # sanitize commonly used params
    betas_raw = params.pop("betas", params.pop("beta", None))
    betas = _to_betas_tuple(betas_raw)

    eps_raw = params.pop("eps", params.pop("epsilon", None))
    eps = _to_float_or_none(eps_raw)
    if eps is None:
        eps = 1e-8

    weight_decay_raw = params.pop("weight_decay", None)
    weight_decay = _to_float_or_none(weight_decay_raw) or 0.0

    # Remove any unknown non-basic params to avoid passing strings etc.
    # We'll pass only those keyword args that torch optimizers accept below.
    if name in ("adam", "adamw"):
        if name == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr_val, betas=betas, eps=eps, weight_decay=weight_decay)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=lr_val, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name in ("sgd",):
        momentum = _to_float_or_none(params.pop("momentum", 0.9)) or 0.9
        opt = torch.optim.SGD(model.parameters(), lr=lr_val, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {name}")

    logger.info(f"Built optimizer {name} lr={lr_val} betas={betas} eps={eps} weight_decay={weight_decay}")
    return opt


def build_scheduler(optimizer, sched_cfg: dict):
    name = sched_cfg.get("name", "step").lower()
    params = dict(sched_cfg.get("params", {}) or {})

    # безопасно приводим параметры к float/int
    factor = float(params.get("factor", 0.1))
    min_lr = float(params.get("min_lr", 0.0))
    patience = int(params.get("patience", 10))
    step_size = int(params.get("step_size", 10))

    if name in ("reducelronplateau", "plateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=factor, min_lr=min_lr, patience=patience
        )
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=factor
        )
    else:
        raise ValueError(f"Unsupported scheduler {name}")
    
    return scheduler


# === DEBUG PATCH: проверка данных и обучения ===
def debug_train_batch(dataloader, model, device, model_type):
    print("=== DEBUG: first batch check ===")
    model.train()
    batch = next(iter(dataloader))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # разделяем по типу модели
    if model_type.lower() in ("mf", "ncf", "rating", "pointwise"):
        users, items, ratings = batch
        # проверяем индексы
        print("users idx min/max:", users.min().item(), users.max().item())
        print("items idx min/max:", items.min().item(), items.max().item())
        
        # конвертация в float
        ratings = ratings.float()
        
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        preds = model(users, items)
        loss = nn.MSELoss()(preds, ratings)
        print("Initial preds:", preds[:5].detach().cpu().numpy())
        print("Initial ratings:", ratings[:5].detach().cpu().numpy())
        print("Initial loss:", loss.item())
        
        loss.backward()
        for name, param in model.named_parameters():
            grad_norm = param.grad.norm().item() if param.grad is not None else None
            print(f"Param: {name}, grad norm: {grad_norm}")
        
        optimizer.step()
        print("Step applied. Check if model params changed.")
        
    elif model_type.lower() in ("bpr", "pairwise"):
        users, pos, neg = batch
        users, pos, neg = users.to(device), pos.to(device), neg.to(device)
        pos_scores = model(users, pos)
        neg_scores = model(users, neg)
        diff = pos_scores - neg_scores
        loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))
        print("Initial BPR loss:", loss.item())
        loss.backward()
        for name, param in model.named_parameters():
            grad_norm = param.grad.norm().item() if param.grad is not None else None
            print(f"Param: {name}, grad norm: {grad_norm}")
        optimizer.step()
    print("=== DEBUG END ===\n")


def main():
    args = parse_args()
    cfg = utils.load_yaml(args.config)
    # apply overrides
    cfg = apply_overrides(cfg, args.override)

    # setup logging
    log_file = cfg.get("logging", {}).get("log_file")
    log_level = cfg.get("logging", {}).get("level", "INFO")
    utils.setup_basic_logger(level=log_level, log_file=log_file)
    logger.info(f"Loaded config from {args.config}")

    # set seed
    seed = int(cfg.get("seed", 42))
    utils.set_seed(seed)

    # preprocess if requested
    meta = {}
    if args.preprocess:
        logger.info("Running preprocessing step...")
        meta = preprocess_mod.preprocess_from_config(cfg, write_config_back=args.write_config)
        logger.info(f"Preprocessing done. Meta: {meta}")
        # if preprocess wrote into config on disk, we still have in-memory cfg updated? not necessarily, so update cfg with detected sizes
        if meta:
            cfg.setdefault("model", {}).setdefault("model_init_args", {})
            cfg["model"]["model_init_args"]["n_users"] = meta["n_users"]
            cfg["model"]["model_init_args"]["n_items"] = meta["n_items"]
            # also store processed paths
            cfg["data"]["processed_dir"] = meta.get("processed_dir")
            cfg["data"]["train_path"] = meta.get("train_path")
            cfg["data"]["val_path"] = meta.get("val_path")
            cfg["data"]["test_path"] = meta.get("test_path")

    # fallback: if processed files exist and meta empty, try to infer
    processed_dir = cfg.get("data", {}).get("ml100k_dir")
    processed_subdir = os.path.join(processed_dir or "data/ml-100k", "processed")
    train_path = cfg.get("data", {}).get("train_path") or os.path.join(processed_subdir, "train.csv")
    val_path = cfg.get("data", {}).get("val_path") or os.path.join(processed_subdir, "val.csv")
    test_path = cfg.get("data", {}).get("test_path") or os.path.join(processed_subdir, "test.csv")

    # determine n_users/n_items from config or from processed files
    model_init_args = cfg.get("model", {}).get("model_init_args", {}) or {}
    n_users = model_init_args.get("n_users")
    n_items = model_init_args.get("n_items")
    if (n_users is None) or (n_items is None):
        # try to infer from processed files if they exist
        if os.path.exists(train_path):
            import pandas as pd
            df_all = pd.concat([pd.read_csv(p) for p in [train_path, val_path, test_path] if os.path.exists(p)], ignore_index=True, sort=False)
            if "user_idx" in df_all.columns and "item_idx" in df_all.columns:
                inferred_n_users = int(df_all["user_idx"].max() + 1)
                inferred_n_items = int(df_all["item_idx"].max() + 1)
                n_users = n_users or inferred_n_users
                n_items = n_items or inferred_n_items
                cfg.setdefault("model", {}).setdefault("model_init_args", {})
                cfg["model"]["model_init_args"]["n_users"] = n_users
                cfg["model"]["model_init_args"]["n_items"] = n_items
                logger.info(f"Inferred n_users={n_users}, n_items={n_items} from processed files.")
        else:
            logger.warning("Processed train/test files not found and model_init_args.n_users/n_items not set. Preprocess first or fill these in config.")

    # create dataloaders
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 1024))
    num_workers = int(train_cfg.get("num_workers", 0))
    model_type = str(train_cfg.get("model_type", cfg.get("model", {}).get("type", "mf")))

    logger.info(f"Preparing dataloaders with model_type={model_type}, batch_size={batch_size}, num_workers={num_workers}")
    dataloaders = dataset_mod.get_dataloaders(
        train_path=train_path,
        val_path=val_path if os.path.exists(val_path) else None,
        test_path=test_path if os.path.exists(test_path) else None,
        model_type=model_type,
        n_users=n_users,
        n_items=n_items,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=True,
        seed=seed,
    )

    # build model
    model_cfg = cfg.get("model", {}) or {}
    # ensure model_init_args has n_users/n_items
    model_init_args = model_cfg.get("model_init_args", {})
    if model_init_args.get("n_users") is None:
        model_init_args["n_users"] = dataloaders.n_users
    if model_init_args.get("n_items") is None:
        model_init_args["n_items"] = dataloaders.n_items
    model_cfg["model_init_args"] = model_init_args

    logger.info(f"Building model from config. model_cfg keys: {list(model_cfg.keys())}")
    model = model_mod.build_model_from_cfg(model_cfg)
    # device handling
    device_str = train_cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested in config but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    model.to(device)
    logger.info(f"Model built and moved to device {device}")

    # optionally resume weights
    if args.resume:
        resume_dir = args.resume
        try:
            model, loaded_cfg = model_mod.MFModel.from_pretrained  # dummy to avoid lint
        except Exception:
            pass
        # we use BaseRecModel.from_pretrained on appropriate class - try to detect by cfg
        try:
            # attempt to use model class's from_pretrained
            model_loaded, _ = model.__class__.from_pretrained(args.resume, map_location=device)
            model.load_state_dict(model_loaded.state_dict())
            logger.info(f"Loaded pretrained weights from {args.resume}")
        except Exception as e:
            logger.exception(f"Failed to load pretrained weights from {args.resume}: {e}")

    # optimizer & scheduler
    optim_cfg = cfg.get("optimizer", {"name": "adam", "params": {"lr": train_cfg.get("lr", 1e-3)}})
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}))

    # training loop
    epochs = int(train_cfg.get("epochs", 10))
    use_amp = bool(train_cfg.get("use_amp", False))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 5))
    save_dir = cfg.get("save", {}).get("save_dir", "artifacts/model")
    ensure_dir = utils.ensure_dir
    ensure_dir(save_dir)

    metrics_log = {"train": [], "val": []}
    best_val_rmse = float("inf")
    best_epoch = -1

    logger.info(f"Starting training for {epochs} epochs. use_amp={use_amp}")

    # вызови перед первым epoch
    debug_train_batch(dataloaders.train, model, device, model_type)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_res = engine.train_epoch(
            model=model,
            dataloader=dataloaders.train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            model_type=model_type,
            loss_fn=None,
            use_amp=use_amp,
            max_grad_norm=float(train_cfg.get("max_grad_norm")) if train_cfg.get("max_grad_norm") is not None else None,
        )
        t1 = time.time()
        logger.info(f"Epoch {epoch} train loss: {train_res['loss']:.6f}, examples={train_res['examples']}, time={(t1-t0):.1f}s")

        # validation
        if dataloaders.val is not None:
            eval_cfg = cfg.get("metrics", {})
            topk = int(eval_cfg.get("eval_topk", 10))
            compute_full = bool(cfg.get("metrics", {}).get("compute_full_ranking", False))
            eval_batch = int(cfg.get("evaluation", {}).get("eval_batch_size", 1024))
            eval_full_batch = int(cfg.get("evaluation", {}).get("full_ranking_batch", 4096))

            val_metrics = engine.evaluate(
                model=model,
                dataloader=dataloaders.val,
                device=device,
                model_type=model_type,
                compute_full_ranking=compute_full,
                topk=topk,
                n_items=dataloaders.n_items,
                eval_full_batch_size=eval_full_batch,
            )
            logger.info(f"Epoch {epoch} val metrics: {json.dumps(val_metrics)}")
            metrics_log["val"].append({"epoch": epoch, **val_metrics})
            # scheduler step for ReduceLROnPlateau expects metric
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # use validation loss or rmse
                    monitor = val_metrics.get("rmse", val_metrics.get("loss", None))
                    if monitor is not None:
                        scheduler.step(monitor)
                else:
                    scheduler.step()

            # track best
            val_rmse = val_metrics.get("rmse", None)
            if val_rmse is not None and val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                # save best model
                if not args.no_save:
                    logger.info(f"New best val RMSE {best_val_rmse:.6f} at epoch {epoch}. Saving best model.")
                    engine.save_checkpoint(model, optimizer, epoch, save_dir, config=cfg)

        # optionally checkpoint every N epochs
        if (checkpoint_every > 0) and (epoch % checkpoint_every == 0) and (not args.no_save):
            logger.info(f"Saving checkpoint at epoch {epoch}")
            engine.save_checkpoint(model, optimizer, epoch, save_dir, config=cfg)

    # final evaluation on test (if available)
    if dataloaders.test is not None:
        test_metrics = engine.evaluate(
            model=model,
            dataloader=dataloaders.test,
            device=device,
            model_type=model_type,
            compute_full_ranking=bool(cfg.get("metrics", {}).get("compute_full_ranking", False)),
            topk=int(cfg.get("metrics", {}).get("eval_topk", 10)),
            n_items=dataloaders.n_items,
            eval_full_batch_size=int(cfg.get("evaluation", {}).get("full_ranking_batch", 4096)),
        )
        logger.info(f"Test metrics: {json.dumps(test_metrics)}")
        metrics_log["test"] = test_metrics

    # save final model & metrics
    if not args.no_save:
        logger.info("Saving final model and metrics...")
        try:
            engine.save_checkpoint(model, optimizer, epoch, save_dir, config=cfg)
        except Exception:
            logger.exception("Failed to save final model.")

        metrics_path = os.path.join(save_dir, "metrics.json")
        utils.save_json(metrics_log, metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")

        # save used config
        config_used_path = os.path.join(save_dir, "config_used.yaml")
        utils.save_yaml(cfg, config_used_path)
        logger.info(f"Saved used config to {config_used_path}")

    logger.info("Training finished.")
    if best_epoch > 0:
        logger.info(f"Best val RMSE {best_val_rmse:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
