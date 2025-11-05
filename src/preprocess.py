import argparse
import json
import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str = None):
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), "INFO"), format=fmt)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)


def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_ratings(ratings_file: str, sep: str = None) -> pd.DataFrame:
    """
    Load MovieLens ratings file (u.data). Supports files with no header (user item rating timestamp)
    or files with header.

    Returns DataFrame with columns: ['user_id', 'item_id', 'rating', 'timestamp']
    """
    if sep is None:
        # try to auto-detect: both '\t' and '\s+' possibilities
        try_seps = ["\t", "\s+"]
    else:
        try_seps = [sep]

    last_exc = None
    for s in try_seps:
        try:
            df = pd.read_csv(
                ratings_file,
                sep=s,
                engine="python",
                header=None,
                comment="#",
                names=["user_id", "item_id", "rating", "timestamp"],
            )
            # check shape
            if df.shape[1] >= 4:
                return df[["user_id", "item_id", "rating", "timestamp"]]
        except Exception as e:
            last_exc = e
            continue

    raise ValueError(
        f"Could not read ratings file {ratings_file}. Last error: {last_exc}"
    )


def validate_ratings_df(df: pd.DataFrame):
    required = {"user_id", "item_id", "rating", "timestamp"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Ratings dataframe missing required columns: {required - set(df.columns)}"
        )

    if df[["user_id", "item_id", "rating", "timestamp"]].isnull().any().any():
        raise ValueError("Ratings dataframe contains NaN values in required columns")

    if not np.issubdtype(df["rating"].dtype, np.number):
        raise ValueError("Rating column must be numeric")

    # common MovieLens ratings are in [1,5]
    rmin, rmax = df["rating"].min(), df["rating"].max()
    logger.info(f"Rating range in file: min={rmin}, max={rmax}")


def filter_by_interactions(
    df: pd.DataFrame,
    min_user_interactions: int = 1,
    min_item_interactions: int = 1,
) -> pd.DataFrame:
    logger.info(
        f"Filtering users with < {min_user_interactions} interactions and items with < {min_item_interactions} interactions"
    )
    # filter users
    user_counts = df["user_id"].value_counts()
    users_keep = user_counts[user_counts >= min_user_interactions].index
    df = df[df["user_id"].isin(users_keep)].copy()

    # filter items
    item_counts = df["item_id"].value_counts()
    items_keep = item_counts[item_counts >= min_item_interactions].index
    df = df[df["item_id"].isin(items_keep)].copy()

    df = df.reset_index(drop=True)
    logger.info(
        f"After filtering: interactions={len(df)}, users={df['user_id'].nunique()}, items={df['item_id'].nunique()}"
    )
    return df


def build_id_mappings(
    df: pd.DataFrame,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    users = sorted(df["user_id"].unique().tolist())
    items = sorted(df["item_id"].unique().tolist())
    user2idx = {int(u): int(i) for i, u in enumerate(users)}
    item2idx = {int(iid): int(i) for i, iid in enumerate(items)}
    return user2idx, item2idx


def apply_mappings(
    df: pd.DataFrame, user2idx: Dict[int, int], item2idx: Dict[int, int]
) -> pd.DataFrame:
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["item_idx"] = df["item_id"].map(item2idx)

    if df["user_idx"].isnull().any() or df["item_idx"].isnull().any():
        raise ValueError(
            "Some user_id/item_id could not be mapped to index after building mappings."
        )
    df["user_idx"] = df["user_idx"].astype(int)
    df["item_idx"] = df["item_idx"].astype(int)
    return df


def leave_one_out_split(
    df: pd.DataFrame, min_for_val: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each user, sort interactions by timestamp ascending.
    - If user has >= 3 interactions: last -> test, second last -> val, rest -> train
    - If user has 2 interactions: last -> test, first(s) -> train, val empty
    - If user has 1 interaction: train only (or can be dropped depending on filtering)
    """
    logger.info("Performing leave-one-out split per user (time-based)")
    df_sorted = df.sort_values(by=["user_idx", "timestamp"]).copy()
    train_rows = []
    val_rows = []
    test_rows = []

    for user, user_df in df_sorted.groupby("user_idx"):
        interactions = user_df.to_dict("records")
        n = len(interactions)
        if n >= 3:
            # everything except last two -> train
            for rec in interactions[:-2]:
                train_rows.append(rec)
            val_rows.append(interactions[-2])
            test_rows.append(interactions[-1])
        elif n == 2:
            train_rows.append(interactions[0])
            test_rows.append(interactions[1])
        elif n == 1:
            # put into train (alternatively could drop)
            train_rows.append(interactions[0])
        else:
            # no interactions? shouldn't happen
            continue

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows) if val_rows else pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame(columns=df.columns)

    # ensure columns
    for col in df.columns:
        if col not in train_df.columns:
            train_df[col] = pd.Series(dtype=df[col].dtype)
        if col not in val_df.columns:
            val_df[col] = pd.Series(dtype=df[col].dtype)
        if col not in test_df.columns:
            test_df[col] = pd.Series(dtype=df[col].dtype)

    logger.info(
        f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_processed_splits(
    out_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")
    # save with columns: user_idx,item_idx,rating,timestamp, (optionally original ids)
    save_cols = [
        "user_idx",
        "item_idx",
        "rating",
        "timestamp",
        "user_id",
        "item_id",
    ]

    # ensure columns order exists
    def _safe_save(df, path):
        if df is None or len(df) == 0:
            # create empty with headers
            pd.DataFrame(columns=save_cols).to_csv(path, index=False)
            return
        cols = [c for c in save_cols if c in df.columns]
        df.to_csv(path, columns=cols, index=False)

    _safe_save(train_df, train_path)
    _safe_save(val_df, val_path)
    _safe_save(test_df, test_path)
    logger.info(f"Saved processed splits to {out_dir}: train.csv / val.csv / test.csv")


def preprocess_from_config(cfg: dict, write_config_back: bool = False) -> dict:
    """
    Main preprocessing entrypoint. Returns dictionary with metadata:
      {
        'n_users': int,
        'n_items': int,
        'user2idx_path': str,
        'item2idx_path': str,
        'processed_dir': str,
        'train_path': str,
        'val_path': str,
        'test_path': str
      }
    """
    data_cfg = cfg.get("data", {})
    ratings_file = data_cfg.get("ratings_file")
    sep = data_cfg.get("sep", None)
    min_user_interactions = int(data_cfg.get("min_user_interactions", 1))
    min_item_interactions = int(data_cfg.get("min_item_interactions", 1))
    ml100k_dir = data_cfg.get(
        "ml100k_dir",
        os.path.dirname(ratings_file) if ratings_file else "data/ml-100k",
    )
    processed_dir = os.path.join(ml100k_dir, "processed")

    if ratings_file is None:
        raise ValueError("ratings_file must be set in config under data.ratings_file")

    # load
    logger.info(f"Loading ratings from {ratings_file}")
    ratings = load_ratings(ratings_file, sep=sep)
    validate_ratings_df(ratings)

    # filter by interactions (optional)
    ratings = filter_by_interactions(
        ratings, min_user_interactions, min_item_interactions
    )

    # build mappings
    user2idx, item2idx = build_id_mappings(ratings)
    logger.info(f"Built id mappings: n_users={len(user2idx)}, n_items={len(item2idx)}")

    # apply mappings
    ratings_mapped = apply_mappings(ratings, user2idx, item2idx)

    # leave-one-out split
    train_df, val_df, test_df = leave_one_out_split(ratings_mapped)

    # save artifacts
    os.makedirs(processed_dir, exist_ok=True)
    user2idx_path = os.path.join(processed_dir, "user2idx.json")
    item2idx_path = os.path.join(processed_dir, "item2idx.json")
    save_json(user2idx, user2idx_path)
    save_json(item2idx, item2idx_path)
    save_processed_splits(processed_dir, train_df, val_df, test_df)

    meta = {
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "user2idx_path": user2idx_path,
        "item2idx_path": item2idx_path,
        "processed_dir": processed_dir,
        "train_path": os.path.join(processed_dir, "train.csv"),
        "val_path": os.path.join(processed_dir, "val.csv"),
        "test_path": os.path.join(processed_dir, "test.csv"),
    }

    # optionally write back sizes into config file (model.model_init_args)
    if write_config_back:
        cfg_path = getattr(preprocess_from_config, "_last_cfg_path", None)
        if cfg_path is None:
            logger.warning(
                "write_config_back requested but original config path not set; skipping writing to config."
            )
        else:
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_disk = yaml.safe_load(f)
                # ensure nested keys exist
                cfg_disk.setdefault("model", {}).setdefault("model_init_args", {})
                cfg_disk["model"]["model_init_args"]["n_users"] = meta["n_users"]
                cfg_disk["model"]["model_init_args"]["n_items"] = meta["n_items"]
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg_disk, f, sort_keys=False)
                logger.info(f"Wrote n_users/n_items into config at {cfg_path}")
            except Exception as e:
                logger.exception(f"Failed to write back to config: {e}")

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens-100k for recsys project"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="If set, write detected n_users/n_items back into config.model.model_init_args",
    )
    args = parser.parse_args()

    cfg = read_config(args.config)
    log_file = cfg.get("logging", {}).get("log_file")
    log_level = cfg.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level, log_file=log_file)

    # store config path so preprocess function can use it when writing back
    preprocess_from_config._last_cfg_path = args.config

    logger.info(f"Starting preprocessing with config: {args.config}")
    meta = preprocess_from_config(cfg, write_config_back=args.write_config)
    logger.info("Preprocessing finished. Metadata:")
    logger.info(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
