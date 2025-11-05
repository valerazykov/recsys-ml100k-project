from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def read_interaction_csv(path: str) -> pd.DataFrame:
    """
    Loads a CSV produced by preprocess.py with at least columns:
      - user_idx
      - item_idx
      - rating (optional)
      - timestamp (optional)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Interaction file not found: {path}")
    df = pd.read_csv(path)
    required = {"user_idx", "item_idx"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Interaction CSV {path} must contain columns {required}. Found: {df.columns.tolist()}"
        )
    return df


class RatingDataset(Dataset):
    """
    Pointwise dataset for rating prediction / regression.
    Returns: tuple (user_idx: LongTensor, item_idx: LongTensor, rating: FloatTensor)
    """

    def __init__(self, interactions: Union[str, pd.DataFrame]):
        """
        interactions: path to csv OR pandas.DataFrame with columns ['user_idx','item_idx','rating',...]
        """
        if isinstance(interactions, str):
            interactions = read_interaction_csv(interactions)
        self.df = interactions.reset_index(drop=True)
        if "rating" not in self.df.columns:
            raise ValueError(
                "RatingDataset expects 'rating' column in interactions"
            )
        # ensure integer indices
        self.users = self.df["user_idx"].astype(int).to_numpy()
        self.items = self.df["item_idx"].astype(int).to_numpy()
        self.ratings = self.df["rating"].astype(float).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        u = int(self.users[idx])
        i = int(self.items[idx])
        r = float(self.ratings[idx])
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
        )


class BPRDataset(Dataset):
    """
    Pairwise dataset for BPR-style training.
    Returns: (user, pos_item, neg_item) as LongTensors.

    Negative sampling is done on-the-fly uniformly from all items that are NOT in the user's positive set.
    """

    def __init__(
        self,
        interactions: Union[str, pd.DataFrame],
        num_items: int,
        rng: Optional[np.random.RandomState] = None,
    ):
        """
        interactions: path to csv OR pandas.DataFrame with columns ['user_idx', 'item_idx', ...]
        num_items: total number of distinct items (indexed 0..num_items-1)
        rng: optional numpy RandomState for deterministic negative sampling; if None, uses np.random
        """
        if isinstance(interactions, str):
            interactions = read_interaction_csv(interactions)
        self.df = interactions.reset_index(drop=True)
        self.num_items = int(num_items)
        self.users = self.df["user_idx"].astype(int).to_numpy()
        self.pos_items = self.df["item_idx"].astype(int).to_numpy()
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng

        # build mapping user -> set(pos_items)
        self.user_pos = build_user_pos_dict(self.df)
        self.user_list = sorted(self.user_pos.keys())
        # precompute list of users repeated by number of interactions for sampling if needed
        self.user_interactions = [
            (u, np.array(sorted(list(self.user_pos[u])), dtype=np.int64))
            for u in self.user_list
        ]

        # for __len__ we return number of positive interactions (len of df)
        self.length = len(self.df)

    def __len__(self) -> int:
        return self.length

    def sample_negative(self, user: int) -> int:
        """
        Uniformly sample negative item id for given user (i.e., item not in user's positive set).
        May loop for a few attempts; if user has interacted with almost all items, falls back to random choice.
        """
        user_pos_set = self.user_pos.get(user, set())
        # quick path: if user_pos_set small compared to all items, sample until not in set
        max_tries = 50
        for _ in range(max_tries):
            neg = int(self.rng.randint(0, self.num_items))
            if neg not in user_pos_set:
                return neg
        # fallback: build candidates
        all_items = set(range(self.num_items))
        candidates = np.array(list(all_items - user_pos_set), dtype=np.int64)
        if len(candidates) == 0:
            # fallback to random in full space
            return int(self.rng.randint(0, self.num_items))
        return int(self.rng.choice(candidates))

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        idx indexes into the original interactions dataframe; returns (user, pos_item, neg_item)
        """
        u = int(self.users[idx])
        pos = int(self.pos_items[idx])
        neg = int(self.sample_negative(u))
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def build_user_pos_dict(df: pd.DataFrame) -> Dict[int, set]:
    """
    Build mapping user_idx -> set(item_idx) from interactions df.
    """
    user_pos: Dict[int, set] = {}
    for row in df.itertuples(index=False):
        u = int(getattr(row, "user_idx"))
        i = int(getattr(row, "item_idx"))
        user_pos.setdefault(u, set()).add(i)
    return user_pos


def _collate_rating_batch(
    batch: List[Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]]
):
    users = torch.stack([b[0] for b in batch])
    items = torch.stack([b[1] for b in batch])
    ratings = torch.stack([b[2] for b in batch])
    return users, items, ratings


def _collate_bpr_batch(
    batch: List[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]
):
    users = torch.stack([b[0] for b in batch])
    pos = torch.stack([b[1] for b in batch])
    neg = torch.stack([b[2] for b in batch])
    return users, pos, neg


@dataclass
class DataloaderBundle:
    train: DataLoader
    val: Optional[DataLoader]
    test: Optional[DataLoader]
    n_users: int
    n_items: int
    train_len: int
    val_len: int
    test_len: int


def get_dataloaders(
    train_path: str,
    val_path: Optional[str],
    test_path: Optional[str],
    model_type: str,
    n_users: int,
    n_items: int,
    batch_size: int = 1024,
    num_workers: int = 0,
    shuffle_train: bool = True,
    seed: int = 42,
) -> DataloaderBundle:
    """
    Create DataLoaders for train/val/test.

    model_type:
      - "mf" or "ncf" -> uses RatingDataset (user,item,rating)
      - "bpr" or "pairwise" -> uses BPRDataset (user,pos,neg) for train, and RatingDataset for val/test (if available)

    Returns DataloaderBundle with dataloaders and metadata.
    """
    rng = np.random.RandomState(seed)
    # load files into dfs (even if empty)
    train_df = read_interaction_csv(train_path)
    val_df = (
        read_interaction_csv(val_path)
        if (val_path and os.path.exists(val_path))
        else pd.DataFrame(columns=train_df.columns)
    )
    test_df = (
        read_interaction_csv(test_path)
        if (test_path and os.path.exists(test_path))
        else pd.DataFrame(columns=train_df.columns)
    )

    # basic checks
    logger.info(
        f"Loaded interactions: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )
    # determine n_users/n_items if not provided? function arguments require them; but allow fallback
    if n_users is None or n_items is None:
        # attempt to infer
        n_users_in_data = int(
            max(
                train_df["user_idx"].max() if len(train_df) > 0 else -1,
                val_df["user_idx"].max() if len(val_df) > 0 else -1,
                test_df["user_idx"].max() if len(test_df) > 0 else -1,
            )
            + 1
        )
        n_items_in_data = int(
            max(
                train_df["item_idx"].max() if len(train_df) > 0 else -1,
                val_df["item_idx"].max() if len(val_df) > 0 else -1,
                test_df["item_idx"].max() if len(test_df) > 0 else -1,
            )
            + 1
        )
        logger.warning(
            f"n_users/n_items not provided. Inferred n_users={n_users_in_data}, n_items={n_items_in_data}"
        )
        n_users = n_users_in_data
        n_items = n_items_in_data

    if model_type.lower() in ("mf", "ncf", "rating", "pointwise"):
        train_ds = RatingDataset(train_df)
        val_ds = RatingDataset(val_df) if len(val_df) > 0 else None
        test_ds = RatingDataset(test_df) if len(test_df) > 0 else None

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            collate_fn=_collate_rating_batch,
            pin_memory=True,
        )
        val_loader = (
            DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_collate_rating_batch,
            )
            if val_ds is not None
            else None
        )
        test_loader = (
            DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_collate_rating_batch,
            )
            if test_ds is not None
            else None
        )

    elif model_type.lower() in ("bpr", "pairwise"):
        # For pairwise training use BPRDataset for train (with negatives) and rating dataset for val/test
        train_ds = BPRDataset(train_df, num_items=n_items, rng=rng)
        val_ds = RatingDataset(val_df) if len(val_df) > 0 else None
        test_ds = RatingDataset(test_df) if len(test_df) > 0 else None

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            collate_fn=_collate_bpr_batch,
            pin_memory=True,
        )
        val_loader = (
            DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_collate_rating_batch,
            )
            if val_ds is not None
            else None
        )
        test_loader = (
            DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_collate_rating_batch,
            )
            if test_ds is not None
            else None
        )
    else:
        raise ValueError(
            f"Unknown model_type {model_type}. Choose from 'mf', 'ncf', 'bpr'."
        )

    bundle = DataloaderBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        n_users=int(n_users),
        n_items=int(n_items),
        train_len=len(train_df),
        val_len=len(val_df),
        test_len=len(test_df),
    )
    return bundle


# If run as script, demonstrate a small smoke-test (does not run heavy operations)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=str, default="data/ml-100k/processed/train.csv"
    )
    parser.add_argument(
        "--val", type=str, default="data/ml-100k/processed/val.csv"
    )
    parser.add_argument(
        "--test", type=str, default="data/ml-100k/processed/test.csv"
    )
    parser.add_argument("--model-type", type=str, default="mf")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    bundle = get_dataloaders(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        model_type=args.model_type,
        n_users=None,
        n_items=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    logger.info(
        f"Train batches: approx {len(bundle.train)} (len train_df={bundle.train_len})"
    )
    # iterate a single batch to check shapes
    for batch in bundle.train:
        if args.model_type.lower() in ("mf", "ncf"):
            users, items, ratings = batch
            logger.info(
                f"Sample batch shapes: users={users.shape}, items={items.shape}, ratings={ratings.shape}"
            )
        else:
            users, pos, neg = batch
            logger.info(
                f"Sample batch shapes: users={users.shape}, pos={pos.shape}, neg={neg.shape}"
            )
        break
