import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import dataset as ds


def _write_csv(df: pd.DataFrame, path: str, index: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)


def make_simple_interactions():
    """
    Return a small DataFrame already in processed format:
    columns: user_idx, item_idx, rating, timestamp
    """
    rows = [
        {"user_idx": 0, "item_idx": 0, "rating": 5.0, "timestamp": 1},
        {"user_idx": 0, "item_idx": 1, "rating": 4.0, "timestamp": 2},
        {"user_idx": 1, "item_idx": 0, "rating": 3.0, "timestamp": 3},
        {"user_idx": 1, "item_idx": 2, "rating": 2.0, "timestamp": 4},
        {"user_idx": 2, "item_idx": 3, "rating": 4.0, "timestamp": 5},
    ]
    return pd.DataFrame(rows)


def test_ratingdataset_and_dataloader_shapes(tmp_path):
    df = make_simple_interactions()
    train_csv = str(tmp_path / "train.csv")
    _write_csv(df, train_csv)

    # Create dataloaders (mf mode). Let function infer n_users/n_items
    bundle = ds.get_dataloaders(
        train_path=train_csv,
        val_path=None,
        test_path=None,
        model_type="mf",
        n_users=None,
        n_items=None,
        batch_size=2,
        num_workers=0,
        shuffle_train=False,
        seed=42,
    )

    assert bundle.train_len == len(df)
    assert isinstance(bundle.train, DataLoader)
    # iterate one batch
    for batch in bundle.train:
        users, items, ratings = batch
        assert isinstance(users, torch.LongTensor) or users.dtype == torch.int64
        assert isinstance(items, torch.LongTensor) or items.dtype == torch.int64
        assert isinstance(ratings, torch.FloatTensor) or ratings.dtype == torch.float32
        assert users.ndim == 1 and items.ndim == 1 and ratings.ndim == 1
        assert users.shape[0] <= 2  # batch size
        break


def test_get_dataloaders_infers_n_users_and_items(tmp_path):
    # create df with user_idx up to 4 and item_idx values 0..4 -> expect n_users=5, n_items=5
    rows = []
    for u in range(5):
        rows.append({"user_idx": u, "item_idx": u % 8, "rating": 4.0, "timestamp": u})
    df = pd.DataFrame(rows)
    p = tmp_path / "train.csv"
    _write_csv(df, str(p))

    bundle = ds.get_dataloaders(
        train_path=str(p),
        val_path=None,
        test_path=None,
        model_type="mf",
        n_users=None,
        n_items=None,
        batch_size=4,
        num_workers=0,
        shuffle_train=False,
        seed=0,
    )
    assert bundle.n_users == 5
    # expected n_items equals (max(item_idx) + 1) -> here max is 4 -> n_items == 5
    assert bundle.n_items == int(df["item_idx"].max()) + 1


def test_bprdataset_negative_sampling_and_getitem(tmp_path):
    # prepare a case where a user has interacted with all items except one -> ensure sampling returns the remaining
    num_items = 6
    rows = []
    # user 0: interacted with items 0..4 (not 5)
    for it in range(num_items - 1):
        rows.append({"user_idx": 0, "item_idx": it, "rating": 5.0, "timestamp": it})
    # user 1: small pos set {0,2}
    rows.append({"user_idx": 1, "item_idx": 0, "rating": 4.0, "timestamp": 10})
    rows.append({"user_idx": 1, "item_idx": 2, "rating": 3.0, "timestamp": 11})
    df = pd.DataFrame(rows)

    # instantiate BPRDataset with deterministic RNG
    rng = np.random.RandomState(123)
    bpr = ds.BPRDataset(df, num_items=num_items, rng=rng)

    # sample many times for user 0 -> should always get 5 (the only negative)
    negs_user0 = {bpr.sample_negative(0) for _ in range(30)}
    assert negs_user0 == {num_items - 1}

    # for user 1, sample a negative and ensure it's NOT in pos set {0,2}
    neg_sample = bpr.sample_negative(1)
    assert neg_sample not in {0, 2}
    # check __getitem__ returns tuple tensors and neg != pos
    u_t, pos_t, neg_t = bpr[0]
    assert isinstance(u_t, torch.LongTensor)
    assert isinstance(pos_t, torch.LongTensor)
    assert isinstance(neg_t, torch.LongTensor)
    assert pos_t.item() != neg_t.item()


def test_get_dataloaders_pairwise_batches(tmp_path):
    # Build a small interactions df for pairwise training
    rows = []
    # users 0..3 interacting with items 0..5
    for u in range(4):
        for it in range(2):  # two interactions per user
            rows.append({"user_idx": u, "item_idx": (u * 2 + it) % 6, "rating": 5.0, "timestamp": u * 10 + it})
    df = pd.DataFrame(rows)
    train_path = str(tmp_path / "train_bpr.csv")
    _write_csv(df, train_path)

    bundle = ds.get_dataloaders(
        train_path=train_path,
        val_path=None,
        test_path=None,
        model_type="bpr",
        n_users=None,
        n_items=6,
        batch_size=3,
        num_workers=0,
        shuffle_train=False,
        seed=42,
    )

    # iterate one batch from train (users, pos, neg)
    for batch in bundle.train:
        users, pos, neg = batch
        assert users.ndim == 1 and pos.ndim == 1 and neg.ndim == 1
        assert users.shape[0] <= 3
        # all negatives must not be equal to corresponding positives for the batch
        for p, n in zip(pos.tolist(), neg.tolist()):
            assert p != n
        break


def test_read_interaction_csv_raises_on_missing_columns(tmp_path):
    # write CSV missing user_idx column
    bad_df = pd.DataFrame({"uid": [1, 2], "item_idx": [1, 2]})
    path = tmp_path / "bad.csv"
    bad_df.to_csv(path, index=False)
    try:
        ds.read_interaction_csv(str(path))
        # expected to raise, if not - fail
        assert False, "read_interaction_csv should have raised ValueError for missing required columns"
    except ValueError:
        pass
