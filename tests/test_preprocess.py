import os

import numpy as np
import pandas as pd
import pytest

from src import preprocess as pp


def test_filter_and_mappings_roundtrip(small_ratings_df):
    # filter users with at least 2 interactions, items at least 1
    filtered = pp.filter_by_interactions(
        small_ratings_df, min_user_interactions=2, min_item_interactions=1
    )
    # users kept: user 1 (2 interactions) and user 3 (3 interactions)
    assert set(filtered["user_id"].unique()) == {1, 3}
    # build mappings
    user2idx, item2idx = pp.build_id_mappings(filtered)
    assert min(user2idx.values()) == 0
    assert min(item2idx.values()) == 0
    # apply mappings -> add user_idx and item_idx
    mapped = pp.apply_mappings(filtered, user2idx, item2idx)
    assert "user_idx" in mapped.columns and "item_idx" in mapped.columns
    # indices are ints and 0-based
    assert mapped["user_idx"].dtype == int or mapped["user_idx"].dtype == np.int64
    assert mapped["user_idx"].min() == 0


def test_leave_one_out_split_various_counts():
    # Build a tiny mapped DataFrame already with user_idx/item_idx/timestamp
    rows = []
    # user 0 -> 1 interaction
    rows.append({"user_idx": 0, "item_idx": 0, "rating": 5.0, "timestamp": 1})
    # user 1 -> 2 interactions
    rows.append({"user_idx": 1, "item_idx": 1, "rating": 4.0, "timestamp": 1})
    rows.append({"user_idx": 1, "item_idx": 2, "rating": 3.0, "timestamp": 2})
    # user 2 -> 4 interactions
    rows.extend(
        [
            {"user_idx": 2, "item_idx": 3, "rating": 5.0, "timestamp": 1},
            {"user_idx": 2, "item_idx": 4, "rating": 4.0, "timestamp": 2},
            {"user_idx": 2, "item_idx": 5, "rating": 3.0, "timestamp": 3},
            {"user_idx": 2, "item_idx": 6, "rating": 2.0, "timestamp": 4},
        ]
    )
    df = pd.DataFrame(rows)
    train, val, test = pp.leave_one_out_split(df)
    # user 0: only train
    assert 0 in train["user_idx"].values
    assert 0 not in val["user_idx"].values and 0 not in test["user_idx"].values
    # user 1: train + test
    assert 1 in train["user_idx"].values and 1 in test["user_idx"].values
    # user 2: train includes first n-2 interactions, val second last, test last
    assert (
        2 in train["user_idx"].values
        and 2 in val["user_idx"].values
        and 2 in test["user_idx"].values
    )
    # check counts
    assert len(test) == 2  # users 1 and 2 -> last interactions
    assert len(val) == 1  # only user 2 had a val


def test_load_ratings_auto_detects_sep(tmp_ratings_file):
    # load_ratings should accept tab-separated file with no header
    df = pp.load_ratings(tmp_ratings_file, sep="\t")
    assert set(df.columns) == {"user_id", "item_id", "rating", "timestamp"}
    assert len(df) > 0


def test_validate_ratings_df_raises_on_missing_columns():
    df = pd.DataFrame({"user_id": [1, 2], "item_id": [3, 4]})
    with pytest.raises(ValueError):
        pp.validate_ratings_df(df)


def test_preprocess_from_config_writes_processed(tmp_path, small_ratings_df):
    # integration-like test: write small ratings to file and run preprocess_from_config
    ml_dir = tmp_path / "ml100k"
    ml_dir.mkdir()
    ratings_path = ml_dir / "u.data"
    small_ratings_df.to_csv(ratings_path, sep="\t", header=False, index=False)

    cfg = {
        "data": {
            "ml100k_dir": str(ml_dir),
            "ratings_file": str(ratings_path),
            "sep": "\t",
            "min_user_interactions": 1,
            "min_item_interactions": 1,
        }
    }

    meta = pp.preprocess_from_config(cfg, write_config_back=False)
    # meta must contain keys and processed files must exist
    assert "n_users" in meta and "n_items" in meta
    processed_dir = meta["processed_dir"]
    assert os.path.exists(os.path.join(processed_dir, "train.csv"))
    assert os.path.exists(os.path.join(processed_dir, "val.csv"))
    assert os.path.exists(os.path.join(processed_dir, "test.csv"))
    # user2idx/item2idx jsons exist
    assert os.path.exists(os.path.join(processed_dir, "user2idx.json"))
    assert os.path.exists(os.path.join(processed_dir, "item2idx.json"))
