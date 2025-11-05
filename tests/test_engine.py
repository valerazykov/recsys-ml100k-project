import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import dataset as dataset_mod
from src import engine as engine_mod
from src import model as model_mod


def _make_rating_df(n_users=5, n_items=5, rating_value=3.0):
    rows = []
    ts = 0
    for u in range(n_users):
        for i in range(
            min(2, n_items)
        ):  # a couple of interactions per user (small)
            rows.append(
                {
                    "user_idx": u,
                    "item_idx": (u + i) % n_items,
                    "rating": float(rating_value),
                    "timestamp": ts,
                }
            )
            ts += 1
    return pd.DataFrame(rows)


def test_train_epoch_changes_parameters_and_returns_counts():
    # small synthetic dataset
    df = _make_rating_df(n_users=5, n_items=5, rating_value=3.0)
    ds = dataset_mod.RatingDataset(df)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    # simple MF model on CPU
    model = model_mod.MFModel(
        n_users=5, n_items=5, embedding_dim=8, use_bias=True
    )
    device = torch.device("cpu")
    model.to(device)

    # copy param snapshot
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    res = engine_mod.train_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=device,
        epoch=0,
        model_type="mf",
        loss_fn=None,
        use_amp=False,
        max_grad_norm=None,
    )

    # train_epoch must return dict with loss and examples
    assert isinstance(res, dict)
    assert "loss" in res and "examples" in res
    assert int(res["examples"]) == len(df)

    # at least one parameter should have changed after the optimization step
    changed = False
    for n, p in model.named_parameters():
        if not torch.allclose(p.detach(), before[n], atol=1e-8):
            changed = True
            break
    assert (
        changed
    ), "Expected at least one model parameter to change after train_epoch"


def test_evaluate_rmse_perfect_prediction():
    # Build dataset where all ratings == 2.5; prepare model that predicts constant 2.5 via item_bias
    n_users, n_items = 4, 6
    df = _make_rating_df(n_users=n_users, n_items=n_items, rating_value=2.5)
    ds = dataset_mod.RatingDataset(df)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    model = model_mod.MFModel(
        n_users=n_users, n_items=n_items, embedding_dim=8, use_bias=True
    )
    # set embeddings to zero and item_bias to 2.5 -> predictions == 2.5 exactly
    with torch.no_grad():
        model.user_emb.weight.zero_()
        model.item_emb.weight.zero_()
        if model.use_bias:
            model.user_bias.weight.zero_()
            model.item_bias.weight.fill_(2.5)

    metrics = engine_mod.evaluate(
        model=model,
        dataloader=loader,
        device=torch.device("cpu"),
        model_type="mf",
        compute_full_ranking=False,
    )
    # rmse should be ~0
    assert "rmse" in metrics
    assert float(metrics["rmse"]) < 1e-6


def test_evaluate_full_ranking_precision_one(tmp_path):
    # For full ranking: create val df where each user has a single true item (item == user)
    n_users, n_items = 3, 5
    val_rows = []
    for u in range(n_users):
        true_item = u  # ensure true item exists and is < n_items
        val_rows.append(
            {
                "user_idx": u,
                "item_idx": true_item,
                "rating": 1.0,
                "timestamp": u,
            }
        )
    val_df = pd.DataFrame(val_rows)
    val_ds = dataset_mod.RatingDataset(val_df)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Use a dummy model that *deterministically* scores the true item highest:
    # forward(users, items) -> 10.0 when items == users else 0.0
    class DummyRankModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, users, items):
            # users, items are tensors on some device
            return (items == users).float() * 10.0

    model = DummyRankModel()

    metrics = engine_mod.evaluate(
        model=model,
        dataloader=val_loader,
        device=torch.device("cpu"),
        model_type="mf",
        compute_full_ranking=True,
        topk=1,
        n_items=n_items,
        eval_full_batch_size=16,
    )
    # precision@1 should be 1.0 because model places the true item first for each user
    assert "precision@k" in metrics
    assert float(metrics["precision@k"]) == 1.0


def test_save_checkpoint_creates_files(tmp_path):
    model = model_mod.MFModel(
        n_users=4, n_items=4, embedding_dim=4, use_bias=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    save_dir = str(tmp_path / "chk")
    engine_mod.save_checkpoint(
        model,
        optimizer,
        epoch=2,
        save_dir=save_dir,
        config={"model_init_args": {"n_users": 4, "n_items": 4}},
    )
    # check files
    assert os.path.exists(os.path.join(save_dir, "checkpoint_epoch_2.pth"))
    assert os.path.exists(os.path.join(save_dir, "pytorch_model.bin"))
    assert os.path.exists(os.path.join(save_dir, "config.json"))
