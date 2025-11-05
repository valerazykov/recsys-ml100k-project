import numpy as np
from src import metrics as metrics_mod

def test_rmse_simple():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    assert metrics_mod.rmse(y_true, y_pred) == 0.0

    y_pred2 = [2.0, 2.0, 2.0]
    # MSE = ((1)^2 + 0 + (1)^2)/3 = 2/3 -> RMSE = sqrt(2/3)
    assert abs(metrics_mod.rmse(y_true, y_pred2) - np.sqrt(2/3)) < 1e-6

def test_precision_recall_ndcg_single_cases():
    recs = [1,2,3,4]
    true = {2, 99}
    assert metrics_mod.precision_at_k_single(recs, true, k=1) == 0.0
    assert metrics_mod.precision_at_k_single(recs, true, k=2) == 0.5
    assert metrics_mod.recall_at_k_single(recs, true, k=2) == 0.5

    # ndcg: if first relevant at pos 2 -> DCG = 1/log2(2+1)
    ndcg = metrics_mod.ndcg_at_k_single(recs, true, k=4)
    assert ndcg >= 0.0 and ndcg <= 1.0

def test_evaluate_ranking_full_dummy_model():
    # dummy model that returns high score for item == user
    import torch
    class Dummy(torch.nn.Module):
        def forward(self, users, items):
            return (users == items).float() * 10.0

    model = Dummy()
    users = [0,1,2]
    ground_truth = {0:{0}, 1:{1}, 2:{2}}
    res = metrics_mod.evaluate_ranking_full(model, users, ground_truth, n_items=10, k=1, batch_size=5, device="cpu")
    precision = res.get("precision", res.get("precision@k"))
    assert precision == 1.0
