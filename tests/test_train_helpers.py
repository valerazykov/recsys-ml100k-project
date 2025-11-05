import torch
from src import train as train_mod

def test_apply_overrides_simple():
    cfg = {"a": {"b": 1}, "x": 2}
    overrides = ["a.b=10", "x=3", "new.val=[1,2,3]", "flag=true"]
    out = train_mod.apply_overrides(cfg, overrides)
    assert out["a"]["b"] == 10
    assert out["x"] == 3
    assert out["new"]["val"] == [1,2,3]
    assert out["flag"] is True

def test_to_betas_tuple_variants():
    assert train_mod._to_betas_tuple([0.9, 0.999]) == (0.9, 0.999)
    assert train_mod._to_betas_tuple("0.9,0.999") == (0.9, 0.999)

def test_build_optimizer_and_scheduler():
    model = torch.nn.Linear(4,2)
    opt = train_mod.build_optimizer(model, {"name": "adam", "params": {"betas": [0.9, 0.99], "eps": 1e-8, "lr": 1e-3}})
    assert isinstance(opt, torch.optim.Optimizer)
    
    sched = train_mod.build_scheduler(opt, {"name":"step", "params":{"step_size":1, "factor":0.5}})
    # Сделаем один "шаг обучения", чтобы StepLR был корректно вызван
    x = torch.randn(2,4)
    y = torch.randn(2,2)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    
    opt.step()     # сначала optimizer
    sched.step()   # потом scheduler
