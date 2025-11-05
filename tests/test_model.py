import numpy as np
import torch

from src import model as model_mod


def test_build_mf_model_and_forward_predict():
    cfg = {
        "type": "mf",
        "model_init_args": {
            "n_users": 10,
            "n_items": 20,
            "embedding_dim": 8,
            "use_bias": True,
        },
    }
    m = model_mod.build_model_from_cfg(cfg)
    assert isinstance(m, model_mod.MFModel)
    # small batch
    users = torch.tensor([0, 1, 2], dtype=torch.long)
    items = torch.tensor([0, 2, 3], dtype=torch.long)
    preds = m(users, items)
    assert preds.shape == (3,)
    # predict wrapper (accepts python lists/ndarrays)
    out = m.predict([0, 1, 2], [0, 2, 3])
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 3


def test_build_ncf_hidden_mapping_and_forward():
    # pass ncf_hidden at top-level to ensure mapping to hidden_dims works
    cfg = {
        "type": "ncf",
        "embedding_dim": 6,
        "ncf_hidden": [16, 8],
        "model_init_args": {"n_users": 5, "n_items": 7},
    }
    m = model_mod.build_model_from_cfg(cfg)
    assert isinstance(m, model_mod.NCFModel)
    # hidden dims should be set
    assert m.hidden_dims == [16, 8]
    users = torch.randint(0, 5, (4,), dtype=torch.long)
    items = torch.randint(0, 7, (4,), dtype=torch.long)
    preds = m(users, items)
    assert preds.shape == (4,)


def test_clamp_preds_applies_clamping():
    # Build MF and force embeddings to large values so raw dot product huge -> clamp must apply
    cfg = {
        "type": "mf",
        "model_init_args": {
            "n_users": 3,
            "n_items": 3,
            "embedding_dim": 4,
            "use_bias": True,
            "clamp_preds": True,
            "min_rating": 1.0,
            "max_rating": 5.0,
        },
    }
    m = model_mod.build_model_from_cfg(cfg)
    # set embeddings to large positive
    with torch.no_grad():
        m.user_emb.weight.fill_(10.0)
        m.item_emb.weight.fill_(10.0)
        if m.use_bias:
            m.user_bias.weight.fill_(0.0)
            m.item_bias.weight.fill_(0.0)
    users = torch.tensor([0, 1], dtype=torch.long)
    items = torch.tensor([0, 1], dtype=torch.long)
    preds = m(users, items)
    # Because unclamped dot would be huge, after clamp preds must be within [1.0, 5.0]
    assert torch.all(preds <= 5.0 + 1e-6)
    assert torch.all(preds >= 1.0 - 1e-6)


def test_save_and_from_pretrained_roundtrip(tmp_path):
    # Create model, save with config, load back and compare a forward pass
    cfg = {
        "type": "mf",
        "model_init_args": {
            "n_users": 8,
            "n_items": 9,
            "embedding_dim": 6,
            "use_bias": True,
        },
    }
    m = model_mod.build_model_from_cfg(cfg)
    # small deterministic init for test
    torch.manual_seed(123)
    for p in m.parameters():
        if p.requires_grad:
            torch.nn.init.constant_(p, 0.01)

    save_dir = str(tmp_path / "model_save")
    # Save with explicit config so from_pretrained can instantiate correctly
    save_cfg = {"model_init_args": cfg["model_init_args"]}
    m.save_pretrained(save_dir, config=save_cfg)

    # load using classmethod
    m_loaded, loaded_cfg = model_mod.MFModel.from_pretrained(
        save_dir, map_location="cpu"
    )
    assert isinstance(m_loaded, model_mod.MFModel)
    # forward on same random inputs should be equal
    users = torch.tensor([0, 2, 3], dtype=torch.long)
    items = torch.tensor([1, 4, 5], dtype=torch.long)
    out1 = m(users, items).detach().cpu().numpy()
    out2 = m_loaded(users, items).detach().cpu().numpy()
    assert out1.shape == out2.shape
    # Values should be numerically close
    assert np.allclose(out1, out2, atol=1e-6)


def test_build_model_ignores_extra_init_args():
    # provide extra/unexpected keys in model_init_args; build should still succeed
    cfg = {
        "type": "mf",
        "model_init_args": {
            "n_users": 4,
            "n_items": 5,
            "embedding_dim": 3,
            "some_trash_key": 12345,
        },
    }
    m = model_mod.build_model_from_cfg(cfg)
    assert isinstance(m, model_mod.MFModel)
    # ensure forward runs
    users = torch.tensor([0], dtype=torch.long)
    items = torch.tensor([0], dtype=torch.long)
    _ = m(users, items)
