from __future__ import annotations

import inspect
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class BaseRecModel(nn.Module):
    """
    Base class providing save/load helpers compatible with 'save_pretrained' pattern.
    Subclasses should implement a constructor that accepts the arguments saved in config['model_init_args']
    and/or provide their own save/load logic but can use these helpers.
    """

    def save_pretrained(
        self, save_dir: str, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model state_dict and optional config dict to disk.

        - state_dict -> <save_dir>/pytorch_model.bin
        - config (json) -> <save_dir>/config.json
        """
        _ensure_dir(save_dir)
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        logger.info(f"Saved model state_dict to {model_path}")

        if config is None:
            config = {}
        # ensure model_init_args exists so from_pretrained can recreate
        if "model_init_args" not in config:
            config["model_init_args"] = {}

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved config to {config_path}")

    @classmethod
    def from_pretrained(
        cls, load_dir: str, map_location: Optional[str] = None
    ) -> Tuple["BaseRecModel", Dict[str, Any]]:
        """
        Load model and config from disk. Returns (model, config_dict).
        Expects config.json to contain 'model_init_args' that are passed to the constructor.
        """
        config_path = os.path.join(load_dir, "config.json")
        model_path = os.path.join(load_dir, "pytorch_model.bin")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found in {load_dir} (expected config.json)"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_init_args = config.get("model_init_args", {})
        logger.info(
            f"Loading model of type {cls.__name__} with init args: {model_init_args}"
        )

        # instantiate model - subclasses must accept **model_init_args
        model = cls(**model_init_args)  # type: ignore[arg-type]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found in {load_dir} (expected pytorch_model.bin)"
            )

        map_loc = map_location if map_location is not None else None
        state = torch.load(model_path, map_location=map_loc)
        model.load_state_dict(state)
        logger.info(f"Loaded state_dict from {model_path}")

        return model, config


class MFModel(BaseRecModel):
    """
    Matrix Factorization model with optional biases.

    Args (model_init_args):
      - n_users: int
      - n_items: int
      - embedding_dim: int
      - use_bias: bool (default True)
      - clamp_preds: bool (default False)
      - min_rating: float (optional)
      - max_rating: float (optional)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        use_bias: bool = True,
        clamp_preds: bool = False,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
    ):
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.use_bias = bool(use_bias)
        self.clamp_preds = bool(clamp_preds)
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.user_emb = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim)

        # biases as embeddings of size 1
        if self.use_bias:
            self.user_bias = nn.Embedding(self.n_users, 1)
            self.item_bias = nn.Embedding(self.n_items, 1)
        else:
            self.user_bias = None
            self.item_bias = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        if self.use_bias:
            nn.init.constant_(self.user_bias.weight, 0.0)
            nn.init.constant_(self.item_bias.weight, 0.0)

    def forward(
        self, user_idx: torch.LongTensor, item_idx: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        user_idx, item_idx: LongTensor of shape (batch,)
        returns: FloatTensor shape (batch,) containing predicted scores
        """
        # embeddings shape: (batch, embedding_dim)
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        # element-wise product and sum -> dot product
        dot = torch.sum(u * v, dim=-1)  # (batch,)
        if self.use_bias:
            ub = self.user_bias(user_idx).squeeze(-1)
            ib = self.item_bias(item_idx).squeeze(-1)
            out = dot + ub + ib
        else:
            out = dot

        if self.clamp_preds:
            if (self.min_rating is None) or (self.max_rating is None):
                raise ValueError(
                    "clamp_preds=True but min_rating/max_rating not set"
                )
            out = torch.clamp(out, min=self.min_rating, max=self.max_rating)
        return out

    def predict(
        self,
        users: Sequence[int],
        items: Sequence[int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convenience wrapper to predict on lists/arrays of user/item indices.
        Returns a torch.FloatTensor on CPU by default (unless device provided).
        """
        if device is None:
            device = next(self.parameters()).device
        u = torch.tensor(users, dtype=torch.long, device=device)
        i = torch.tensor(items, dtype=torch.long, device=device)
        self.eval()
        with torch.no_grad():
            out = self.forward(u, i)
        return out.cpu()


class NCFModel(BaseRecModel):
    """
    Neural Collaborative Filtering.

    Args (model_init_args):
      - n_users: int
      - n_items: int
      - embedding_dim: int
      - hidden_dims: list[int]  (e.g. [128,64])
      - dropout: float (default 0.0)
      - clamp_preds: bool (default False)
      - min_rating, max_rating (for clamp)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        clamp_preds: bool = False,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
    ):
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = (
            list(hidden_dims) if hidden_dims is not None else [128, 64]
        )
        self.dropout = float(dropout)
        self.clamp_preds = bool(clamp_preds)
        self.min_rating = min_rating
        self.max_rating = max_rating

        # embeddings
        self.user_emb = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim)

        # MLP
        mlp_layers: List[nn.Module] = []
        input_dim = self.embedding_dim * 2
        for h in self.hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, h))
            mlp_layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0:
                mlp_layers.append(nn.Dropout(p=self.dropout))
            input_dim = h
        # final linear to scalar
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self, user_idx: torch.LongTensor, item_idx: torch.LongTensor
    ) -> torch.FloatTensor:
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        x = torch.cat([u, v], dim=-1)  # (batch, 2*emb)
        out = self.mlp(x).squeeze(-1)  # (batch,)
        if self.clamp_preds:
            if (self.min_rating is None) or (self.max_rating is None):
                raise ValueError(
                    "clamp_preds=True but min_rating/max_rating not set"
                )
            out = torch.clamp(out, min=self.min_rating, max=self.max_rating)
        return out

    def predict(
        self,
        users: Sequence[int],
        items: Sequence[int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        u = torch.tensor(users, dtype=torch.long, device=device)
        i = torch.tensor(items, dtype=torch.long, device=device)
        self.eval()
        with torch.no_grad():
            out = self.forward(u, i)
        return out.cpu()


def _filter_init_args_for_class(
    init_args: Dict[str, Any], cls
) -> Dict[str, Any]:
    """
    Keep only keys that are accepted by cls.__init__ (excluding 'self').
    """
    sig = inspect.signature(cls.__init__)
    valid_params = [
        p.name
        for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    filtered = {}
    for k, v in init_args.items():
        if k in valid_params:
            filtered[k] = v
        else:
            # try some common name mappings
            if k == "ncf_hidden" and "hidden_dims" in valid_params:
                filtered["hidden_dims"] = v
            # else silently drop unknown keys
    return filtered


# Convenience factory
def build_model_from_cfg(model_cfg: Dict[str, Any]) -> BaseRecModel:
    """
    model_cfg should contain keys such as:
      - type: "mf" or "ncf"
      - model_init_args: {n_users, n_items, ...}
    This function now sanitizes init args to avoid passing unexpected keys.
    """
    mtype = model_cfg.get("type", model_cfg.get("model_type", "mf")).lower()
    init_args = dict(model_cfg.get("model_init_args", {}) or {})

    # merge convenient top-level shortcuts
    if "embedding_dim" in model_cfg and "embedding_dim" not in init_args:
        init_args["embedding_dim"] = model_cfg["embedding_dim"]
    if "ncf_hidden" in model_cfg and "hidden_dims" not in init_args:
        init_args["hidden_dims"] = model_cfg["ncf_hidden"]
    if "dropout" in model_cfg and "dropout" not in init_args:
        init_args["dropout"] = model_cfg["dropout"]
    if "clamp_preds" in model_cfg and "clamp_preds" not in init_args:
        init_args["clamp_preds"] = model_cfg["clamp_preds"]
    if "min_rating" in model_cfg and "min_rating" not in init_args:
        init_args["min_rating"] = model_cfg["min_rating"]
    if "max_rating" in model_cfg and "max_rating" not in init_args:
        init_args["max_rating"] = model_cfg["max_rating"]

    if mtype in ("mf", "matrixfactorization", "matrix_factorization"):
        # filter init args to those accepted by MFModel
        filtered_args = _filter_init_args_for_class(init_args, MFModel)
        return MFModel(**filtered_args)
    elif mtype in ("ncf", "neural"):
        # map ncf_hidden -> hidden_dims if necessary
        if "hidden_dims" not in init_args and "ncf_hidden" in init_args:
            init_args["hidden_dims"] = init_args.pop("ncf_hidden")
        filtered_args = _filter_init_args_for_class(init_args, NCFModel)
        return NCFModel(**filtered_args)
    else:
        raise ValueError(
            f"Unknown model type: {mtype}. Supported: 'mf', 'ncf'."
        )


# Example quick smoke test when run as script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # tiny smoke
    n_users, n_items = 100, 500
    m = MFModel(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=16,
        clamp_preds=True,
        min_rating=1.0,
        max_rating=5.0,
    )
    users = torch.randint(0, n_users, (8,), dtype=torch.long)
    items = torch.randint(0, n_items, (8,), dtype=torch.long)
    preds = m(users, items)
    logger.info(
        f"MF preds shape: {preds.shape}, min={preds.min().item():.4f}, max={preds.max().item():.4f}"
    )

    m2 = NCFModel(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=8,
        hidden_dims=[32, 16],
        dropout=0.1,
    )
    preds2 = m2(users, items)
    logger.info(f"NCF preds shape: {preds2.shape}")
