import json
import os
from typing import List, Dict, Any
import torch
from ts.torch_handler.base_handler import BaseHandler

# предполагается, что мы передаём model class в extra-files
# и импортируем как model_loader.build_model_from_cfg
try:
    # модельный код будет перечислен в extra-files и доступен по имени model.py
    import model as model_loader  # noqa
except Exception:
    model_loader = None

class RecModelHandler(BaseHandler):
    """
    Custom TorchServe handler for RecSys model (MF/NCF).
    Expects extra-files: config.json and model.py (with model builder).
    Expects serialized file to be state_dict (pytorch_model.bin).
    """

    def initialize(self, context):
        """
        context.system_properties holds model_dir and device info.
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")  # path to extracted .mar contents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load config
        cfg_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(cfg_path):
            # try alternative location
            cfg_path = os.path.join(model_dir, "artifacts", "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Reconstruct model: we expect model.py to provide build_model_from_cfg
        if model_loader is None:
            # try to import from local path if extra-files used a package name
            raise RuntimeError("model code not found. Ensure model.py is included in --extra-files")

        model_cfg = cfg.get("model", {}) or {}
        # ensure model_init_args contains n_users/n_items if present in config
        init_args = model_cfg.get("model_init_args", {}) or {}
        # instantiate model via helper
        self.model = model_loader.build_model_from_cfg(model_cfg)
        self.model.to(self.device)
        self.model.eval()

        # load state_dict
        state_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(state_path):
            state_path = os.path.join(model_dir, "artifacts", "pytorch_model.bin")
        state = torch.load(state_path, map_location=self.device)
        # if handler loaded BaseRecModel improperly, try HACK: if state contains keys, load to model
        try:
            self.model.load_state_dict(state)
        except Exception:
            # if state contained nested dict like {'model': state_dict}
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                # last resort: attempt to remove "module." prefixes
                new_state = {}
                for k, v in state.items():
                    nk = k.replace("module.", "") if k.startswith("module.") else k
                    new_state[nk] = v
                self.model.load_state_dict(new_state)

        # precompute all items tensor for fast top-K scoring (optional)
        n_items = init_args.get("n_items") or model_cfg.get("model_init_args", {}).get("n_items")
        if n_items:
            self.all_items_tensor = torch.arange(n_items, dtype=torch.long, device=self.device)
        else:
            self.all_items_tensor = None

    def preprocess(self, data):
        """
        TorchServe sends a list of records.
        record["body"] can be bytes, str, or already a dict.
        """
        if not data or not isinstance(data, list):
            raise ValueError("Invalid input format: expected a list")

        record = data[0]
        body = record.get("body")

        if body is None:
            raise ValueError("Request body is empty")

        # Case 1: body is already a dict (TorchServe parsed JSON for us)
        if isinstance(body, dict):
            payload = body

        # Case 2: body is bytes or bytearray
        elif isinstance(body, (bytes, bytearray)):
            payload = json.loads(body.decode("utf-8"))

        # Case 3: body is string
        elif isinstance(body, str):
            payload = json.loads(body)

        else:
            raise ValueError(f"Unsupported body type: {type(body)}")

        # Validate payload
        if "user" in payload:
            users = [int(payload["user"])]
        elif "users" in payload:
            users = [int(u) for u in payload["users"]]
        else:
            raise ValueError("Payload must contain 'user' or 'users'")

        top_k = int(payload.get("top_k", 10))

        return {
            "users": users,
            "top_k": top_k
        }



    def inference(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        users = inputs["users"]
        top_k = inputs["top_k"]
        results = []
        # score items for each user
        for u in users:
            u_t = torch.tensor([u], dtype=torch.long, device=self.device)
            if self.all_items_tensor is not None:
                items = self.all_items_tensor
                users_t = u_t.repeat(items.shape[0])
                with torch.no_grad():
                    scores = self.model(users_t, items).cpu().numpy()
                # get top-k indices
                topk_idx = scores.argsort()[-top_k:][::-1]
                recs = topk_idx.tolist()
            else:
                # fallback: score nothing
                recs = []
            results.append({"user": u, "recs": recs})
        return results

    def postprocess(self, inference_output):
        # just return JSON serializable structure
        return inference_output

RecModelHandler()
