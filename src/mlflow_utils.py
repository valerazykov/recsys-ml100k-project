from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Dict, Optional
import re
import numbers

import mlflow


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_\-\. :/]")


def _sanitize_metric_name(name: str) -> str:
    """
    Make metric name MLflow-safe:
      - replace '@' with '_at_'
      - replace '%' with '_pct'
      - replace any other disallowed characters with '_'
    Allowed chars (per MLflow): alnum, underscore, dash, period, space, colon, slash.
    """
    if not isinstance(name, str):
        name = str(name)
    # handle common readable replacements first
    name = name.replace("@", "_at_").replace("%", "_pct")
    # replace remaining disallowed chars with underscore
    name = _SANITIZE_RE.sub("_", name)
    # collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # strip leading/trailing underscores/spaces
    name = name.strip("_ ").strip()
    # ensure non-empty
    return name or "metric"


def compute_file_hash(path: str, algo: str = "sha256") -> str:
    """Compute hex hash of a file's bytes."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> Optional[str]:
    """Return current git commit hash or None if git not available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def mlflow_start_run_from_cfg(cfg: Dict, experiment_name: Optional[str] = None):
    """
    Initialize MLflow experiment and start run.
    - cfg: config dict (your train config)
    - experiment_name: optional experiment name to use (default from cfg or 'default')
    Returns active run object (mlflow.active_run()).
    """
    # decide experiment name
    if experiment_name is None:
        experiment_name = cfg.get("mlflow", {}).get(
            "experiment_name", "recsys-experiment"
        )
    mlflow.set_experiment(experiment_name)

    mlflow.start_run()  # autogenerates run name/id
    # Log top-level config as params (flatten small set)
    mlflow.log_param("config_path", cfg.get("_config_path", "unknown"))
    # optional: log some important train params
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    mlflow.log_params(
        {
            "epochs": int(train_cfg.get("epochs", 0)),
            "batch_size": int(train_cfg.get("batch_size", 0)),
            "lr": float(
                train_cfg.get("lr", 0.0) or train_cfg.get("learning_rate", 0.0)
            ),
            "model_type": model_cfg.get("type", None),
            "embedding_dim": model_cfg.get("embedding_dim", None),
        }
    )

    # Git commit tag
    commit = git_commit_hash()
    if commit:
        mlflow.set_tag("git.commit", commit)

    return mlflow.active_run()


def mlflow_log_epoch_metrics(epoch: int, metrics: Dict[str, float]):
    """
    Log metrics of one epoch (with step=epoch) after sanitizing metric names
    and casting values to float where possible.
    """
    safe_metrics = {}
    for k, v in metrics.items():
        safe_key = _sanitize_metric_name(k)
        # ensure the metric value is numeric; try to cast
        if isinstance(v, numbers.Number):
            safe_val = float(v)
        else:
            try:
                safe_val = float(v)
            except Exception:
                # skip non-numeric metrics, but you may want to log them as params/tags instead
                continue
        safe_metrics[safe_key] = safe_val

    if safe_metrics:
        mlflow.log_metrics(safe_metrics, step=epoch)


def mlflow_log_artifacts_and_meta(
    model_dir: str,
    metrics_path: Optional[str] = None,
    dvc_lock_path: Optional[str] = None,
):
    """Log artifacts: model_dir (all files), metrics json, dvc.lock file. Also log dvc.lock hash as tag."""
    # log model directory as artifacts/model
    if os.path.exists(model_dir):
        mlflow.log_artifacts(model_dir, artifact_path="model")
    # log metrics file
    if metrics_path and os.path.exists(metrics_path):
        mlflow.log_artifact(metrics_path, artifact_path="metrics")
    # log dvc.lock if present
    if dvc_lock_path and os.path.exists(dvc_lock_path):
        mlflow.log_artifact(dvc_lock_path, artifact_path="dvc")
        try:
            h = compute_file_hash(dvc_lock_path, algo="sha256")
            mlflow.set_tag("dvc.lock.sha256", h)
        except Exception:
            pass


def mlflow_end_run():
    mlflow.end_run()
