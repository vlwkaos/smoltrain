import os
import platform
import toml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def data_dir() -> Path:
    if platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "smoltrain"


def task_dir(name: str) -> Path:
    return data_dir() / name


def dataset_path(name: str) -> Path:
    return task_dir(name) / "dataset.jsonl"


def model_dir(name: str) -> Path:
    return task_dir(name) / "model"


def onnx_dir(name: str) -> Path:
    return task_dir(name) / "onnx"


def onnx_path(name: str) -> Path:
    return onnx_dir(name) / "model_int8.onnx"


def tokenizer_path(name: str) -> Path:
    return onnx_dir(name) / "tokenizer.json"


def config_path(name: str) -> Path:
    return task_dir(name) / "smoltrain.toml"


def socket_path(name: str) -> Path:
    return task_dir(name) / "serve.sock"


@dataclass
class TaskConfig:
    name: str
    goal: str
    classes: list
    class_descriptions: dict = field(default_factory=dict)
    base_model: str = "distilbert-base-uncased"
    supervisor: str = "claude-oauth"
    supervisor_model: str = "claude-haiku-4-5"
    n_per_class: int = 150
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 16
    max_seq_len: int = 128


def load(name: str) -> TaskConfig:
    path = config_path(name)
    if not path.exists():
        raise FileNotFoundError(f"No config at {path}. Run: smoltrain new {name}")
    raw = toml.loads(path.read_text())
    task = raw["task"]
    gen = raw.get("gen", {})
    train = raw.get("train", {})
    base = raw.get("base_model", {})
    return TaskConfig(
        name=task["name"],
        goal=task["goal"],
        classes=task["classes"],
        class_descriptions=task.get("class_descriptions", {}),
        base_model=base.get("repo", "distilbert-base-uncased"),
        supervisor=gen.get("supervisor", "claude-oauth"),
        supervisor_model=gen.get("supervisor_model", "claude-haiku-4-5"),
        n_per_class=gen.get("n_per_class", 150),
        epochs=train.get("epochs", 3),
        lr=train.get("lr", 2e-5),
        batch_size=train.get("batch_size", 16),
        max_seq_len=train.get("max_seq_len", 128),
    )


def new_task(name: str, classes: list, goal: str):
    d = task_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "task": {"name": name, "goal": goal, "classes": classes},
        "base_model": {"repo": "distilbert-base-uncased"},
        "gen": {
            "supervisor": "claude-oauth",
            "supervisor_model": "claude-haiku-4-5",
            "n_per_class": 150,
        },
        "train": {"epochs": 3, "lr": 2e-5, "batch_size": 16, "max_seq_len": 128},
        "serve": {"socket": str(socket_path(name))},
    }
    config_path(name).write_text(toml.dumps(cfg))
