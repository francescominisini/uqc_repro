from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_ALLOWED_NAMES = {"pi": math.pi}


def parse_angle_expr(expr: str) -> float:
    """Parse simple expressions like 'pi/2', '2.2', '3*pi/4'."""
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty angle expression")
    try:
        value = eval(expr, {"__builtins__": {}}, _ALLOWED_NAMES)  # noqa: S307 - controlled namespace
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid angle expression: {expr!r}") from exc
    return float(value)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class JsonlLogger:
    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path) or ".")
        self.path = path

    def write(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Expected dataclass instance, got {type(obj)!r}")


def angle_str(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")
