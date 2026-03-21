from __future__ import annotations

from functools import reduce
from typing import Iterable, Tuple

import numpy as np
import torch


ArrayLike = np.ndarray


def destroy(n: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=np.complex128)
    for i in range(1, n):
        out[i - 1, i] = np.sqrt(i)
    return out



def qeye(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.complex128)



def tensor(*ops: Iterable[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, ops)



def destroy_torch(n: int, device: torch.device | None = None, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    out = torch.zeros((n, n), dtype=dtype, device=device)
    for i in range(1, n):
        out[i - 1, i] = np.sqrt(i)
    return out



def qeye_torch(n: int, device: torch.device | None = None, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return torch.eye(n, dtype=dtype, device=device)



def tensor_torch(*ops: Iterable[torch.Tensor]) -> torch.Tensor:
    return reduce(torch.kron, ops)



def paulis() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return sx, sy, sz



def paulis_torch(device: torch.device | None = None, dtype: torch.dtype = torch.complex128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    return sx, sy, sz
