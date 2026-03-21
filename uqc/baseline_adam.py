from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .operators import destroy_torch, paulis_torch, qeye_torch, tensor_torch
from .physics import UFOCostWeights


@dataclass(frozen=True)
class AdamBaselineConfig:
    dt_ns: float = 2.0
    runtime_norm_ns: float = 60.0
    eta_base_mhz: float = -200.0
    max_control_mhz: float = 20.0
    filter_bandwidth_mhz: float = 10.0
    lr: float = 3e-2
    adam_iters: int = 400
    horizons_ns: Tuple[float, ...] = (60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0)
    train_noise_std_mhz: float = 0.0
    train_noise_samples: int = 1
    seed: int = 1
    cost_weights: UFOCostWeights = UFOCostWeights()


class TorchGmonObjective:
    def __init__(self, config: AdamBaselineConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.complex128
        self.rdtype = torch.float64
        self.n_levels = 3
        self.dim = self.n_levels ** 2
        self.mhz_to_rad_per_ns = 2.0 * np.pi * 1e-3

        a = destroy_torch(self.n_levels, device=self.device, dtype=self.dtype)
        eye = qeye_torch(self.n_levels, device=self.device, dtype=self.dtype)
        self.a1 = tensor_torch(a, eye)
        self.a2 = tensor_torch(eye, a)
        self.ad1 = self.a1.conj().T
        self.ad2 = self.a2.conj().T
        self.n1 = self.ad1 @ self.a1
        self.n2 = self.ad2 @ self.a2
        I9 = tensor_torch(eye, eye)
        self.eye = I9
        self.op_coupling = self.ad2 @ self.a1 + self.ad1 @ self.a2
        eye9 = torch.eye(self.dim, dtype=self.dtype, device=self.device)
        self.drift_shape = 0.5 * (self.n1 @ (self.n1 - eye9) + self.n2 @ (self.n2 - eye9))
        self.comp_indices = torch.tensor([0, 1, 3, 4], dtype=torch.long, device=self.device)
        self.subspaces = [[0, 1, 3, 4], [2, 5, 6, 7], [8]]
        self.block_mask = torch.zeros((self.dim, self.dim), dtype=torch.bool, device=self.device)
        for idxs in self.subspaces:
            ii = torch.tensor(idxs, dtype=torch.long, device=self.device)
            self.block_mask[ii.unsqueeze(1), ii.unsqueeze(0)] = True
        self.offdiag_mask = ~self.block_mask
        self.subspace_coeffs = torch.tensor([0.0, 1.0, 2.0], dtype=self.rdtype, device=self.device)
        basis_coeffs = torch.zeros(self.dim, dtype=self.rdtype, device=self.device)
        for coeff, idxs in zip(self.subspace_coeffs.tolist(), self.subspaces):
            ii = torch.tensor(idxs, dtype=torch.long, device=self.device)
            basis_coeffs[ii] = coeff
        self.coeff_diff_matrix = basis_coeffs.unsqueeze(0) - basis_coeffs.unsqueeze(1)
        self.denom_mask = self.offdiag_mask & (~torch.isclose(self.coeff_diff_matrix, torch.zeros_like(self.coeff_diff_matrix)))
        self.paulis = self._build_paulis()

        f_sample_ghz = 1.0 / config.dt_ns
        bw_ghz = config.filter_bandwidth_mhz * 1e-3
        alpha = float(np.exp(-np.pi * bw_ghz / f_sample_ghz))
        self.filter_a1 = (1.0 - alpha) ** 2
        self.filter_b1 = -2.0 * alpha
        self.filter_b2 = alpha ** 2

    def _build_paulis(self) -> List[torch.Tensor]:
        I = torch.eye(2, dtype=self.dtype, device=self.device)
        sx, sy, sz = paulis_torch(device=self.device, dtype=self.dtype)
        singles = [I, sx, sy, sz]
        return [torch.kron(a, b) for a in singles for b in singles]

    def target_gate(self, alpha: float, gamma: float) -> torch.Tensor:
        sx, sy, sz = paulis_torch(device=self.device, dtype=self.dtype)
        XX = torch.kron(sx, sx)
        YY = torch.kron(sy, sy)
        ZZ = torch.kron(sz, sz)
        h = alpha * XX + alpha * YY + gamma * ZZ
        return torch.matrix_exp(1j * h)

    def _blockdiag(self, M: torch.Tensor) -> torch.Tensor:
        return torch.where(self.block_mask, M, torch.zeros_like(M))

    def _offdiag(self, M: torch.Tensor) -> torch.Tensor:
        return torch.where(self.offdiag_mask, M, torch.zeros_like(M))

    def _comm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A @ B - B @ A

    def h0(self, eta_mhz: torch.Tensor) -> torch.Tensor:
        return eta_mhz * self.mhz_to_rad_per_ns * self.drift_shape

    def hamiltonian(self, controls: torch.Tensor, eta_mhz: torch.Tensor) -> torch.Tensor:
        d1, d2, f1, phi1, f2, phi2, g = [controls[i] for i in range(7)]
        scale = self.mhz_to_rad_per_ns
        eta = eta_mhz * scale
        h_drift = eta * self.drift_shape
        h_det = d1 * scale * self.n1 + d2 * scale * self.n2
        h_c = g * scale * self.op_coupling
        h_drive = torch.zeros((self.dim, self.dim), dtype=self.dtype, device=self.device)
        h_drive = h_drive + 1j * f1 * scale * (self.a1 * torch.exp(1j * phi1) - self.ad1 * torch.exp(-1j * phi1))
        h_drive = h_drive + 1j * f2 * scale * (self.a2 * torch.exp(1j * phi2) - self.ad2 * torch.exp(-1j * phi2))
        return h_drift + h_det + h_c + h_drive

    def decompose_hamiltonian(self, H: torch.Tensor, eta_mhz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = self.h0(eta_mhz)
        h_rest = H - h0
        h1 = self._blockdiag(h_rest)
        h2 = h_rest - h1
        energies = self.subspace_coeffs * eta_mhz * self.mhz_to_rad_per_ns
        return h0, h1, h2, energies

    def _solve_offdiag(self, numerator: torch.Tensor, eta_mhz: torch.Tensor) -> torch.Tensor:
        denom = eta_mhz * self.mhz_to_rad_per_ns * self.coeff_diff_matrix
        denom_safe = torch.where(self.denom_mask, denom, torch.ones_like(denom))
        solved = torch.where(self.denom_mask, numerator / denom_safe, torch.zeros_like(numerator))
        return self._offdiag(solved)

    def compute_s1(self, h2: torch.Tensor, eta_mhz: torch.Tensor) -> torch.Tensor:
        return self._solve_offdiag(h2, eta_mhz)

    def compute_s2(self, h1: torch.Tensor, s1: torch.Tensor, ds1_dt: torch.Tensor, eta_mhz: torch.Tensor) -> torch.Tensor:
        rhs = self._offdiag(self._comm(h1, s1) - 1j * ds1_dt)
        return self._solve_offdiag(rhs, eta_mhz)

    def apply_filter(self, controls: torch.Tensor) -> torch.Tensor:
        # controls: [T, 7] in physical units.
        T = controls.shape[0]
        out = []
        prev1 = torch.zeros(7, dtype=self.rdtype, device=self.device)
        prev2 = torch.zeros(7, dtype=self.rdtype, device=self.device)
        for t in range(T):
            current = self.filter_a1 * controls[t] - self.filter_b1 * prev1 - self.filter_b2 * prev2
            phase = torch.remainder(current[[3, 5]], 2.0 * np.pi)
            current = current.clone()
            current[[3, 5]] = phase
            out.append(current)
            prev2 = prev1
            prev1 = current
        return torch.stack(out, dim=0)

    def raw_to_controls(self, raw: torch.Tensor) -> torch.Tensor:
        normed = torch.tanh(raw)
        controls = torch.zeros_like(normed)
        linear = torch.tensor([0, 1, 2, 4, 6], dtype=torch.long, device=self.device)
        phase = torch.tensor([3, 5], dtype=torch.long, device=self.device)
        controls[:, linear] = normed[:, linear] * self.config.max_control_mhz
        controls[:, phase] = (normed[:, phase] + 1.0) * np.pi
        return controls

    def operator_norm(self, M: torch.Tensor) -> torch.Tensor:
        return torch.linalg.svdvals(M)[0]

    def simulate_cost(self, filtered_controls: torch.Tensor, target_gate: torch.Tensor, sigma_mhz: float = 0.0) -> torch.Tensor:
        T = filtered_controls.shape[0]
        U = self.eye
        prev_s1 = None
        prev_s2 = None
        hod_hist: List[torch.Tensor] = []
        delta_hist: List[torch.Tensor] = []

        for t in range(T):
            controls = filtered_controls[t]
            eta = torch.tensor(self.config.eta_base_mhz, dtype=self.rdtype, device=self.device)
            if sigma_mhz > 0.0:
                eta = eta + torch.randn((), dtype=self.rdtype, device=self.device) * sigma_mhz
                noise = torch.randn(5, dtype=self.rdtype, device=self.device) * sigma_mhz
                noisy = controls.clone()
                noisy[0] = noisy[0] + noise[0]
                noisy[1] = noisy[1] + noise[1]
                noisy[2] = noisy[2] + noise[2]
                noisy[4] = noisy[4] + noise[3]
                noisy[6] = noisy[6] + noise[4]
                noisy[[3, 5]] = torch.remainder(noisy[[3, 5]], 2.0 * np.pi)
            else:
                noisy = controls
            H = self.hamiltonian(noisy, eta)
            U = torch.matrix_exp(-1j * H * self.config.dt_ns) @ U

            _, h1, h2, _ = self.decompose_hamiltonian(H, eta)
            s1 = self.compute_s1(h2, eta)
            ds1_dt = torch.zeros_like(s1) if prev_s1 is None else (s1 - prev_s1) / self.config.dt_ns
            s2 = self.compute_s2(h1, s1, ds1_dt, eta)
            ds2_dt = torch.zeros_like(s2) if prev_s2 is None else (s2 - prev_s2) / self.config.dt_ns
            residual = self._comm(h1, s2) + (1.0 / 3.0) * self._comm(self._comm(h2, s1), s1) - 1j * ds2_dt
            hod_eff = self._offdiag(0.5 * (residual + residual.conj().T))
            hod_hist.append(hod_eff)
            delta_hist.append(torch.abs(eta) * self.mhz_to_rad_per_ns)
            prev_s1 = s1
            prev_s2 = s2

        K = U[self.comp_indices][:, self.comp_indices]
        tr = torch.trace(K.conj().T @ target_gate)
        fidelity = (torch.abs(tr) ** 2) / (target_gate.shape[0] ** 2)

        if len(hod_hist) == 0:
            leakage_total = torch.tensor(0.0, dtype=self.rdtype, device=self.device)
        else:
            first_norm = self.operator_norm(hod_hist[0])
            last_norm = self.operator_norm(hod_hist[-1])
            boundary = first_norm / torch.clamp(delta_hist[0], min=1e-12)
            boundary = boundary + last_norm / torch.clamp(delta_hist[-1], min=1e-12)
            integral = torch.tensor(0.0, dtype=self.rdtype, device=self.device)
            if len(hod_hist) >= 3:
                prev2 = hod_hist[0]
                prev1 = hod_hist[1]
                for i in range(2, len(hod_hist)):
                    current = hod_hist[i]
                    d2 = (current - 2.0 * prev1 + prev2) / (self.config.dt_ns ** 2)
                    integral = integral + self.operator_norm(d2) * self.config.dt_ns / torch.clamp(delta_hist[i] ** 2, min=1e-12)
                    prev2 = prev1
                    prev1 = current
            leakage_total = boundary + integral

        c0 = filtered_controls[0]
        ct = filtered_controls[-1]
        boundary_raw = c0[6] ** 2 + c0[2] ** 2 + c0[4] ** 2 + ct[6] ** 2 + ct[2] ** 2 + ct[4] ** 2
        boundary_cost = self.config.cost_weights.mu * boundary_raw
        time_cost = self.config.cost_weights.kappa * ((T * self.config.dt_ns) / self.config.runtime_norm_ns)
        cost = (
            self.config.cost_weights.chi * (1.0 - fidelity.real)
            + self.config.cost_weights.beta * leakage_total.real
            + boundary_cost.real
            + time_cost
        )
        return cost.real

    def optimize_for_horizon(self, target_alpha: float, target_gamma: float, horizon_ns: float) -> Dict[str, object]:
        torch.manual_seed(self.config.seed)
        steps = int(round(horizon_ns / self.config.dt_ns))
        raw = nn.Parameter(torch.zeros((steps, 7), dtype=self.rdtype, device=self.device))
        opt = Adam([raw], lr=self.config.lr)
        target = self.target_gate(target_alpha, target_gamma)

        best_cost = float("inf")
        best_controls = None
        history: List[Dict[str, float]] = []

        for it in range(self.config.adam_iters):
            opt.zero_grad(set_to_none=True)
            controls = self.raw_to_controls(raw)
            filtered = self.apply_filter(controls)
            if self.config.train_noise_samples > 1 and self.config.train_noise_std_mhz > 0.0:
                losses = [
                    self.simulate_cost(filtered, target, sigma_mhz=self.config.train_noise_std_mhz)
                    for _ in range(self.config.train_noise_samples)
                ]
                loss = torch.stack(losses).mean()
            else:
                loss = self.simulate_cost(filtered, target, sigma_mhz=self.config.train_noise_std_mhz)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 10.0)
            opt.step()

            cost_val = float(loss.item())
            history.append({"iter": float(it), "cost": cost_val})
            if cost_val < best_cost:
                best_cost = cost_val
                best_controls = filtered.detach().cpu().numpy()

        return {
            "horizon_ns": float(horizon_ns),
            "best_cost": float(best_cost),
            "controls": np.asarray(best_controls, dtype=np.float64),
            "history": history,
        }

    def optimize(self, target_alpha: float, target_gamma: float) -> Dict[str, object]:
        results = [self.optimize_for_horizon(target_alpha, target_gamma, horizon) for horizon in self.config.horizons_ns]
        best = min(results, key=lambda x: x["best_cost"])
        return {"best": best, "all_results": results}
